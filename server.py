import os, json, requests
from typing import List, Dict
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -------- env --------
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")
AUTO_SEED = os.getenv("AUTO_SEED", "1")  # "1" = seed on startup

# -------- clients/app --------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI()

# allow browser calls from your site
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://smbshopper1.pythonanywhere.com", "https://www.smbshopper1.pythonanywhere.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- models --------
class UpsertItem(BaseModel):
    id: int
    vector: List[float]
    payload: Dict

class SearchReq(BaseModel):
    vector: List[float]
    limit: int = 3

class Intent(BaseModel):
    name: str
    route: str
    role: str = "any"
    phrases: List[str]

class SeedReq(BaseModel):
    collection: str
    dim: int = 1024
    intents: List[Intent]

class SmartSearchReq(BaseModel):
    collection: str
    text: str
    limit: int = 3

# -------- helpers --------
def embed_jina(texts: List[str]) -> List[List[float]]:
    """Embed with Jina; minimal payload to avoid 422 'Extra inputs are not permitted'."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY not set on server")

    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }

    # Try OpenAI-style: {"model": "...", "input": ["a", "b", ...]}
    payload1 = {"model": "jina-embeddings-v3", "input": texts}
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json=payload1,
        timeout=30,
    )
    if r.status_code == 200:
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    # If schema-compliant still fails (422 etc.), try object-style: {"input":[{"text":"..."}]}
    if r.status_code == 422:
        payload2 = {"model": "jina-embeddings-v3", "input": [{"text": t} for t in texts]}
        r2 = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json=payload2,
            timeout=30,
        )
        r2.raise_for_status()
        data = r2.json()
        return [d["embedding"] for d in data["data"]]

    # Log other errors verbosely for Render logs
    print("Jina error:", r.status_code, r.text)
    r.raise_for_status()

def upsert_intents(collection: str, dim: int, intents: List[Dict]):
    # (re)create collection
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # flatten phrases
    items, texts = [], []
    for intent in intents:
        for phrase in intent["phrases"]:
            items.append({"name": intent["name"], "route": intent["route"], "role": intent["role"], "text": phrase})
            texts.append(phrase)
    # embed + upsert
    vecs = embed_jina(texts)
    points = [qm.PointStruct(id=i+1, vector=vecs[i], payload=items[i]) for i in range(len(items))]
    client.upsert(collection, points=points)
    return len(points)

# -------- endpoints --------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug_embed")
def debug_embed():
    # quick sanity check: returns embedding length (should be 1024)
    vec = embed_jina(["hello world"])[0]
    return {"ok": True, "dim": len(vec)}

# ---------- DEBUG: raw embedding endpoints ----------
@app.post("/embed_text")
def embed_text(body: Dict[str, str]):
    """
    Return the raw embedding for a given text (POST).
    Body: { "text": "..." }
    """
    text = (body or {}).get("text", "").strip()
    if not text:
        return {"error": "missing 'text'"}
    vec = embed_jina([text])[0]
    return {
        "ok": True,
        "dim": len(vec),
        "preview": vec[:20],  # first 20 numbers
        "embedding": vec
    }

@app.get("/embed_text")
def embed_text_get(text: str = Query(..., description="Text to embed")):
    """
    Return the raw embedding for a given text (GET).
    Usage: /embed_text?text=hello
    """
    t = (text or "").strip()
    if not t:
        return {"error": "missing 'text' query param"}
    vec = embed_jina([t])[0]
    return {
        "ok": True,
        "dim": len(vec),
        "preview": vec[:20],
        "embedding": vec
    }

# Optional: list all routes for a quick sanity check
@app.get("/__routes")
def __routes():
    return [r.path for r in app.routes]

@app.post("/recreate/{collection}")
def recreate(collection: str, dim: int, distance: str = "COSINE"):
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=getattr(qm.Distance, distance)),
    )
    return {"ok": True}

@app.post("/upsert/{collection}")
def upsert(collection: str, items: List[UpsertItem]):
    points = [qm.PointStruct(id=i.id, vector=i.vector, payload=i.payload) for i in items]
    client.upsert(collection_name=collection, points=points)
    return {"ok": True}

class SearchReq(BaseModel):
    vector: List[float]
    limit: int = 3

@app.post("/search/{collection}")
def search(collection: str, req: SearchReq):
    hits = client.search(collection_name=collection, query_vector=req.vector,
                         limit=req.limit, with_payload=True)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]

@app.post("/seed")
def seed(req: SeedReq):
    count = upsert_intents(req.collection, req.dim, [i.dict() for i in req.intents])
    return {"ok": True, "count": count}

@app.post("/smart_search")
def smart_search(req: SmartSearchReq):
    vec = embed_jina([req.text])[0]
    hits = client.search(collection_name=req.collection, query_vector=vec,
                         limit=req.limit, with_payload=True)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]

# -------- auto-seed on startup --------
@app.on_event("startup")
def startup_seed():
    if AUTO_SEED != "1":
        return
    intents = [
        {
            "name": "Pending Orders",
            "route": "/custneworders?tab=pending",
            "role": "customer",
            "phrases": [
                "show my pending orders",
                "what orders are still waiting",
                "orders not accepted yet",
                "pending customer orders",
            ],
        },
        {
            "name": "Accepted Orders",
            "route": "/custneworders?tab=accepted",
            "role": "customer",
            "phrases": [
                "which orders got accepted",
                "show accepted orders",
                "orders that shops accepted",
            ],
        },
        {
            "name": "New Orders (Shop)",
            "route": "/shopneworders",
            "role": "shop",
            "phrases": [
                "show new customer orders",
                "what new orders are available",
                "orders I can accept",
            ],
        },
    ]
    upsert_intents("ai_actions_v1", 1024, intents)
