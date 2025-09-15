import os
import json
from typing import List, Dict, Optional

import requests
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -------------------- ENV --------------------
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
AUTO_SEED = os.getenv("AUTO_SEED", "1").strip()  # "1" = seed on startup

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
ARBITER_MODEL = os.getenv(
    "ARBITER_MODEL",
    "mistralai/mistral-small-3.2-24b-instruct:free",
).strip()

# -------------------- CLIENTS / APP --------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://smbshopper1.pythonanywhere.com",
        "https://www.smbshopper1.pythonanywhere.com",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- MODELS --------------------
class UpsertItem(BaseModel):
    id: int
    vector: List[float]
    payload: Dict

class SearchReq(BaseModel):
    vector: List[float]
    limit: int = 3

class Intent(BaseModel):
    name: str
    role: str = "any"
    phrases: List[str]
    route: Optional[str] = None               # for navigation
    response: Optional[str] = None            # for FAQs / answers
    kind: str = Field(default="nav")          # "nav" or "faq"

class SeedReq(BaseModel):
    collection: str
    dim: int = 1024
    intents: List[Intent]

class SmartSearchReq(BaseModel):
    collection: str
    text: str
    limit: int = 3
    kind: Optional[str] = None                # "nav" | "faq" (optional filter)
    role: Optional[str] = None                # prefer results matching role

class AssistReq(BaseModel):
    collection: str
    text: str
    role: Optional[str] = None
    limit: int = 5
    threshold: float = 0.80                   # ambiguity threshold
    # If you want to force an arbiter pass (even for clear queries)
    force_arbiter: bool = False

# -------------------- HELPERS --------------------
def embed_jina(texts: List[str]) -> List[List[float]]:
    """Embed with Jina; minimal payload to avoid 422s."""
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY not set on server")

    headers = {"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"}

    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers=headers,
        json={"model": "jina-embeddings-v3", "input": texts},
        timeout=30,
    )
    if r.status_code == 200:
        data = r.json()
        return [d["embedding"] for d in data["data"]]

    if r.status_code == 422:
        r2 = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={"model": "jina-embeddings-v3", "input": [{"text": t} for t in texts]},
            timeout=30,
        )
        r2.raise_for_status()
        data = r2.json()
        return [d["embedding"] for d in data["data"]]

    print("Jina error:", r.status_code, r.text)
    r.raise_for_status()

def upsert_intents(collection: str, dim: int, intents: List[Dict]) -> int:
    """(Re)create collection, embed phrases, upsert points."""
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    items, texts = [], []
    for intent in intents:
        name = intent.get("name")
        role = intent.get("role", "any")
        kind = intent.get("kind", "nav")
        route = intent.get("route")
        response = intent.get("response")

        for phrase in intent["phrases"]:
            items.append({
                "name": name,
                "role": role,
                "kind": kind,
                "route": route,
                "response": response,
                "text": phrase,
            })
            texts.append(phrase)

    vecs = embed_jina(texts)
    points = [
        qm.PointStruct(id=i + 1, vector=vecs[i], payload=items[i])
        for i in range(len(items))
    ]
    client.upsert(collection, points=points)
    return len(points)

def _prefer_role(hits: List[Dict], role: Optional[str]) -> List[Dict]:
    if not role:
        return sorted(hits, key=lambda h: h["score"], reverse=True)
    return sorted(
        hits,
        key=lambda h: ((h["payload"].get("role") == role), h["score"]),
        reverse=True,
    )

def _hits_to_public(hits) -> List[Dict]:
    return [{"score": float(h.score), "payload": h.payload} for h in hits]

def _ambiguous(hits: List[Dict], threshold: float) -> bool:
    """Heuristic to decide if we should ask the LLM to arbitrate."""
    if not hits:
        return True
    top = hits[0]["score"]
    if top < threshold:
        return True
    if len(hits) >= 2 and (top - hits[1]["score"]) < 0.08:
        return True
    # mixed kinds in the top-3?
    kinds = { (h["payload"].get("kind") or "nav") for h in hits[:3] }
    if len(kinds) > 1:
        return True
    return False

def call_arbiter(user_query: str, hits: List[Dict]) -> Optional[Dict]:
    """
    Calls the LLM arbiter via OpenRouter to:
      - clarify (ask a question),
      - navigate (pick a route),
      - answer (use FAQ response).

    Returns a dict matching the schema below, or None on failure.

    Schema to follow:
      {
        "mode": "clarify" | "navigate" | "answer",
        "message": "short message for the user",
        "route": "/path-or-null",
        "picked_index": 0  # the index of chosen candidate (optional)
      }
    """
    if not OPENROUTER_API_KEY:
        return None

    # Only pass minimal needed data to the LLM
    compact = []
    for i, h in enumerate(hits[:5]):
        p = h["payload"]
        compact.append({
            "idx": i,
            "score": round(h["score"], 4),
            "name": p.get("name"),
            "kind": p.get("kind"),
            "role": p.get("role"),
            "route": p.get("route"),
            "response": p.get("response"),
            "text_example": p.get("text"),
        })

    system = (
        "You are a fast, precise support arbiter. "
        "You must ONLY use the provided candidates—do not invent routes or answers. "
        "If the user intent is unclear, return a single short clarification question. "
        "If it is a navigation request, pick exactly one route from the candidates. "
        "If it is an FAQ request, return a concise answer using the candidate's 'response'. "
        "Output strict JSON matching this schema:\n"
        "{"
        "\"mode\":\"clarify|navigate|answer\","
        "\"message\":\"string\","
        "\"route\":\"string|null\","
        "\"picked_index\":0"
        "}"
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"User query: {user_query}"},
        {"role": "user", "content": f"Candidates:\n{json.dumps(compact, ensure_ascii=False)}"},
    ]

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                # Optional attribution headers:
                "HTTP-Referer": "https://qdrant-proxy.onrender.com",
                "X-Title": "SMB Shopper Arbiter",
            },
            json={
                "model": ARBITER_MODEL,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "max_tokens": 200,
                "temperature": 0.2,
            },
            timeout=20,
        )
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        data = json.loads(content)
        # light validation
        if data.get("mode") in {"clarify", "navigate", "answer"}:
            return data
        return None
    except Exception as e:
        print("Arbiter error:", repr(e))
        return None

# -------------------- ENDPOINTS --------------------
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug_embed")
def debug_embed():
    vec = embed_jina(["hello world"])[0]
    return {"ok": True, "dim": len(vec)}

@app.post("/embed_text")
def embed_text(body: Dict[str, str]):
    text = (body or {}).get("text", "").strip()
    if not text:
        return {"error": "missing 'text'"}
    vec = embed_jina([text])[0]
    return {"ok": True, "dim": len(vec), "preview": vec[:20], "embedding": vec}

@app.get("/embed_text")
def embed_text_get(text: str = Query(..., description="Text to embed")):
    t = (text or "").strip()
    if not t:
        return {"error": "missing 'text' query param"}
    vec = embed_jina([t])[0]
    return {"ok": True, "dim": len(vec), "preview": vec[:20], "embedding": vec}

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

@app.post("/search/{collection}")
def search(collection: str, req: SearchReq):
    raw_hits = client.search(
        collection_name=collection,
        query_vector=req.vector,
        limit=req.limit,
        with_payload=True,
    )
    return _hits_to_public(raw_hits)

@app.post("/seed")
def seed(req: SeedReq):
    count = upsert_intents(req.collection, req.dim, [i.dict() for i in req.intents])
    return {"ok": True, "count": count}

@app.post("/smart_search")
def smart_search(req: SmartSearchReq):
    vec = embed_jina([req.text])[0]
    raw_hits = client.search(
        collection_name=req.collection,
        query_vector=vec,
        limit=max(req.limit, 10),
        with_payload=True,
    )
    hits = _hits_to_public(raw_hits)

    if req.kind:
        k = req.kind.lower()
        hits = [h for h in hits if (h["payload"].get("kind") or "nav").lower() == k]

    hits = _prefer_role(hits, req.role)
    return hits[: req.limit]

# -------------------- High-level ASSIST endpoint --------------------
@app.post("/assist")
def assist(req: AssistReq):
    """
    Orchestrated helper:
      1) Embed + search
      2) Ambiguity check
      3) If ambiguous (or force_arbiter), call LLM arbiter
      4) Return a normalized object {mode,message,route,picked,candidates}
    """
    vec = embed_jina([req.text])[0]
    raw_hits = client.search(
        collection_name=req.collection,
        query_vector=vec,
        limit=max(req.limit, 10),
        with_payload=True,
    )
    hits = _hits_to_public(raw_hits)
    hits = _prefer_role(hits, req.role)
    hits = hits[: req.limit]

    # Nothing found → ask a generic clarification (no LLM needed)
    if not hits:
        return {
            "mode": "clarify",
            "message": "I couldn’t match that. Do you want to open a page (e.g., Pending/Accepted orders) or ask a question about orders?",
            "route": None,
            "picked": None,
            "candidates": hits,
        }

    # If forced or ambiguous, try arbiter
    if req.force_arbiter or _ambiguous(hits, req.threshold):
        arb = call_arbiter(req.text, hits)
        if arb:
            picked = None
            if isinstance(arb.get("picked_index"), int):
                idx = arb["picked_index"]
                if 0 <= idx < len(hits):
                    picked = hits[idx]["payload"]
            # If mode=navigate but LLM forgot route, try to lift from picked
            route = arb.get("route")
            if arb.get("mode") == "navigate" and not route and picked:
                route = picked.get("route")
            return {
                "mode": arb.get("mode", "clarify"),
                "message": arb.get("message", "").strip() or "Can you clarify?",
                "route": route,
                "picked": picked,
                "candidates": hits,
            }
        # Arbiter failed → fall back

    # Clear enough → pick top hit deterministically
    top = hits[0]["payload"]
    if (top.get("kind") or "nav") == "nav" and top.get("route"):
        return {
            "mode": "navigate",
            "message": f"Taking you to {top.get('name')}.",
            "route": top.get("route"),
            "picked": top,
            "candidates": hits,
        }
    # FAQ answer path
    if (top.get("kind") or "faq") == "faq" and top.get("response"):
        return {
            "mode": "answer",
            "message": top.get("response"),
            "route": None,
            "picked": top,
            "candidates": hits,
        }

    # Fallback if payload incomplete
    return {
        "mode": "fallback",
        "message": "I found a likely match but cannot complete the action automatically.",
        "route": top.get("route"),
        "picked": top,
        "candidates": hits,
    }

# -------------------- AUTO-SEED (nav intents) --------------------
@app.on_event("startup")
def startup_seed():
    if AUTO_SEED != "1":
        return
    intents = [
        {
            "name": "Pending Orders",
            "route": "/custneworders?tab=pending",
            "role": "customer",
            "kind": "nav",
            "phrases": [
                "show my pending orders",
                "what orders are still waiting",
                "orders not accepted yet",
                "pending customer orders",
                "where to see incomplete orders",
            ],
        },
        {
            "name": "Accepted Orders",
            "route": "/custneworders?tab=accepted",
            "role": "customer",
            "kind": "nav",
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
            "kind": "nav",
            "phrases": [
                "show new customer orders",
                "what new orders are available",
                "orders I can accept",
            ],
        },
    ]
    upsert_intents("ai_actions_v1", 1024, intents)
