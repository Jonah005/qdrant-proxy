import os, json, requests
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")  # optional, for embeddings

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI()

# ---------- models ----------
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
    dim: int = 1024     # for jina-embeddings-v3
    intents: List[Intent]

class SmartSearchReq(BaseModel):
    collection: str
    text: str
    limit: int = 3

# ---------- helpers ----------
def embed_jina(texts: List[str]) -> List[List[float]]:
    if not JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY not set on server")
    r = requests.post(
        "https://api.jina.ai/v1/embeddings",
        headers={"Authorization": f"Bearer {JINA_API_KEY}",
                 "Content-Type": "application/json"},
        data=json.dumps({"model": "jina-embeddings-v3",
                         "input": texts,
                         "encoding_format": "float"}),
        timeout=30
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]

# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/recreate/{collection}")
def recreate(collection: str, dim: int, distance: str = "COSINE"):
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=getattr(qm.Distance, distance))
    )
    return {"ok": True}

@app.post("/upsert/{collection}")
def upsert(collection: str, items: List[UpsertItem]):
    points = [qm.PointStruct(id=i.id, vector=i.vector, payload=i.payload) for i in items]
    client.upsert(collection_name=collection, points=points)
    return {"ok": True}

@app.post("/search/{collection}")
def search(collection: str, req: SearchReq):
    hits = client.search(collection_name=collection, query_vector=req.vector,
                         limit=req.limit, with_payload=True)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]

@app.post("/seed")
def seed(req: SeedReq):
    # (re)create collection with right dim
    client.recreate_collection(
        collection_name=req.collection,
        vectors_config=qm.VectorParams(size=req.dim, distance=qm.Distance.COSINE),
    )
    # flatten phrases and embed
    items, texts = [], []
    for intent in req.intents:
        for phrase in intent.phrases:
            items.append({"name": intent.name, "route": intent.route,
                          "role": intent.role, "text": phrase})
            texts.append(phrase)
    vecs = embed_jina(texts)
    # upsert
    points = [
        qm.PointStruct(
            id=i+1,
            vector=vecs[i],
            payload={"name": itm["name"], "route": itm["route"],
                     "role": itm["role"], "text": itm["text"]}
        )
        for i, itm in enumerate(items)
    ]
    client.upsert(req.collection, points=points)
    return {"ok": True, "count": len(points)}

@app.post("/smart_search")
def smart_search(req: SmartSearchReq):
    vec = embed_jina([req.text])[0]
    hits = client.search(collection_name=req.collection, query_vector=vec,
                         limit=req.limit, with_payload=True)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]
