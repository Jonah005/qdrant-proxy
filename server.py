import os, json
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
app = FastAPI()

class UpsertItem(BaseModel):
    id: int
    vector: list[float]
    payload: dict

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
def upsert(collection: str, items: list[UpsertItem]):
    points = [qm.PointStruct(id=i.id, vector=i.vector, payload=i.payload) for i in items]
    client.upsert(collection_name=collection, points=points)
    return {"ok": True}

class SearchReq(BaseModel):
    vector: list[float]
    limit: int = 3

@app.post("/search/{collection}")
def search(collection: str, req: SearchReq):
    hits = client.search(collection_name=collection, query_vector=req.vector,
                         limit=req.limit, with_payload=True)
    return [{"score": float(h.score), "payload": h.payload} for h in hits]
