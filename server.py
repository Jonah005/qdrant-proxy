import os, json, requests
from typing import List, Dict, Optional

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

# -------------------- ENV --------------------
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
JINA_API_KEY = os.getenv("JINA_API_KEY", "").strip()
AUTO_SEED = os.getenv("AUTO_SEED", "1").strip()

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
    route: Optional[str] = None
    response: Optional[str] = None
    kind: str = Field(default="nav")  # "nav" or "faq"

class SeedReq(BaseModel):
    collection: str
    dim: int = 1024
    intents: List[Intent]

class SmartSearchReq(BaseModel):
    collection: str
    text: str
    limit: int = 3
    kind: Optional[str] = None
    role: Optional[str] = None

class AssistReq(BaseModel):
    collection: str
    text: str
    role: Optional[str] = None
    limit: int = 5
    threshold: float = 0.80
    force_arbiter: bool = False
    # Lightweight context to maintain coherence across turns
    context: Optional[str] = None                    # a single short line like "We were discussing pending vs accepted"
    history: Optional[List[Dict[str, str]]] = None   # [{ "role": "user|assistant", "content": "..." }]
    prior_candidates: Optional[List[Dict]] = None    # previously returned "candidates" from /assist

# -------------------- HELPERS --------------------
def embed_jina(texts: List[str]) -> List[List[float]]:
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
        return [d["embedding"] for d in r.json()["data"]]

    if r.status_code == 422:
        r2 = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers=headers,
            json={"model": "jina-embeddings-v3", "input": [{"text": t} for t in texts]},
            timeout=30,
        )
        r2.raise_for_status()
        return [d["embedding"] for d in r2.json()["data"]]

    print("Jina error:", r.status_code, r.text)
    r.raise_for_status()

def upsert_intents(collection: str, dim: int, intents: List[Dict]) -> int:
    client.recreate_collection(
        collection_name=collection,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    items, texts = [], []
    for intent in intents:
        for phrase in intent["phrases"]:
            items.append({
                "name": intent.get("name"),
                "role": intent.get("role", "any"),
                "kind": intent.get("kind", "nav"),
                "route": intent.get("route"),
                "response": intent.get("response"),
                "text": phrase,
            })
            texts.append(phrase)

    vecs = embed_jina(texts)
    points = [qm.PointStruct(id=i+1, vector=vecs[i], payload=items[i]) for i in range(len(items))]
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
    if not hits:
        return True
    top = hits[0]["score"]
    if top < threshold:
        return True
    if len(hits) >= 2 and (top - hits[1]["score"]) < 0.08:
        return True
    kinds = {(h["payload"].get("kind") or "nav") for h in hits[:3]}
    if len(kinds) > 1:
        return True
    return False

def _clip_history(history: Optional[List[Dict[str, str]]], max_turns: int = 4) -> List[Dict[str, str]]:
    """Keep only the last N turns (user/assistant messages)."""
    if not history:
        return []
    clean = [m for m in history if m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str)]
    return clean[-max_turns:]

def _compact_candidates(hits: List[Dict]) -> List[Dict]:
    """Minify candidate payloads sent to the LLM."""
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
    return compact

def _topic_hint_from_hits(hits: List[Dict]) -> str:
    """A tiny hint to help the arbiter resolve pronouns like 'those two'."""
    names = []
    kinds = set()
    for h in hits[:3]:
        p = h["payload"]
        if p.get("name"):
            names.append(p["name"])
        if p.get("kind"):
            kinds.add(p["kind"])
    hint = ""
    if names:
        hint += "Candidates include: " + ", ".join(names) + ". "
    if kinds:
        hint += "Kinds present: " + ", ".join(sorted(kinds)) + "."
    return hint.strip()

def _safe_json_loads(s: str) -> Optional[Dict]:
    """Tolerate accidental code fences around JSON."""
    try:
        return json.loads(s)
    except Exception:
        pass
    # strip common fences if present
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = s2.strip("`")
        # after stripping backticks, try again
        try:
            return json.loads(s2)
        except Exception:
            pass
    return None

def call_arbiter(
    user_query: str,
    hits: List[Dict],
    context: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None
) -> Optional[Dict]:
    if not OPENROUTER_API_KEY:
        return None

    compact = _compact_candidates(hits)
    topic_hint = _topic_hint_from_hits(hits)

    system = (
        "You are a precise, fast support arbiter. "
        "Use ONLY the provided candidates—do not invent routes or answers. "
        "If intent is unclear, return ONE short clarification question. "
        "If navigation, pick exactly one candidate route. "
        "If FAQ, return the candidate's 'response'. "
        "Return strict JSON: "
        "{\"mode\":\"clarify|navigate|answer\",\"message\":\"string\",\"route\":\"string|null\",\"picked_index\":0}"
    )

    messages = [{"role": "system", "content": system}]
    for m in _clip_history(history, max_turns=4):
        messages.append({"role": m["role"], "content": m["content"]})
    if context:
        messages.append({"role": "user", "content": f"Context: {context}"})
    if topic_hint:
        messages.append({"role": "user", "content": f"Topic hint: {topic_hint}"})

    messages.append({"role": "user", "content": f"User query: {user_query}"})
    messages.append({"role": "user", "content": f"Candidates:\n{json.dumps(compact, ensure_ascii=False)}"})

    try:
        r = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
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
        data = _safe_json_loads(content) or {}
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
def embed_text_get(text: str = Query(...)):
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
        hits = [h for h in hits if (h["payload"].get("kind") or "nav").lower() == req.kind.lower()]
    hits = _prefer_role(hits, req.role)
    return hits[:req.limit]

# -------------------- High-level ASSIST (context-aware) --------------------
@app.post("/assist")
def assist(req: AssistReq):
    # If the client passed a previous shortlist, sanitize and reuse for continuity
    if req.prior_candidates:
        hits = []
        for h in req.prior_candidates[: req.limit]:
            # Accept both raw dicts or already-public hits
            if "payload" in h and "score" in h:
                hits.append({"score": float(h["score"]), "payload": dict(h["payload"])})
        # Fall back to fresh search if the provided list is empty
        if not hits:
            req.prior_candidates = None

    if not req.prior_candidates:
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

    if not hits:
        return {
            "mode": "clarify",
            "message": "I couldn’t match that. Do you want to open a page or ask a question about orders?",
            "route": None,
            "picked": None,
            "candidates": hits,
        }

    if req.force_arbiter or _ambiguous(hits, req.threshold):
        arb = call_arbiter(req.text, hits, context=req.context, history=req.history)
        if arb:
            picked = None
            if isinstance(arb.get("picked_index"), int):
                idx = arb["picked_index"]
                if 0 <= idx < len(hits):
                    picked = hits[idx]["payload"]
            route = arb.get("route") or (picked and picked.get("route"))
            return {
                "mode": arb.get("mode", "clarify"),
                "message": (arb.get("message") or "Can you clarify?").strip(),
                "route": route,
                "picked": picked,
                "candidates": hits,
            }

    top = hits[0]["payload"]
    if (top.get("kind") or "nav") == "nav" and top.get("route"):
        return {
            "mode": "navigate",
            "message": f"Taking you to {top.get('name')}.",
            "route": top.get("route"),
            "picked": top,
            "candidates": hits,
        }
    if (top.get("kind") or "faq") == "faq" and top.get("response"):
        return {
            "mode": "answer",
            "message": top.get("response"),
            "route": None,
            "picked": top,
            "candidates": hits,
        }

    return {
        "mode": "fallback",
        "message": "I found a likely match but cannot complete the action automatically.",
        "route": top.get("route"),
        "picked": top,
        "candidates": hits,
    }

# -------------------- AUTO-SEED --------------------
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
