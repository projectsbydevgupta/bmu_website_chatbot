"""
BMU Retrieval Microservice
==========================
Standalone FastAPI microservice providing:
- Local SentenceTransformer embedding inference
- ChromaDB Cloud vector similarity search
- Hybrid BM25-style keyword + dense semantic ranking
- Health check and monitoring endpoint
"""

import os
import re
import time
from typing import Any, Dict, List, Optional

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from voyage_client import voyage_client


# Load environment variables
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Collections
PROD_COLLECTION_NAME = "bmu_summaries"
VOYAGE_COLLECTION_NAME = os.getenv("VOYAGE_COLLECTION_NAME", "bmu_voyage_knowledge")
DEFAULT_RETRIEVAL_ENGINE = os.getenv("RETRIEVAL_ENGINE", "minilm").lower()

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BMU Semantic Retrieval Microservice",
    description="Dedicated microservice for sentence embeddings, ChromaDB search, hybrid ranking, and Voyage AI migration.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_prewarm():
    """Pre-warms ChromaDB connection and embedding model on boot so first request is instant."""
    try:
        print("[Retrieval Service] Pre-warming embedding model and Chroma collection...")
        get_prod_collection()
        get_embedder()
        print("[Retrieval Service] Pre-warming complete - service is 100% ready!")
    except Exception as exc:
        print(f"[Retrieval Service] Pre-warming warning: {exc}")


# ---------------------------------------------------------------------------
# Lazy Loaded Resources
# ---------------------------------------------------------------------------
_chroma_client = None
_prod_collection = None
_voyage_collection = None
_embedding_model = None


def get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
            raise RuntimeError("Missing required ChromaDB configuration in environment (.env).")
        _chroma_client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    return _chroma_client


def get_prod_collection():
    global _prod_collection
    if _prod_collection is None:
        client = get_chroma_client()
        _prod_collection = client.get_or_create_collection(name=PROD_COLLECTION_NAME)
    return _prod_collection


def get_voyage_collection():
    global _voyage_collection
    if _voyage_collection is None:
        client = get_chroma_client()
        _voyage_collection = client.get_or_create_collection(name=VOYAGE_COLLECTION_NAME)
    return _voyage_collection


def get_embedder():
    """Returns a lightweight embedder object with an .encode(texts) interface.
    Uses Chroma's built-in ONNXMiniLM_L6_V2 (ONNX Runtime, 0 MB PyTorch, ~90 MB RAM),
    falling back to SentenceTransformer only if explicitly installed locally."""
    global _embedding_model
    if _embedding_model is None:
        try:
            import chromadb.utils.embedding_functions as ef
            import numpy as np

            class _OnnxEmbedderWrapper:
                def __init__(self):
                    self._fn = ef.ONNXMiniLM_L6_V2()

                def encode(self, texts, **kwargs):
                    if isinstance(texts, str):
                        texts = [texts]
                    res = self._fn(list(texts))
                    return np.array(res)

            _embedding_model = _OnnxEmbedderWrapper()
            print("[Embedder] Initialized lightweight ONNX MiniLM embedder (Zero PyTorch overhead, ~90MB RAM).")
        except Exception as exc:
            print(f"[Embedder] Notice: ONNX loader exception ({exc}). Attempting SentenceTransformer fallback.")
            try:
                from sentence_transformers import SentenceTransformer
                _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as st_exc:
                raise RuntimeError(f"Could not initialize embedding engine: {st_exc}") from st_exc
    return _embedding_model



def get_chroma_and_embedder():
    """Backward compatibility helper for MiniLM production engine."""
    return get_prod_collection(), get_embedder()


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(default=8, ge=1, le=50, description="Maximum number of ranked results to return")
    engine: Optional[str] = Field(
        default=None,
        description="'minilm' (production bmu_summaries) or 'voyage' (migration bmu_voyage_knowledge). Defaults to env RETRIEVAL_ENGINE.",
    )


class RankedDocument(BaseModel):
    summary: str
    url: str
    score: float
    semantic_sim: float
    keyword_score: float
    title: Optional[str] = ""
    engine: Optional[str] = "minilm"


class RetrieveResponse(BaseModel):
    query: str
    count: int
    latency_ms: float
    engine: str
    collection: str
    results: List[RankedDocument]


class HealthResponse(BaseModel):
    status: str
    service: str
    default_engine: str
    minilm_model: str
    minilm_collection: str
    minilm_document_count: Optional[int] = None
    voyage_model: str
    voyage_collection: str
    voyage_document_count: Optional[int] = None
    voyage_configured: bool
    latency_ms: float
    # Backward compatibility fields
    model: str
    collection: str
    document_count: Optional[int] = None


# ---------------------------------------------------------------------------
# Ranking Logic
# ---------------------------------------------------------------------------
def _keyword_score(query: str, text: str) -> float:
    """Calculates word overlap ratio between query and candidate text."""
    query_words = set(re.findall(r"\w+", query.lower()))
    text_words = set(re.findall(r"\w+", text.lower()))
    if not query_words:
        return 0.0
    overlap = query_words.intersection(text_words)
    return len(overlap) / len(query_words)


# ---------------------------------------------------------------------------
# Microservice Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    start_t = time.perf_counter()
    prod_count = None
    voyage_count = None
    status_str = "ok"

    try:
        prod_col = get_prod_collection()
        prod_count = prod_col.count()
    except Exception as exc:
        status_str = f"prod_degraded: {exc}"

    try:
        voyage_col = get_voyage_collection()
        voyage_count = voyage_col.count()
    except Exception as exc:
        pass

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return HealthResponse(
        status=status_str,
        service="BMU Retrieval Microservice",
        default_engine=DEFAULT_RETRIEVAL_ENGINE,
        minilm_model=EMBEDDING_MODEL_NAME,
        minilm_collection=PROD_COLLECTION_NAME,
        minilm_document_count=prod_count,
        voyage_model=voyage_client.model_name,
        voyage_collection=VOYAGE_COLLECTION_NAME,
        voyage_document_count=voyage_count,
        voyage_configured=voyage_client.is_configured,
        latency_ms=latency_ms,
        # Backward compatibility
        model=EMBEDDING_MODEL_NAME,
        collection=PROD_COLLECTION_NAME,
        document_count=prod_count,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    start_t = time.perf_counter()
    clean_query = req.query.strip().lower()
    if not clean_query:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty or whitespace.",
        )

    engine = (req.engine or DEFAULT_RETRIEVAL_ENGINE).lower()
    if engine not in ["minilm", "voyage"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported retrieval engine '{engine}'. Use 'minilm' or 'voyage'.",
        )

    try:
        ranked: List[RankedDocument] = []
        n_fetch = min(max(req.top_k, 8), 25)

        if engine == "voyage":
            # =========================================================
            # Voyage AI Engine (1024-dim, bmu_voyage_knowledge)
            # =========================================================
            if not voyage_client.is_configured:
                raise RuntimeError("Voyage AI client is not configured in .env.")
            col = get_voyage_collection()
            query_emb = [voyage_client.embed_query(clean_query)]
            raw_results = col.query(query_embeddings=query_emb, n_results=n_fetch)
            active_collection = VOYAGE_COLLECTION_NAME

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                sem_sim = max(0.0, 1.0 - dist)
                kw_sc = _keyword_score(clean_query, doc)
                final_sc = (0.68 * sem_sim) + (0.32 * kw_sc)

                ranked.append(
                    RankedDocument(
                        summary=doc,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        score=round(float(final_sc), 4),
                        semantic_sim=round(float(sem_sim), 4),
                        keyword_score=round(float(kw_sc), 4),
                        engine="voyage",
                    )
                )

        else:
            # =========================================================
            # MiniLM Production Engine (384-dim, bmu_summaries)
            # =========================================================
            col, embedder = get_chroma_and_embedder()
            query_emb = embedder.encode([clean_query]).tolist()
            raw_results = col.query(query_embeddings=query_emb, n_results=n_fetch)
            active_collection = PROD_COLLECTION_NAME

            documents = raw_results.get("documents", [[]])[0]
            metadatas = raw_results.get("metadatas", [[]])[0]
            distances = raw_results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                sem_sim = max(0.0, 1.0 - dist)
                kw_sc = _keyword_score(clean_query, doc)
                final_sc = (0.68 * sem_sim) + (0.32 * kw_sc)

                ranked.append(
                    RankedDocument(
                        summary=doc,
                        url=meta.get("url", ""),
                        title=meta.get("title", ""),
                        score=round(float(final_sc), 4),
                        semantic_sim=round(float(sem_sim), 4),
                        keyword_score=round(float(kw_sc), 4),
                        engine="minilm",
                    )
                )

        ranked.sort(key=lambda x: x.score, reverse=True)
        final_results = ranked[:req.top_k]

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval query execution failed ({engine}): {exc}",
        ) from exc

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    return RetrieveResponse(
        query=req.query,
        count=len(final_results),
        latency_ms=latency_ms,
        engine=engine,
        collection=active_collection,
        results=final_results,
    )


@app.post("/retrieve/compare")
def retrieve_compare(req: RetrieveRequest) -> Dict[str, Any]:
    """
    Migration Verification Tool: Runs query against BOTH MiniLM (production bmu_summaries)
    and Voyage AI (migration bmu_voyage_knowledge) side-by-side to verify quality and latency.
    """
    clean_query = req.query.strip().lower()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_minilm = time.perf_counter()
    minilm_res = retrieve(RetrieveRequest(query=clean_query, top_k=req.top_k, engine="minilm"))
    minilm_time = round((time.perf_counter() - start_minilm) * 1000, 2)

    start_voyage = time.perf_counter()
    voyage_res = retrieve(RetrieveRequest(query=clean_query, top_k=req.top_k, engine="voyage"))
    voyage_time = round((time.perf_counter() - start_voyage) * 1000, 2)

    return {
        "query": req.query,
        "minilm_production": {
            "collection": PROD_COLLECTION_NAME,
            "model": EMBEDDING_MODEL_NAME,
            "latency_ms": minilm_time,
            "count": minilm_res.count,
            "top_url": minilm_res.results[0].url if minilm_res.results else None,
            "top_score": minilm_res.results[0].score if minilm_res.results else 0.0,
            "results": minilm_res.results[:3],
        },
        "voyage_migration": {
            "collection": VOYAGE_COLLECTION_NAME,
            "model": voyage_client.model_name,
            "latency_ms": voyage_time,
            "count": voyage_res.count,
            "top_url": voyage_res.results[0].url if voyage_res.results else None,
            "top_score": voyage_res.results[0].score if voyage_res.results else 0.0,
            "results": voyage_res.results[:3],
        },
    }


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("RETRIEVAL_HOST", "0.0.0.0")
    port = int(os.getenv("RETRIEVAL_SERVICE_PORT", "8001"))
    print(f"Starting BMU Retrieval Microservice on http://{host}:{port}...")
    uvicorn.run("retrieval_service:app", host=host, port=port, reload=False)


