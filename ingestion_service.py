"""
BMU Knowledge Ingestion Microservice (Voyage AI + ChromaDB Cloud)
==================================================================
Standalone FastAPI microservice providing:
- Web crawling and link discovery across BMU web properties (bmu.edu.in)
- HTML cleaning and boiler-plate removal (Trafilatura)
- Paragraph-aware semantic text chunking
- Content-hash SHA-256 deduplication (skips redundant embeddings)
- Dense 1024-dim vector generation via Voyage AI API (ai.mongodb.com)
- Separate ChromaDB Cloud vector indexing (bmu_voyage_knowledge)
- Full health monitoring, retry handling, and migration verification
"""

from collections import deque
from datetime import datetime, timezone
import hashlib
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import trafilatura

from voyage_client import mask_key, voyage_client

# ---------------------------------------------------------------------------
# Environment & Configuration Setup
# ---------------------------------------------------------------------------
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [BMU-Ingestion] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ingestion_service")

# ChromaDB Cloud Settings
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")

# Collection Names
PROD_COLLECTION_NAME = "bmu_summaries"  # Read-only production collection (MiniLM 384-dim)
VOYAGE_COLLECTION_NAME = os.getenv("VOYAGE_COLLECTION_NAME", "bmu_voyage_knowledge")  # New Voyage collection (1024-dim)

# Microservice Port
INGESTION_PORT = int(os.getenv("INGESTION_SERVICE_PORT", "8002"))

# ---------------------------------------------------------------------------
# FastAPI App Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BMU Knowledge Ingestion Microservice",
    description="Dedicated microservice for BMU web crawling, chunking, Voyage AI embeddings, and ChromaDB Cloud indexing.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# ChromaDB Connection Management
# ---------------------------------------------------------------------------
_chroma_client = None
_voyage_collection = None


def get_chroma_client() -> chromadb.CloudClient:
    """Returns cached ChromaDB CloudClient."""
    global _chroma_client
    if _chroma_client is None:
        if not all([CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DATABASE]):
            raise RuntimeError("Missing required ChromaDB Cloud configuration in .env.")
        _chroma_client = chromadb.CloudClient(
            api_key=CHROMA_API_KEY,
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
        )
    return _chroma_client


def get_voyage_collection():
    """Returns the dedicated Voyage AI collection (bmu_voyage_knowledge)."""
    global _voyage_collection
    if _voyage_collection is None:
        client = get_chroma_client()
        _voyage_collection = client.get_or_create_collection(
            name=VOYAGE_COLLECTION_NAME,
            metadata={"description": "BMU Website knowledge indexed with Voyage AI voyage-4-large 1024-dim embeddings."},
        )
    return _voyage_collection


# ---------------------------------------------------------------------------
# Text Cleaning, Chunking & Deduplication
# ---------------------------------------------------------------------------
def compute_sha256(text: str) -> str:
    """Computes SHA-256 hash of normalized text for deduplication."""
    normalized = " ".join(text.strip().lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def clean_text(raw_text: str) -> str:
    """Removes non-printable characters and redundant whitespace."""
    if not raw_text:
        return ""
    cleaned = re.sub(r"[\r\t\f\v]", " ", raw_text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r" {2,}", " ", cleaned)
    return cleaned.strip()


def chunk_document(
    text: str,
    url: str,
    title: str = "",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Dict[str, Any]]:
    """
    Splits text into context-preserving, overlapping semantic chunks.
    Preserves paragraph and sentence boundaries wherever possible.
    """
    clean_content = clean_text(text)
    if not clean_content or len(clean_content) < 80:
        return []

    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:10]
    paragraphs = clean_content.split("\n\n")

    chunks_text: List[str] = []
    current_chunk: List[str] = []
    current_len = 0

    for para in paragraphs:
        para_clean = para.strip()
        if not para_clean:
            continue

        if current_len + len(para_clean) <= chunk_size:
            current_chunk.append(para_clean)
            current_len += len(para_clean) + 2
        else:
            # If current chunk has content, commit it
            if current_chunk:
                chunks_text.append("\n\n".join(current_chunk))
                # Retain overlap from end of previous chunk if possible
                overlap_text = current_chunk[-1] if len(current_chunk[-1]) <= chunk_overlap else ""
                current_chunk = [overlap_text, para_clean] if overlap_text else [para_clean]
                current_len = sum(len(p) for p in current_chunk) + len(current_chunk) * 2
            else:
                # Single paragraph is larger than chunk_size: split by sentences
                sentences = re.split(r"(?<=[.!?])\s+", para_clean)
                for sent in sentences:
                    if current_len + len(sent) <= chunk_size:
                        current_chunk.append(sent)
                        current_len += len(sent) + 1
                    else:
                        if current_chunk:
                            chunks_text.append(" ".join(current_chunk))
                        current_chunk = [sent]
                        current_len = len(sent)

    if current_chunk:
        chunks_text.append("\n\n".join(current_chunk))

    # Build chunk objects with metadata and content hashes
    doc_chunks: List[Dict[str, Any]] = []
    total = len(chunks_text)
    now_iso = datetime.now(timezone.utc).isoformat()

    for idx, c_text in enumerate(chunks_text):
        c_clean = c_text.strip()
        if len(c_clean) < 60:
            continue  # Skip trivial fragments

        c_hash = compute_sha256(c_clean)
        chunk_id = f"voyage_{url_hash}_{idx:03d}"

        doc_chunks.append({
            "id": chunk_id,
            "text": c_clean,
            "url": url,
            "title": title or "BMU Web Document",
            "chunk_index": idx,
            "total_chunks": total,
            "content_hash": c_hash,
            "char_length": len(c_clean),
            "timestamp": now_iso,
            "model": voyage_client.model_name,
        })

    return doc_chunks


# ---------------------------------------------------------------------------
# BMU Web Crawler
# ---------------------------------------------------------------------------
class BMUWebCrawler:
    """Bounded crawler for BMU official web domains with robots and extension filtering."""

    DEFAULT_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) BMU-IngestionBot/1.0"
    EXCLUDED_EXTENSIONS = (
        ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".css", ".js",
        ".zip", ".tar", ".gz", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx"
    )

    def __init__(self, request_timeout: int = 12):
        self.timeout = request_timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.DEFAULT_USER_AGENT})

    def is_valid_bmu_url(self, url: str) -> bool:
        """Verifies URL belongs to bmu.edu.in and is not a media/binary asset."""
        try:
            parsed = urlparse(url)
            if not parsed.netloc or "bmu.edu.in" not in parsed.netloc:
                return False
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in self.EXCLUDED_EXTENSIONS):
                return False
            return True
        except Exception:
            return False

    def normalize_url(self, raw_url: str, base_url: str = "https://www.bmu.edu.in/") -> Optional[str]:
        """Normalizes and canonicalizes internal URL."""
        try:
            full = urljoin(base_url, raw_url.strip())
            parsed = urlparse(full)
            if not self.is_valid_bmu_url(full):
                return None
            # Standardize scheme and strip query/fragment
            path = parsed.path.rstrip("/")
            if not path:
                path = "/"
            return f"https://{parsed.netloc.lower()}{path}"
        except Exception:
            return None

    def fetch_page(self, url: str) -> Optional[Tuple[str, str, str]]:
        """
        Fetches web page, extracts title and clean main content.
        Returns: (clean_text, page_title, raw_html) or None
        """
        try:
            resp = self.session.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url} - HTTP {resp.status_code}")
                return None

            html = resp.text
            # Extract page title
            title_match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = ""
            if title_match:
                title = re.sub(r"\s+", " ", title_match.group(1)).strip()
                title = re.sub(r"[\r\n|–-].*$", "", title).strip()

            # Extract clean main content with trafilatura
            extracted = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=True,
                no_fallback=False,
                favor_recall=True,
            )

            if not extracted or len(extracted.strip()) < 80:
                logger.info(f"Trafilatura extracted insufficient text from {url} ({len(extracted or '')} chars)")
                return None

            return extracted.strip(), title or "BMU Academic Information", html

        except Exception as exc:
            logger.error(f"Error fetching page {url}: {exc}")
            return None

    def extract_links(self, html: str, page_url: str) -> List[str]:
        """Extracts valid internal BMU hyperlinks from HTML."""
        links: Set[str] = set()
        pattern = r'href=["\'](/[^"\'#\s>]+|https?://(?:www\.)?bmu\.edu\.in[^"\'#\s>]*)["\']'
        for match in re.findall(pattern, html, re.IGNORECASE):
            norm = self.normalize_url(match, page_url)
            if norm and norm != page_url:
                links.add(norm)
        return list(links)

    def crawl(
        self,
        seed_urls: List[str],
        max_pages: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Crawls up to max_pages BMU pages using BFS.
        Returns list of page dictionaries with clean text and metadata.
        """
        queue: deque[str] = deque()
        visited: Set[str] = set()
        pages: List[Dict[str, Any]] = []

        for seed in seed_urls:
            norm = self.normalize_url(seed)
            if norm and norm not in visited:
                queue.append(norm)
                visited.add(norm)

        logger.info(f"Starting BMU crawler with {len(queue)} seeds, max_pages={max_pages}")

        while queue and len(pages) < max_pages:
            current_url = queue.popleft()
            logger.info(f"Crawling ({len(pages) + 1}/{max_pages}): {current_url}")

            result = self.fetch_page(current_url)
            if not result:
                continue

            clean_content, title, raw_html = result
            pages.append({
                "url": current_url,
                "title": title,
                "text": clean_content,
                "char_count": len(clean_content),
            })

            # Discover links if more pages needed
            if len(pages) < max_pages:
                discovered = self.extract_links(raw_html, current_url)
                for link in discovered:
                    if link not in visited:
                        visited.add(link)
                        queue.append(link)

        logger.info(f"Crawl completed. Retrieved {len(pages)} pages.")
        return pages


# ---------------------------------------------------------------------------
# Ingestion Pipeline Core
# ---------------------------------------------------------------------------
def ingest_document_chunks(
    chunks: List[Dict[str, Any]],
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Ingests pre-computed chunks into ChromaDB Cloud with deduplication:
    1. Checks if chunk ID or content hash already exists
    2. Embeds missing chunks using Voyage AI (voyage-4-large)
    3. Adds embeddings and metadata to bmu_voyage_knowledge collection
    """
    if not chunks:
        return {"total": 0, "indexed": 0, "skipped": 0, "status": "no_chunks"}

    collection = get_voyage_collection()

    # Deduplication check
    existing_ids: Set[str] = set()
    if not force_refresh:
        try:
            chunk_ids = [c["id"] for c in chunks]
            get_res = collection.get(ids=chunk_ids)
            existing_ids = set(get_res.get("ids", []))
        except Exception as exc:
            logger.warning(f"Could not check existing IDs from collection: {exc}")

    to_embed = [c for c in chunks if c["id"] not in existing_ids or force_refresh]
    skipped_count = len(chunks) - len(to_embed)

    if not to_embed:
        logger.info(f"Deduplication: all {len(chunks)} chunks already indexed. Skipped.")
        return {
            "total": len(chunks),
            "indexed": 0,
            "skipped": skipped_count,
            "status": "deduplicated",
        }

    # Generate dense vector embeddings with Voyage AI in batches
    texts_to_embed = [c["text"] for c in to_embed]
    logger.info(f"Generating Voyage AI embeddings for {len(texts_to_embed)} chunks...")
    start_emb = time.perf_counter()
    embeddings = voyage_client.embed_documents(texts_to_embed, batch_size=16)
    emb_latency_ms = round((time.perf_counter() - start_emb) * 1000, 2)

    # Format metadatas for ChromaDB
    ids = [c["id"] for c in to_embed]
    documents = [c["text"] for c in to_embed]
    metadatas = [
        {
            "url": c["url"],
            "title": c["title"],
            "chunk_index": c["chunk_index"],
            "total_chunks": c["total_chunks"],
            "content_hash": c["content_hash"],
            "char_length": c["char_length"],
            "timestamp": c["timestamp"],
            "model": c["model"],
        }
        for c in to_embed
    ]

    # Index into ChromaDB Cloud in batches
    logger.info(f"Indexing {len(ids)} chunks into '{VOYAGE_COLLECTION_NAME}'...")
    batch_sz = 32
    for b_idx in range(0, len(ids), batch_sz):
        collection.add(
            ids=ids[b_idx : b_idx + batch_sz],
            embeddings=embeddings[b_idx : b_idx + batch_sz],
            documents=documents[b_idx : b_idx + batch_sz],
            metadatas=metadatas[b_idx : b_idx + batch_sz],
        )

    return {
        "total": len(chunks),
        "indexed": len(to_embed),
        "skipped": skipped_count,
        "embedding_latency_ms": emb_latency_ms,
        "sample_ids": ids[:3],
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------
class IngestUrlRequest(BaseModel):
    url: str = Field(..., description="BMU page URL to ingest")
    force_refresh: bool = Field(default=False, description="Re-embed even if chunk hash exists")
    chunk_size: int = Field(default=800, ge=300, le=2000)
    chunk_overlap: int = Field(default=150, ge=50, le=500)


class IngestCrawlRequest(BaseModel):
    seed_urls: List[str] = Field(
        default_factory=lambda: [
            "https://www.bmu.edu.in/courses/b-tech/",
            "https://www.bmu.edu.in/courses/mba/",
            "https://www.bmu.edu.in/admission/",
        ],
        description="Starting BMU URLs for crawler",
    )
    max_pages: int = Field(default=3, ge=1, le=20, description="Maximum number of pages to crawl")
    chunk_size: int = Field(default=800, ge=300, le=2000)
    chunk_overlap: int = Field(default=150, ge=50, le=500)
    force_refresh: bool = Field(default=False)


class IngestRawRequest(BaseModel):
    url: str = Field(..., description="Source URL reference")
    title: str = Field(default="BMU Academic Entry")
    text: str = Field(..., min_length=20, description="Raw content to clean, chunk, and index")
    force_refresh: bool = Field(default=False)


class HealthResponse(BaseModel):
    status: str
    service: str
    voyage_model: str
    voyage_endpoint: str
    voyage_dimension: int
    voyage_key_preview: str
    voyage_collection: str
    voyage_document_count: Optional[int] = None
    production_collection: str
    production_document_count: Optional[int] = None
    latency_ms: float


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Comprehensive health check verifying Voyage AI connectivity and ChromaDB Cloud status."""
    start_t = time.perf_counter()

    # 1. Test Voyage AI client probe
    probe = voyage_client.test_connection()
    voyage_ok = probe.get("status") == "ok"

    # 2. Check ChromaDB Voyage collection count
    voyage_count = None
    prod_count = None
    chroma_status = "ok"

    try:
        col = get_voyage_collection()
        voyage_count = col.count()

        # Read-only check on prod collection
        client = get_chroma_client()
        prod_col = client.get_collection(name=PROD_COLLECTION_NAME)
        prod_count = prod_col.count()
    except Exception as exc:
        chroma_status = f"chroma_error: {exc}"

    overall_status = "ok" if (voyage_ok and chroma_status == "ok") else f"degraded (voyage: {probe.get('status')}, chroma: {chroma_status})"
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return HealthResponse(
        status=overall_status,
        service="BMU Knowledge Ingestion Microservice",
        voyage_model=voyage_client.model_name,
        voyage_endpoint=voyage_client.endpoint,
        voyage_dimension=probe.get("dimension", 1024),
        voyage_key_preview=probe.get("key_preview", "[MASKED]"),
        voyage_collection=VOYAGE_COLLECTION_NAME,
        voyage_document_count=voyage_count,
        production_collection=PROD_COLLECTION_NAME,
        production_document_count=prod_count,
        latency_ms=latency_ms,
    )


@app.post("/ingest/url")
def ingest_url(req: IngestUrlRequest) -> Dict[str, Any]:
    """Fetches a single BMU page, cleans, chunks, generates Voyage embeddings, and indexes into ChromaDB."""
    start_t = time.perf_counter()
    crawler = BMUWebCrawler()
    norm_url = crawler.normalize_url(req.url)
    if not norm_url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"URL '{req.url}' is not a valid bmu.edu.in web page.",
        )

    page_data = crawler.fetch_page(norm_url)
    if not page_data:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Could not extract meaningful content from {norm_url}.",
        )

    clean_content, title, _ = page_data
    chunks = chunk_document(
        text=clean_content,
        url=norm_url,
        title=title,
        chunk_size=req.chunk_size,
        chunk_overlap=req.chunk_overlap,
    )

    ingest_result = ingest_document_chunks(chunks, force_refresh=req.force_refresh)
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return {
        "url": norm_url,
        "title": title,
        "raw_char_count": len(clean_content),
        "chunks_generated": len(chunks),
        "ingest_result": ingest_result,
        "collection": VOYAGE_COLLECTION_NAME,
        "latency_ms": latency_ms,
    }


@app.post("/ingest/crawl")
def ingest_crawl(req: IngestCrawlRequest) -> Dict[str, Any]:
    """Crawls BMU pages starting from seeds, chunks all discovered pages, and indexes them."""
    start_t = time.perf_counter()
    crawler = BMUWebCrawler()
    pages = crawler.crawl(seed_urls=req.seed_urls, max_pages=req.max_pages)

    if not pages:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Crawler could not retrieve any pages from the provided seeds.",
        )

    all_chunks: List[Dict[str, Any]] = []
    page_summaries = []

    for p in pages:
        p_chunks = chunk_document(
            text=p["text"],
            url=p["url"],
            title=p["title"],
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
        )
        all_chunks.extend(p_chunks)
        page_summaries.append({
            "url": p["url"],
            "title": p["title"],
            "char_count": p["char_count"],
            "chunks": len(p_chunks),
        })

    ingest_res = ingest_document_chunks(all_chunks, force_refresh=req.force_refresh)
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return {
        "pages_crawled": len(pages),
        "total_chunks_prepared": len(all_chunks),
        "page_summaries": page_summaries,
        "ingest_result": ingest_res,
        "collection": VOYAGE_COLLECTION_NAME,
        "latency_ms": latency_ms,
    }


@app.post("/ingest/raw")
def ingest_raw(req: IngestRawRequest) -> Dict[str, Any]:
    """Ingests raw text with metadata directly (useful for batch CSV or custom additions)."""
    start_t = time.perf_counter()
    chunks = chunk_document(
        text=req.text,
        url=req.url,
        title=req.title,
    )
    if not chunks:
        raise HTTPException(status_code=400, detail="Provided text is too short or empty.")

    ingest_res = ingest_document_chunks(chunks, force_refresh=req.force_refresh)
    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)

    return {
        "url": req.url,
        "title": req.title,
        "chunks_generated": len(chunks),
        "ingest_result": ingest_res,
        "collection": VOYAGE_COLLECTION_NAME,
        "latency_ms": latency_ms,
    }


@app.get("/ingest/stats")
def ingest_stats() -> Dict[str, Any]:
    """Returns storage and document statistics for the Voyage AI collection."""
    try:
        col = get_voyage_collection()
        total_count = col.count()

        # Fetch sample of URLs indexed
        sample_urls = set()
        if total_count > 0:
            peek = col.peek(limit=min(total_count, 50))
            for meta in peek.get("metadatas", []):
                if meta and "url" in meta:
                    sample_urls.add(meta["url"])

        return {
            "collection": VOYAGE_COLLECTION_NAME,
            "document_count": total_count,
            "model": voyage_client.model_name,
            "dimension": 1024,
            "sample_urls_indexed": list(sample_urls)[:10],
            "status": "active",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/ingest/preview")
def ingest_preview(query: str = "B.Tech CSE fees eligibility", top_k: int = 3) -> Dict[str, Any]:
    """Tests semantic retrieval against the Voyage AI collection using Voyage query embeddings."""
    start_t = time.perf_counter()
    try:
        col = get_voyage_collection()
        if col.count() == 0:
            return {
                "query": query,
                "collection": VOYAGE_COLLECTION_NAME,
                "results": [],
                "message": "Collection is currently empty. Run /ingest/url or /ingest/crawl first.",
            }

        # Generate query vector with Voyage AI
        q_emb = voyage_client.embed_query(query)
        res = col.query(query_embeddings=[q_emb], n_results=top_k)

        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]

        results = []
        for d, m, dist in zip(docs, metas, dists):
            results.append({
                "url": m.get("url", ""),
                "title": m.get("title", ""),
                "similarity_score": round(max(0.0, 1.0 - dist), 4),
                "snippet": d[:200] + "..." if len(d) > 200 else d,
            })

        latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
        return {
            "query": query,
            "collection": VOYAGE_COLLECTION_NAME,
            "model": voyage_client.model_name,
            "count": len(results),
            "results": results,
            "latency_ms": latency_ms,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("INGESTION_HOST", "0.0.0.0")
    print(f"Starting BMU Ingestion Microservice on http://{host}:{INGESTION_PORT}...")
    uvicorn.run("ingestion_service:app", host=host, port=INGESTION_PORT, reload=False)

