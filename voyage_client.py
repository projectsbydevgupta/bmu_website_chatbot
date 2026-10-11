"""
Voyage AI Embedding Client
==========================
Wrapper for Voyage AI embedding models hosted on MongoDB Atlas (ai.mongodb.com)
or native Voyage AI API.

Features:
- Secure credential management via environment variables
- Resilient retries with exponential backoff
- Batch embedding generation for bulk ingestion
- Strict API key masking (no credential leakage in logs or exceptions)
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import requests

# Load environment
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

logger = logging.getLogger("voyage_client")


def mask_key(key: Optional[str]) -> str:
    """Masks API key for safe logging (e.g. al-m0kv...rqcN)."""
    if not key or len(key) < 12:
        return "[NOT_CONFIGURED]"
    return f"{key[:7]}...{key[-4:]}"


class VoyageEmbeddingClient:
    """Client for generating dense vector embeddings via Voyage AI / MongoDB Atlas."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: int = 35,
        max_retries: int = 5,
    ):
        self.api_key = api_key or os.getenv("VOYAGE_API_KEY", "")
        self.endpoint = endpoint or os.getenv("VOYAGE_API_ENDPOINT", "https://ai.mongodb.com/v1/embeddings")
        self.model_name = model_name or os.getenv("VOYAGE_MODEL_NAME", "voyage-4-large")
        self.timeout = timeout
        self.max_retries = max_retries

        if not self.api_key:
            logger.warning("VoyageEmbeddingClient initialized without VOYAGE_API_KEY in environment.")

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.endpoint and self.model_name)

    def _get_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise ValueError("VOYAGE_API_KEY is not configured in environment.")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _call_api_with_retry(self, input_texts: List[str]) -> List[List[float]]:
        """Invokes the embeddings endpoint with exponential backoff retry."""
        if not input_texts:
            return []

        payload = {
            "input": input_texts,
            "model": self.model_name,
        }
        headers = self._get_headers()

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    raw_data = data.get("data", [])
                    # Sort by index if returned out-of-order
                    sorted_data = sorted(raw_data, key=lambda x: x.get("index", 0))
                    return [item["embedding"] for item in sorted_data]

                # Handle rate limits (429) specifically: tier has 3 RPM limit, so back off 20-25s
                if resp.status_code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = float(retry_after) + 1.0
                    else:
                        delay = 21.0 + (attempt * 2.0)
                    logger.warning(
                        f"Voyage API returned HTTP 429 (rate limit) on attempt {attempt}/{self.max_retries}. "
                        f"Waiting {delay:.1f}s for RPM window reset..."
                    )
                    time.sleep(delay)
                    last_error = f"HTTP 429: {resp.text[:120]}"
                    continue

                # Handle temporary server errors (500, 502, 503, 504)
                if resp.status_code in [500, 502, 503, 504]:
                    delay = (2 ** (attempt - 1)) * 2.0
                    logger.warning(
                        f"Voyage API returned HTTP {resp.status_code} on attempt {attempt}/{self.max_retries}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                    last_error = f"HTTP {resp.status_code}: {resp.text[:120]}"
                    continue

                # Non-retriable client errors
                err_msg = f"Voyage API client error (HTTP {resp.status_code}): {resp.text[:150]}"
                logger.error(err_msg)
                raise RuntimeError(err_msg)

            except requests.RequestException as exc:
                delay = (2 ** (attempt - 1)) * 1.5
                logger.warning(
                    f"Voyage API network error on attempt {attempt}/{self.max_retries}: {exc}. "
                    f"Retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                last_error = str(exc)

        raise RuntimeError(f"Voyage API call failed after {self.max_retries} retries: {last_error}")

    def embed_query(self, query: str) -> List[float]:
        """Generates embedding for a single query string."""
        clean_text = query.strip()
        if not clean_text:
            raise ValueError("Query string cannot be empty.")
        embeddings = self._call_api_with_retry([clean_text])
        if not embeddings:
            raise RuntimeError("Voyage API returned empty embeddings list for query.")
        return embeddings[0]

    def embed_documents(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generates embeddings for a list of document strings with batching."""
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace empty strings with single whitespace to satisfy API requirements
            sanitized_batch = [t if t.strip() else " " for t in batch]
            batch_embeddings = self._call_api_with_retry(sanitized_batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def test_connection(self) -> Dict[str, Any]:
        """Tests API connectivity with a dummy probe string."""
        start_t = time.perf_counter()
        try:
            emb = self.embed_query("BMU BML Munjal University connection probe")
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "ok",
                "endpoint": self.endpoint,
                "model": self.model_name,
                "dimension": len(emb),
                "key_preview": mask_key(self.api_key),
                "latency_ms": latency_ms,
            }
        except Exception as exc:
            latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
            return {
                "status": "error",
                "endpoint": self.endpoint,
                "model": self.model_name,
                "key_preview": mask_key(self.api_key),
                "error": str(exc),
                "latency_ms": latency_ms,
            }


# Global singleton instance
voyage_client = VoyageEmbeddingClient()
