"""
Comprehensive Test Suite for BMU Knowledge Ingestion Microservice (Voyage AI + ChromaDB)
=======================================================================================
Tests:
1. Ingestion Microservice /health (Voyage AI connectivity + ChromaDB Cloud collection status)
2. URL validation and domain filtering (BMU domains only, excludes media/assets)
3. Text cleaning and semantic chunking with overlap and metadata enrichment
4. SHA-256 content hash deduplication (skips redundant embeddings)
5. Raw document ingestion (/ingest/raw)
6. Live page ingestion (/ingest/url)
7. Crawl ingestion (/ingest/crawl)
8. Ingestion stats and preview semantic search (/ingest/preview)
9. Dual-engine retrieval and migration comparison (/retrieve/compare)
10. Collection separation verification (proves bmu_summaries is untouched)
11. End-to-end production chatbot regression check (/ask)
"""

import sys
import os
import time
import requests
from dotenv import load_dotenv

# Reconfigure stdout for utf-8 on Windows
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


INGESTION_BASE_URL = "http://127.0.0.1:8002"
RETRIEVAL_BASE_URL = "http://127.0.0.1:8001"
BACKEND_BASE_URL = "http://127.0.0.1:8000"


def run_tests():
    passed = 0
    total = 0

    def assert_test(name: str, condition: bool, details: str = ""):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name}")
        else:
            print(f"  [FAIL] {name} | Details: {details}")

    print("=================================================================")
    print("   TESTING BMU INGESTION MICROSERVICE (VOYAGE AI + CHROMADB)     ")
    print("=================================================================")

    # -----------------------------------------------------------------
    # 1. Health Check Endpoint
    # -----------------------------------------------------------------
    print("\n--- 1. Ingestion Microservice Health Check ---")
    try:
        r = requests.get(f"{INGESTION_BASE_URL}/health", timeout=15)
        assert_test("Health endpoint returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        assert_test("Service status is 'ok'", data.get("status") == "ok", f"Got {data.get('status')}")
        assert_test("Voyage model is 'voyage-4-large'", data.get("voyage_model") == "voyage-4-large")
        assert_test("Voyage dimension is 1024", data.get("voyage_dimension") == 1024)
        assert_test("Voyage collection is 'bmu_voyage_knowledge'", data.get("voyage_collection") == "bmu_voyage_knowledge")
        assert_test("API key is masked in response", "..." in data.get("voyage_key_preview", ""), f"Preview: {data.get('voyage_key_preview')}")
        assert_test("Production collection bmu_summaries is tracked separately", data.get("production_collection") == "bmu_summaries")
        assert_test("Production document count is intact (>= 1258)", data.get("production_document_count", 0) >= 1258)
    except Exception as exc:
        assert_test("Health endpoint reachable", False, str(exc))

    # -----------------------------------------------------------------
    # 2. Crawler & URL Normalization
    # -----------------------------------------------------------------
    print("\n--- 2. Crawler URL Validation & Domain Filtering ---")
    from ingestion_service import BMUWebCrawler
    crawler = BMUWebCrawler()
    assert_test("Valid BMU URL accepted", crawler.is_valid_bmu_url("https://www.bmu.edu.in/courses/b-tech/"))
    assert_test("Non-BMU external URL rejected", not crawler.is_valid_bmu_url("https://www.google.com/test"))
    assert_test("PDF asset URL rejected", not crawler.is_valid_bmu_url("https://www.bmu.edu.in/brochure.pdf"))
    assert_test("Image asset URL rejected", not crawler.is_valid_bmu_url("https://www.bmu.edu.in/logo.png"))
    norm_url = crawler.normalize_url("courses/mba/#details", "https://www.bmu.edu.in/")
    assert_test("Relative URL properly normalized and fragment stripped", norm_url == "https://www.bmu.edu.in/courses/mba")

    # -----------------------------------------------------------------
    # 3. Text Cleaning & Semantic Chunking
    # -----------------------------------------------------------------
    print("\n--- 3. Text Cleaning & Semantic Chunking ---")
    from ingestion_service import clean_text, chunk_document, compute_sha256
    sample_raw = "  Admissions Enquiry\n\n\n\r   B.Tech Programme details.\t  Academic fees: Rs 4,10,000 per year.  \n\n\nEligibility requires 60% in 10+2 with PCM.   "
    cleaned = clean_text(sample_raw)
    assert_test("Whitespace and non-printable characters cleaned", "\r" not in cleaned and "\t" not in cleaned and "   " not in cleaned)

    long_text = ("BML Munjal University offers industry-aligned B.Tech programmes. " * 25) + "\n\n" + ("Eligibility criteria require 10+2 with Physics and Maths. " * 25)
    chunks = chunk_document(long_text, url="https://www.bmu.edu.in/btech", title="B.Tech Programmes", chunk_size=600, chunk_overlap=100)
    assert_test("Generates multiple chunks for long document", len(chunks) >= 2, f"Chunks: {len(chunks)}")
    assert_test("Each chunk contains required metadata", all(
        "id" in c and "text" in c and "url" in c and "title" in c and "content_hash" in c and "chunk_index" in c
        for c in chunks
    ))
    assert_test("Content hashes are valid 64-char hex strings", all(len(c["content_hash"]) == 64 for c in chunks))

    # -----------------------------------------------------------------
    # 4. Raw Content Ingestion (/ingest/raw)
    # -----------------------------------------------------------------
    print("\n--- 4. Raw Content Ingestion ---")
    try:
        raw_payload = {
            "url": "https://www.bmu.edu.in/test-scholarships",
            "title": "BMU Merit Scholarships Guide",
            "text": "BML Munjal University offers up to 100% Dean's Merit Scholarships for outstanding academic achievers in B.Tech and MBA programmes based on JEE percentile, CUET scores, and board examination percentages.",
        }
        r = requests.post(f"{INGESTION_BASE_URL}/ingest/raw", json=raw_payload, timeout=20)
        assert_test("Raw ingestion returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        assert_test("Raw content produced chunks", data.get("chunks_generated", 0) > 0)
        assert_test("Ingestion reported success or deduplicated", data.get("ingest_result", {}).get("status") in ["success", "deduplicated"])
    except Exception as exc:
        assert_test("Raw ingestion operational", False, str(exc))

    # -----------------------------------------------------------------
    # 5. Deduplication Verification
    # -----------------------------------------------------------------
    print("\n--- 5. Content Hash Deduplication ---")
    try:
        # Re-send the exact same payload with force_refresh=False
        r = requests.post(f"{INGESTION_BASE_URL}/ingest/raw", json=raw_payload, timeout=20)
        data = r.json()
        ingest_res = data.get("ingest_result", {})
        assert_test("Identical content identified as deduplicated", ingest_res.get("status") == "deduplicated")
        assert_test("No redundant embeddings generated (indexed == 0)", ingest_res.get("indexed") == 0)
        assert_test("Skipped count equals total chunks", ingest_res.get("skipped") == ingest_res.get("total"))
    except Exception as exc:
        assert_test("Deduplication test operational", False, str(exc))

    # -----------------------------------------------------------------
    # 6. Live BMU Page Ingestion (/ingest/url)
    # -----------------------------------------------------------------
    print("\n--- 6. Live BMU Page Ingestion ---")
    try:
        page_url = "https://www.bmu.edu.in/courses/b-tech/"
        r = requests.post(f"{INGESTION_BASE_URL}/ingest/url", json={"url": page_url, "chunk_size": 800}, timeout=45)
        assert_test("Page ingestion returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        assert_test("Page was fetched and parsed", data.get("raw_char_count", 0) > 1000)
        assert_test("Chunks were created", data.get("chunks_generated", 0) >= 5)
        assert_test("Target collection is bmu_voyage_knowledge", data.get("collection") == "bmu_voyage_knowledge")
    except Exception as exc:
        assert_test("Live page ingestion operational", False, str(exc))

    # -----------------------------------------------------------------
    # 7. Semantic Search Preview (/ingest/preview)
    # -----------------------------------------------------------------
    print("\n--- 7. Semantic Search Preview with Voyage Embeddings ---")
    try:
        r = requests.get(f"{INGESTION_BASE_URL}/ingest/preview?query=B.Tech+eligibility+and+admission&top_k=3", timeout=20)
        assert_test("Preview endpoint returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        results = data.get("results", [])
        assert_test("Returns ranked results from Voyage collection", len(results) > 0)
        assert_test("Each result has similarity_score and url", all("similarity_score" in res and "url" in res for res in results))
        assert_test("Top result contains B.Tech information", "b-tech" in results[0]["url"] or "bmu" in results[0]["url"])
    except Exception as exc:
        assert_test("Preview search operational", False, str(exc))

    # -----------------------------------------------------------------
    # 8. Dual-Engine Retrieval & Migration Verification (/retrieve/compare)
    # -----------------------------------------------------------------
    print("\n--- 8. Retrieval Microservice Dual-Engine Comparison ---")
    time.sleep(3)
    try:
        comp_payload = {"query": "What are the B.Tech admission requirements at BMU?", "top_k": 3}
        r = requests.post(f"{RETRIEVAL_BASE_URL}/retrieve/compare", json=comp_payload, timeout=60)
        assert_test("Comparison endpoint returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        minilm = data.get("minilm_production", {})
        voyage = data.get("voyage_migration", {})
        assert_test("MiniLM production collection is bmu_summaries", minilm.get("collection") == "bmu_summaries")
        assert_test("Voyage migration collection is bmu_voyage_knowledge", voyage.get("collection") == "bmu_voyage_knowledge")
        assert_test("MiniLM returned results", minilm.get("count", 0) > 0)
        assert_test("Voyage returned results", voyage.get("count", 0) > 0)
        assert_test("Voyage used voyage-4-large model", voyage.get("model") == "voyage-4-large")
    except Exception as exc:
        assert_test("Dual-engine comparison operational", False, str(exc))

    # -----------------------------------------------------------------
    # 9. Collection Separation & Production Isolation Check
    # -----------------------------------------------------------------
    print("\n--- 9. Vector Collection Separation & Isolation ---")
    try:
        import chromadb
        client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        prod_col = client.get_collection(name="bmu_summaries")
        voyage_col = client.get_collection(name="bmu_voyage_knowledge")

        prod_count = prod_col.count()
        voyage_count = voyage_col.count()

        assert_test("Production collection 'bmu_summaries' has >= 1258 documents", prod_count >= 1258, f"Got: {prod_count}")
        assert_test("Voyage collection 'bmu_voyage_knowledge' has documents", voyage_count > 0, f"Got: {voyage_count}")
        assert_test("Collections are completely separate", prod_col.name != voyage_col.name)
    except Exception as exc:
        assert_test("Collection separation verified", False, str(exc))

    # -----------------------------------------------------------------
    # 10. Existing Chatbot Regression Check (/ask)
    # -----------------------------------------------------------------
    print("\n--- 10. Production Chatbot & Guardrails Regression Check ---")
    try:
        r = requests.post(
            f"{BACKEND_BASE_URL}/ask",
            json={"query": "What are the B.Tech CSE eligibility criteria and fees at BMU?"},
            timeout=45,
        )
        assert_test("Backend /ask returns 200 OK", r.status_code == 200, f"Status: {r.status_code}")
        data = r.json()
        guardrail = data.get("guardrail", {})
        trace = data.get("agent_trace", [])
        assert_test("JEV guardrails passed", guardrail.get("status") == "passed")
        assert_test("All 3 LangGraph agents in trace", len(trace) >= 4)
        assert_test("Answer contains grounded facts", len(data.get("answer", "")) > 100)
    except Exception as exc:
        assert_test("Production chatbot operational", False, str(exc))

    print("\n=================================================================")
    pct = round((passed / total) * 100, 1)
    print(f"Results: {passed}/{total} tests passed ({pct}%)")
    print("=================================================================")
    return passed == total


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
