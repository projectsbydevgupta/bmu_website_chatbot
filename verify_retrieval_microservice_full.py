"""
Comprehensive Verification Suite for BMU Retrieval Microservice Implementation
=============================================================================
This test script verifies all 6 core requirements:
1. Both services running on ports 8000 (Backend) and 8001 (Retrieval Microservice).
2. /health endpoints respond correctly on both services.
3. Chatbot requests use `retrieval_mode: "microservice"` with full 3-agent trace.
4. Resilient fallback works without 500 errors when retrieval service is unavailable.
5. Main backend does NOT load SentenceTransformer into memory unnecessarily.
6. Existing ingestion & indexing functionality (ChromaDB collection & vector.py).
"""

import json
import os
import sys
import time
import requests
from dotenv import load_dotenv

# Reconfigure stdout for utf-8
sys.stdout.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)
load_dotenv(dotenv_path=os.path.join(ROOT_DIR, ".env"))


BACKEND_URL = "http://127.0.0.1:8000"
RETRIEVAL_URL = "http://127.0.0.1:8001"

results_record = []


def record_result(category: str, check_name: str, passed: bool, evidence: dict):
    results_record.append({
        "category": category,
        "check": check_name,
        "passed": passed,
        "evidence": evidence,
    })
    status_str = "[PASS]" if passed else "[FAIL]"
    print(f"  {status_str} {check_name}")
    if not passed:
        print(f"         Details: {evidence}")


def test_1_and_2_services_and_health():
    print("\n=======================================================")
    print(" 1 & 2. VERIFYING SERVICE HEALTH & PORTS (8000 & 8001) ")
    print("=======================================================")

    # Backend 8000 /health
    try:
        r_backend = requests.get(f"{BACKEND_URL}/health", timeout=5)
        backend_ok = r_backend.status_code == 200 and r_backend.json().get("status") == "ok"
        record_result(
            "Service Health",
            "Backend Orchestrator port 8000 /health returns 200 OK",
            backend_ok,
            {"status_code": r_backend.status_code, "body": r_backend.json()},
        )
    except Exception as e:
        record_result("Service Health", "Backend Orchestrator port 8000 reachable", False, {"error": str(e)})

    # Retrieval 8001 /health
    try:
        r_retr = requests.get(f"{RETRIEVAL_URL}/health", timeout=5)
        data = r_retr.json()
        retr_ok = (
            r_retr.status_code == 200
            and data.get("status") == "ok"
            and data.get("collection") == "bmu_summaries"
            and data.get("document_count", 0) > 0
        )
        record_result(
            "Service Health",
            "Retrieval Microservice port 8001 /health returns 200 OK with collection stats",
            retr_ok,
            {"status_code": r_retr.status_code, "body": data},
        )
    except Exception as e:
        record_result("Service Health", "Retrieval Microservice port 8001 reachable", False, {"error": str(e)})


def test_3_microservice_retrieval_mode():
    print("\n=======================================================")
    print(" 3. VERIFYING CHATBOT REQUEST USES RETRIEVAL MICROSERVICE")
    print("=======================================================")

    query = "What are the B.Tech CSE eligibility criteria and fees at BMU?"
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json={"query": query},
            timeout=45,
        )
        data = resp.json()
        status_ok = resp.status_code == 200
        guardrail = data.get("guardrail", {})
        trace = data.get("agent_trace", [])

        # Find tool trace
        tool_item = next((item for item in trace if "Tool" in item.get("agent", "")), None)
        tool_details = tool_item.get("details", {}) if tool_item else {}
        mode = tool_details.get("retrieval_mode", "")

        agents_in_trace = [item.get("agent", "") for item in trace]
        all_3_agents = (
            any("Agent 1" in a for a in agents_in_trace)
            and any("Tool" in a for a in agents_in_trace)
            and any("Agent 2" in a for a in agents_in_trace)
            and any("Agent 3" in a for a in agents_in_trace)
        )

        record_result(
            "Chatbot Request",
            "Backend /ask responds 200 OK with JEV guardrails passed",
            status_ok and guardrail.get("status") == "passed",
            {"status_code": resp.status_code, "guardrail_status": guardrail.get("status")},
        )
        record_result(
            "Chatbot Request",
            "Retrieval executed with retrieval_mode: 'microservice'",
            mode == "microservice",
            {"retrieval_mode": mode, "tool_details": tool_details},
        )
        record_result(
            "Chatbot Request",
            "All 3 LangGraph agents executed in sequence with trace",
            all_3_agents,
            {"agent_trace_sequence": agents_in_trace},
        )
    except Exception as e:
        record_result("Chatbot Request", "Backend /ask request failed", False, {"error": str(e)})


def test_4_resilient_fallback():
    print("\n=======================================================")
    print(" 4. VERIFYING RESILIENT FALLBACK ON MICROSERVICE OUTAGE ")
    print("=======================================================")

    import bmu_rag_graph

    # Temporarily redirect RETRIEVAL_SERVICE_URL to unreachable endpoint
    original_url = bmu_rag_graph.RETRIEVAL_SERVICE_URL
    bmu_rag_graph.RETRIEVAL_SERVICE_URL = "http://127.0.0.1:8999"

    try:
        state = {
            "raw_query": "What are the MBA admission requirements at BMU?",
            "optimized_query": "What are the MBA admission requirements at BMU?",
            "keywords": ["mba", "admission"],
            "intent_category": "academic",
            "candidates": [],
            "current_candidate_idx": 0,
            "audit_decision": "TRY_FALLBACK",
            "audit_reason": "",
            "vetted_context": "",
            "final_answer": "",
            "top_results": [],
            "agent_trace": [],
            "error": None,
        }

        fallback_result = bmu_rag_graph.bmu_rag_app.invoke(state)
        trace = fallback_result.get("agent_trace", [])
        tool_item = next((item for item in trace if "Tool" in item.get("agent", "")), None)
        tool_details = tool_item.get("details", {}) if tool_item else {}
        mode = tool_details.get("retrieval_mode", "")
        answer = fallback_result.get("final_answer", "")

        record_result(
            "Resilient Fallback",
            "Fallback engages smoothly without error when microservice is unreachable",
            mode == "in_process_fallback" and len(answer) > 50,
            {
                "retrieval_mode": mode,
                "retrieval_warning": tool_details.get("retrieval_warning", ""),
                "answer_length": len(answer),
            },
        )
    except Exception as e:
        record_result("Resilient Fallback", "Fallback failed with exception", False, {"error": str(e)})
    finally:
        bmu_rag_graph.RETRIEVAL_SERVICE_URL = original_url


def test_5_no_unnecessary_sentence_transformer():
    print("\n=======================================================")
    print(" 5. VERIFYING MAIN BACKEND DOES NOT LOAD SENTENCETRANSFORMER")
    print("=======================================================")

    import subprocess
    cmd = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, r'd:\\BMU_WEBSITE_CHATBOT-main\\BMU_WEBSITE_CHATBOT-main'); "
            "import semantic; "
            "is_loaded = 'sentence_transformers' in sys.modules; "
            "print(f'SENTENCE_TRANSFORMER_LOADED:{is_loaded}'); "
            "sys.exit(0 if not is_loaded else 1)"
        ),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    loaded_on_import = "SENTENCE_TRANSFORMER_LOADED:False" in proc.stdout

    record_result(
        "Module Isolation",
        "semantic.py does NOT load sentence_transformers on startup",
        loaded_on_import,
        {"stdout": proc.stdout.strip(), "returncode": proc.returncode},
    )

    # Also verify that during normal graph invocation with microservice up, sentence_transformers stays unloaded
    cmd_graph = [
        sys.executable,
        "-c",
        (
            "import sys; "
            "sys.path.insert(0, r'd:\\BMU_WEBSITE_CHATBOT-main\\BMU_WEBSITE_CHATBOT-main'); "
            "from bmu_rag_graph import bmu_rag_app; "
            "state = {'raw_query': 'B.Tech fees', 'optimized_query': 'B.Tech fees', 'keywords': [], 'intent_category': 'academic', 'candidates': [], 'current_candidate_idx': 0, 'audit_decision': 'TRY_FALLBACK', 'audit_reason': '', 'vetted_context': '', 'final_answer': '', 'top_results': [], 'agent_trace': [], 'error': None}; "
            "bmu_rag_app.invoke(state); "
            "is_loaded = 'sentence_transformers' in sys.modules; "
            "print(f'GRAPH_RUN_LOADED:{is_loaded}'); "
            "sys.exit(0 if not is_loaded else 1)"
        ),
    ]
    proc_graph = subprocess.run(cmd_graph, capture_output=True, text=True)
    loaded_on_graph = "GRAPH_RUN_LOADED:False" in proc_graph.stdout

    record_result(
        "Module Isolation",
        "Normal graph execution with microservice does NOT load sentence_transformers into memory",
        loaded_on_graph,
        {"stdout": proc_graph.stdout.strip(), "returncode": proc_graph.returncode},
    )


def test_6_ingestion_indexing_verification():
    print("\n=======================================================")
    print(" 6. VERIFYING EXISTING INGESTION & INDEXING (vector.py) ")
    print("=======================================================")

    import chromadb
    import py_compile

    # Verify vector.py compiles cleanly
    try:
        py_compile.compile(os.path.join(ROOT_DIR, "vector.py"), doraise=True)
        vector_syntax_ok = True
    except Exception as e:
        vector_syntax_ok = False

    record_result(
        "Ingestion / Indexing",
        "vector.py compiles cleanly without syntax errors",
        vector_syntax_ok,
        {"script": "vector.py"},
    )

    # Verify ChromaDB connection and collection
    try:
        client = chromadb.CloudClient(
            api_key=os.getenv("CHROMA_API_KEY"),
            tenant=os.getenv("CHROMA_TENANT"),
            database=os.getenv("CHROMA_DATABASE"),
        )
        col = client.get_collection(name="bmu_summaries")
        count = col.count()
        peek = col.peek(limit=1)
        has_docs = count > 1000 and len(peek.get("ids", [])) > 0
        record_result(
            "Ingestion / Indexing",
            "ChromaDB Cloud collection 'bmu_summaries' is indexed and operational",
            has_docs,
            {
                "document_count": count,
                "sample_id": peek.get("ids", [None])[0],
                "sample_url": peek.get("metadatas", [{}])[0].get("url"),
            },
        )
    except Exception as e:
        record_result("Ingestion / Indexing", "ChromaDB connection error", False, {"error": str(e)})


def print_summary():
    print("\n=======================================================")
    print("              FINAL VERIFICATION SUMMARY               ")
    print("=======================================================")
    passed_count = sum(1 for r in results_record if r["passed"])
    total_count = len(results_record)
    for r in results_record:
        mark = "✓" if r["passed"] else "✗"
        print(f" {mark} [{r['category']}] {r['check']}")
    pct = round((passed_count / total_count) * 100, 1)
    print(f"\nTotal: {passed_count}/{total_count} checks passed ({pct}%)\n")
    return passed_count == total_count


if __name__ == "__main__":
    test_1_and_2_services_and_health()
    test_3_microservice_retrieval_mode()
    test_4_resilient_fallback()
    test_5_no_unnecessary_sentence_transformer()
    test_6_ingestion_indexing_verification()
    all_ok = print_summary()
    sys.exit(0 if all_ok else 1)
