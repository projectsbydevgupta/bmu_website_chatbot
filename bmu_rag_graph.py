"""
BMU Multi-Agent RAG Orchestrator (LangGraph + Groq API)
=======================================================
Architecture:
- Deterministic Gate: JEV Input Guardrail (< 3ms)
- Agent 1: Query Specialist (Groq API Key 1 - Query Optimization & Entity Resolution)
- Deterministic Tool: Hybrid ChromaDB Retrieval & Multi-Candidate Extraction
- Agent 2: Context Auditor (Groq API Key 2 - Relevance Evaluation & Autonomous Fallback Decider)
- LangGraph Self-Correction Loop: Evaluates URL 1 -> Fallback URL 2 -> ChromaDB Summary Bundle
- Agent 3: Academic Response Synthesizer (Groq API Key 3 - Grounded Response Generation)
- Deterministic Gate: JEV Output Guardrail (Grounding & Safety Verification)
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from dotenv import load_dotenv
from groq import Groq
from langgraph.graph import END, StateGraph
import requests
import trafilatura

# Load environment
ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
load_dotenv(ENV_PATH)

# Initialize Phoenix AI Observability & Tracing
try:
    from phoenix_tracer import init_phoenix_tracing
    init_phoenix_tracing()
except Exception:
    pass

# Retrieval Microservice URL
RETRIEVAL_SERVICE_URL = os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8001")

# ChromaDB & Resource Settings (used for resilient in-process fallback if microservice is offline)
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")
CHROMA_TENANT = os.getenv("CHROMA_TENANT")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE")
COLLECTION_NAME = "bmu_summaries"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Agent API Keys & Models
GLOBAL_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

GROQ_KEY_QUERY = os.getenv("GROQ_API_KEY_QUERY_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_QUERY = os.getenv("GROQ_MODEL_QUERY_AGENT", "openai/gpt-oss-20b")

GROQ_KEY_CONTEXT = os.getenv("GROQ_API_KEY_CONTEXT_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_CONTEXT = os.getenv("GROQ_MODEL_CONTEXT_AGENT", "openai/gpt-oss-20b")

GROQ_KEY_SYNTHESIS = os.getenv("GROQ_API_KEY_SYNTHESIS_AGENT") or GLOBAL_GROQ_KEY
GROQ_MODEL_SYNTHESIS = os.getenv("GROQ_MODEL_SYNTHESIS_AGENT", "openai/gpt-oss-120b")

# Shared clients cache for fallback
_chroma_collection = None
_embedding_model = None


def get_chroma_and_embedder():
    """Lazily loads local Chroma and SentenceTransformer from shared retrieval_service singleton."""
    try:
        from retrieval_service import get_chroma_and_embedder as _service_get_ce
        return _service_get_ce()
    except Exception:
        global _chroma_collection, _embedding_model
        if _chroma_collection is None:
            import chromadb

            client = chromadb.CloudClient(
                api_key=CHROMA_API_KEY,
                tenant=CHROMA_TENANT,
                database=CHROMA_DATABASE,
            )
            _chroma_collection = client.get_or_create_collection(name=COLLECTION_NAME)
        if _embedding_model is None:
            from retrieval_service import get_embedder
            _embedding_model = get_embedder()
        return _chroma_collection, _embedding_model



# Local LLM Settings (Ollama / OpenAI-compatible endpoint)
LOCAL_LLM_URL = os.getenv("LOCAL_LLM_URL", "http://localhost:11434/v1")
LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
LOCAL_LLM_API_KEY = os.getenv("LOCAL_LLM_API_KEY", "ollama")

_local_llm_client = None


def get_local_llm_client() -> Tuple[Any, str]:
    global _local_llm_client
    model_name = os.getenv("LOCAL_LLM_MODEL", "llama3.2")
    if _local_llm_client is None:
        from openai import OpenAI

        _local_llm_client = OpenAI(base_url=LOCAL_LLM_URL, api_key=LOCAL_LLM_API_KEY, timeout=30.0)
    return _local_llm_client, model_name


def get_groq_client(agent_name: str) -> Tuple[Groq, str]:
    if agent_name == "query":
        return Groq(api_key=GROQ_KEY_QUERY), GROQ_MODEL_QUERY
    elif agent_name == "context":
        return Groq(api_key=GROQ_KEY_CONTEXT), GROQ_MODEL_CONTEXT
    elif agent_name == "synthesis":
        return Groq(api_key=GROQ_KEY_SYNTHESIS), GROQ_MODEL_SYNTHESIS
    return Groq(api_key=GLOBAL_GROQ_KEY), "openai/gpt-oss-120b"


def get_llm_client(agent_name: str, provider: str = "groq") -> Tuple[Any, str]:
    if (provider or "").lower() == "local":
        try:
            return get_local_llm_client()
        except Exception as exc:
            print(f"[Local LLM Warning]: {exc}. Falling back to Groq client.")
            return get_groq_client(agent_name)
    return get_groq_client(agent_name)


# =====================================================================
# State Definitions
# =====================================================================

class CandidateContext(TypedDict):
    source: str
    url: str
    content: str
    score: float
    type: str  # "scraped_page" | "stored_summary"


class AgentTraceItem(TypedDict):
    agent: str
    latency_ms: float
    details: Dict[str, Any]


class BMURAGState(TypedDict):
    raw_query: str
    llm_provider: Optional[str]  # "groq" | "local"

    # Agent 1 outputs
    optimized_query: str
    keywords: List[str]
    intent_category: str

    # Retrieval Tool outputs
    candidates: List[CandidateContext]
    current_candidate_idx: int

    # Agent 2 outputs
    audit_decision: Literal["SUFFICIENT", "TRY_FALLBACK", "NOT_FOUND"]
    audit_reason: str
    vetted_context: str

    # Agent 3 outputs
    final_answer: str
    top_results: List[Dict[str, Any]]

    # Observability
    agent_trace: List[AgentTraceItem]
    error: Optional[str]


# =====================================================================
# Helper Utilities & Domain Lexicon
# =====================================================================

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "on", "at", "for", "to", "of",
    "and", "or", "with", "by", "from", "about", "can", "you", "tell", "me",
    "how", "much", "does", "do", "i", "my", "we", "our", "bmu", "who", "which",
    "give", "please", "want", "know"
}

ACRONYM_MAP = {
    "b.tech": "btech", "b-tech": "btech", "b tech": "btech",
    "m.tech": "mtech", "m-tech": "mtech", "m tech": "mtech",
    "bba": "bba", "b.b.a": "bba", "mba": "mba", "m.b.a": "mba",
    "b.com": "bcom", "bcom": "bcom", "cse": "cse", "c.s.e": "cse",
    "ece": "ece", "e.c.e": "ece",
}


def _normalize_text(text: str) -> str:
    t = (text or "").lower()
    for k, v in ACRONYM_MAP.items():
        t = t.replace(k, v)
    return t


def _stem_token(word: str) -> str:
    w = word.lower().strip()
    if w in ("fees", "tuition"):
        return "fee"
    if w in ("hostels", "hostel", "accommodation", "residence", "dorm", "dorms"):
        return "hostel"
    if w in ("admissions", "admission", "admit"):
        return "admission"
    if w in ("courses", "programmes", "programs", "programme", "program", "course", "degrees", "degree"):
        return "course"
    if w in ("faculty", "faculties", "professors", "professor", "lecturers", "lecturer", "teachers", "teacher", "deans", "dean"):
        return "faculty"
    if w in ("sports", "sport", "athletics", "games", "gymnasium", "gym"):
        return "sport"
    if w in ("placements",):
        return "placement"
    if w in ("scholarships",):
        return "scholarship"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _extract_tokens(text: str) -> set:
    norm = _normalize_text(text)
    words = re.findall(r"[a-z0-9]+", norm)
    tokens = set()
    for w in words:
        if w not in STOPWORDS and len(w) > 1:
            tokens.add(_stem_token(w))
    return tokens


def _keyword_score(query: str, text: str, url: str = "", title: str = "") -> float:
    """Calculates normalized keyword match ratio between query and candidate document/URL/title."""
    q_tokens = _extract_tokens(query)
    if not q_tokens:
        return 0.0

    url_clean = (url or "").replace("-", " ").replace("/", " ").replace(".", " ")
    combined = f"{text or ''} {url_clean} {title or ''}"
    d_tokens = _extract_tokens(combined)

    overlap = q_tokens.intersection(d_tokens)
    base_score = len(overlap) / len(q_tokens)

    # Authority adjustments: prioritize official portal/course/fee/faculty pages over marketing blog posts
    url_low = (url or "").lower()
    if "/social/" in url_low or "/blogs/" in url_low:
        base_score *= 0.65
    elif any(p in url_low for p in ["/courses/", "/fee-structure/", "/admission/", "/programmes/", "/academics/", "/faculty-recruitments/", "/campus-life/sports/"]):
        base_score = min(1.0, base_score * 1.30)

    return round(base_score, 4)


def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Extracts JSON object from LLM response safely."""
    text = (text or "").strip()
    if not text:
        return {}
    # Strip markdown block if present
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(1)
    try:
        return json.loads(text)
    except Exception:
        return {}


# =====================================================================
# Agent 1: Query Specialist Node
# =====================================================================

def query_specialist_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 1: Analyzes student queries, resolves academic terminology/acronyms,
    and produces an optimized search query and target keywords.
    """
    start_t = time.perf_counter()
    raw_query = state.get("raw_query", "")
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("query", provider=provider)

    prompt = f"""You are the BMU Academic Query Specialist.
Analyze the user's inquiry regarding BML Munjal University (BMU) and transform it into an optimal search representation for vector knowledge retrieval.

Core Rules:
1. Acronym Expansion Rule: ONLY expand degree or university acronyms that EXPLICITLY appear in the user's inquiry.
   - If user says "CSE", expand to "Computer Science and Engineering (CSE)".
   - If user says "MBA", expand to "Master of Business Administration (MBA)".
   - If user says "BMU", include "BML Munjal University".
   - NEVER invent or inject degrees or engineering departments (e.g. do NOT inject B.Tech or CSE) if the user did not ask for them.
2. Scope Preservation:
   - For broad queries (e.g. "Faculty details", "Courses offered", "Fee structure", "Admissions process", "Hostel facilities", "Sports"), maintain a university-wide overview representation (e.g. "BML Munjal University academic programmes degrees offered" or "BML Munjal University faculty members teaching staff overview"). Do NOT narrow down to a single professor, single department, or single elective.
   - For specific queries (e.g. "Dr. Deepti Sharma" or "B.Tech CSE fees"), preserve the specific entity or subject.
3. Extract 3-5 core 'keywords'.
4. Classify 'intent' into one of: 'admissions', 'courses', 'fees', 'hostels', 'placements', 'faculty', 'campus', 'general'.

User Query: "{raw_query}"

Respond strictly with a valid JSON object in this format:
{{
  "optimized_query": "...",
  "keywords": ["...", "..."],
  "intent": "..."
}}
"""

    opt_query = raw_query
    keywords = []
    intent = "general"

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a specialized academic query optimizer for BML Munjal University. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=600,
        )
        content = (response.choices[0].message.content or "").strip()
        parsed = _safe_json_parse(content)
        opt_query = parsed.get("optimized_query", "").strip() or raw_query
        keywords = parsed.get("keywords", [])
        intent = parsed.get("intent", "general")
    except Exception as exc:
        print(f"[Agent 1 Exception]: {type(exc).__name__}: {exc}")

    # Heuristic fallback if LLM response was empty or missing required fields
    if not keywords:
        stopwords = {"what", "are", "the", "and", "for", "with", "from", "at", "is", "in", "of", "to", "how", "can", "does", "which"}
        clean_terms = [w.strip("?,.!") for w in raw_query.split() if len(w) > 2 and w.lower() not in stopwords]
        keywords = clean_terms[:5]

    if intent == "general":
        q_lower = raw_query.lower()
        if any(k in q_lower for k in ["fee", "fees", "cost", "scholarship"]):
            intent = "fees"
        elif any(k in q_lower for k in ["eligibility", "admission", "criteria", "apply"]):
            intent = "admissions"
        elif any(k in q_lower for k in ["b.tech", "cse", "mba", "course", "program"]):
            intent = "courses"

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else "Groq Cloud"
    trace_item: AgentTraceItem = {
        "agent": f"Agent 1 (QuerySpecialist) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "optimized_query": opt_query,
            "keywords": keywords,
            "intent": intent,
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "optimized_query": opt_query,
        "keywords": keywords,
        "intent_category": intent,
        "agent_trace": current_trace,
    }


# =====================================================================
# Deterministic Node: Multi-Candidate Retrieval & Extraction Tool
# =====================================================================

def retrieval_tool_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Deterministic Tool: Queries the standalone Retrieval Microservice (default port 8001)
    with the optimized query, performs hybrid ranking, and prepares multiple candidate contexts
    (URL 1 scraped, URL 2 scraped, and Chroma summary bundle).
    Includes resilient fallback to local ChromaDB if microservice is temporarily unreachable.
    """
    start_t = time.perf_counter()
    opt_query = state.get("optimized_query", state["raw_query"]).strip().lower()

    ranked: List[Dict[str, Any]] = []
    retrieval_mode = "microservice"
    retrieval_warning = None

    # 1. Primary path: If external Retrieval Microservice is explicitly configured and running
    if RETRIEVAL_SERVICE_URL and RETRIEVAL_SERVICE_URL.strip() not in ["", "none", "in_process", "in-process", "http://127.0.0.1:8001"]:
        try:
            resp = requests.post(
                f"{RETRIEVAL_SERVICE_URL.rstrip('/')}/retrieve",
                json={"query": opt_query, "top_k": 8},
                timeout=1.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                ranked = data.get("results", [])
                retrieval_mode = "microservice"
            else:
                retrieval_warning = f"Microservice HTTP {resp.status_code}: {resp.text}"
        except Exception as exc:
            retrieval_warning = f"Microservice connection skipped: {exc}"

    # 2. Resilient unified path: Direct in-process retrieval (optimal for cloud containers & instant speed)
    if not ranked:
        try:
            from retrieval_service import retrieve, RetrieveRequest
            engine_type = os.getenv("RETRIEVAL_ENGINE", "minilm").lower()
            res = retrieve(RetrieveRequest(query=opt_query, top_k=8, engine=engine_type))
            ranked = [
                doc.model_dump() if hasattr(doc, "model_dump") else doc.dict()
                for doc in res.results
            ]
            retrieval_mode = "in_process"
        except Exception as in_proc_exc:
            retrieval_warning = f"{retrieval_warning}; In-process retrieval failed: {in_proc_exc}"

    # 3. Last-resort fallback: Direct Chroma collection query
    if not ranked:
        try:
            collection, embedder = get_chroma_and_embedder()
            if embedder is not None:
                query_emb = embedder.encode([opt_query]).tolist()
                results = collection.query(query_embeddings=query_emb, n_results=16)
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                # Correct cosine similarity from squared L2 distance: 1.0 - (dist / 2.0)
                sem_sim = max(0.0, 1.0 - (dist / 2.0))
                kw_sc = _keyword_score(opt_query, doc, url=meta.get("url", ""), title=meta.get("title", ""))
                final_sc = (0.68 * sem_sim) + (0.32 * kw_sc)
                ranked.append({
                    "summary": doc,
                    "url": meta.get("url", ""),
                    "score": float(final_sc),
                })
            ranked.sort(key=lambda x: x["score"], reverse=True)
            retrieval_mode = "in_process_fallback"
        except Exception as fb_exc:
            retrieval_warning = f"{retrieval_warning}; Fallback also failed: {fb_exc}"

    candidates: List[CandidateContext] = []
    top_results: List[Dict[str, Any]] = []

    # If top score is above threshold, construct candidate sources instantly from high-quality Chroma summaries
    if ranked and ranked[0]["score"] >= 0.10:
        enable_scraping = os.getenv("ENABLE_LIVE_SCRAPING", "false").lower() == "true"
        single_candidates: List[CandidateContext] = []
        for i, item in enumerate(ranked[:3]):
            url = item["url"]
            score = item["score"]
            top_results.append({"url": url, "score": score})

            content_text = item["summary"]
            # Optional live scraping: only if explicitly enabled (cloud defaults to instant cached summaries)
            if enable_scraping and url:
                try:
                    resp_web = requests.get(url, timeout=1.5)
                    if resp_web.status_code == 200 and resp_web.text:
                        extracted = trafilatura.extract(resp_web.text)
                        if extracted and len(extracted.strip()) > 100:
                            content_text = extracted[:3500]
                except Exception:
                    pass

            single_candidates.append({
                "source": url or f"Chroma Document #{i+1}",
                "url": url,
                "content": content_text,
                "score": score,
                "type": "scraped_page" if content_text != item["summary"] else "stored_summary",
            })

        # Synthesized multi-document Chroma summaries (top 6 documents)
        summary_bundle = "\n\n---\n\n".join(
            f"Source [{r.get('url', 'BMU Portal')}]:\n{r.get('summary', '')}"
            for r in ranked[:6] if r.get("summary")
        )
        bundle_candidate: Optional[CandidateContext] = None
        if summary_bundle:
            bundle_candidate = {
                "source": "BMU Curated Knowledge Base (Multi-Document Synthesis)",
                "url": ranked[0].get("url", "https://www.bmu.edu.in"),
                "content": summary_bundle,
                "score": ranked[0]["score"],
                "type": "stored_summary",
            }

        # Query Breadth Detection:
        # If the user query is broad / overview-oriented, place the comprehensive multi-document bundle FIRST
        # so synthesis receives the full institutional picture (multiple schools, visiting faculty, degree ranges).
        raw_q_lower = state.get("raw_query", "").lower()
        q_tokens = [w for w in re.findall(r"\w+", raw_q_lower) if w not in STOPWORDS]
        is_broad = (
            len(q_tokens) <= 3
            or any(w in raw_q_lower for w in ["faculty details", "courses offered", "programmes", "programs offered", "what are the courses", "what are the facilities", "sports facilities", "admissions process", "admission process", "fee structure", "fees structure"])
        )

        if is_broad and bundle_candidate:
            candidates.append(bundle_candidate)
            candidates.extend(single_candidates[:2])
        else:
            candidates.extend(single_candidates[:2])
            if bundle_candidate:
                candidates.append(bundle_candidate)
    else:
        # Low similarity score
        for item in ranked[:3]:
            if item["score"] > 0:
                top_results.append({"url": item["url"], "score": item["score"]})

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    trace_details: Dict[str, Any] = {
        "retrieved_count": len(ranked),
        "prepared_candidates": len(candidates),
        "top_score": ranked[0]["score"] if ranked else -1.0,
        "retrieval_mode": retrieval_mode,
        "microservice_url": RETRIEVAL_SERVICE_URL,
    }
    if retrieval_warning:
        trace_details["retrieval_warning"] = retrieval_warning

    trace_item: AgentTraceItem = {
        "agent": "Tool (HybridRetrieval & MultiSourceExtractor)",
        "latency_ms": latency_ms,
        "details": trace_details,
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "candidates": candidates,
        "current_candidate_idx": 0,
        "top_results": top_results,
        "agent_trace": current_trace,
    }


# =====================================================================
# Agent 2: Context Auditor Node
# =====================================================================

def context_auditor_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 2: Evaluates whether the current candidate context contains
    sufficient factual evidence to answer the student's question.
    Drives autonomous LangGraph conditional branching.
    """
    start_t = time.perf_counter()
    candidates = state.get("candidates", [])
    idx = state.get("current_candidate_idx", 0)
    raw_query = state.get("raw_query", "")

    if not candidates or idx >= len(candidates):
        return {
            "audit_decision": "NOT_FOUND",
            "audit_reason": "No candidate sources available or all candidates exhausted.",
            "vetted_context": "",
        }

    candidate = candidates[idx]
    content = candidate["content"]
    source_name = candidate["source"]
    is_last_candidate = (idx >= len(candidates) - 1)
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("context", provider=provider)

    prompt = f"""You are the BMU Context Relevance Auditor.
Assess if the provided Context contains factual information to help answer the Student Question.

Student Question: "{raw_query}"
Candidate Source: {source_name} (Candidate {idx+1} of {len(candidates)})

Context Excerpt:
{content[:6000]}

Evaluate carefully:
- If the context answers the core question or significant parts of it, verdict is 'SUFFICIENT'.
- Multi-Document Synthesis Bundle Rule: If evaluating Candidate Source 'BMU Curated Knowledge Base (Multi-Document Synthesis)', it aggregates multiple official BMU records. If the student asked an overview query (e.g. 'faculty details', 'courses offered', 'admissions process', 'sports facilities') and this bundle contains information across multiple BMU schools, faculty members, or programs, verdict MUST be 'SUFFICIENT'.
- If the context is missing key parts but other candidate pages exist, verdict is 'TRY_FALLBACK'.
- Note: Distinguish specific leadership designations (e.g. 'chancellor' vs 'vice-chancellor'). If the user inquires about the Chancellor and the context only details the Vice-Chancellor without identifying the Chancellor, verdict is 'TRY_FALLBACK' to inspect other leadership candidate sources.
- Broad vs Narrow Scope Check: For single-page candidates, if the student asks for general, university-wide information (e.g. 'faculty details', 'courses offered', 'fee structure') and the candidate context covers ONLY a single specific individual (e.g. one professor's bio) or one single narrow elective, verdict is 'TRY_FALLBACK' to inspect broader candidate sources or the multi-document synthesis bundle.
- If the context is completely unrelated or proves the information does not exist, verdict is 'NOT_FOUND'.

Respond strictly with valid JSON:
{{
  "decision": "SUFFICIENT" or "TRY_FALLBACK" or "NOT_FOUND",
  "reason": "1-sentence justification"
}}
"""

    decision = ""
    reason = "Audited by Context Auditor."

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are the BMU Context Relevance Auditor. Output strictly valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=400,
        )
        content_str = (response.choices[0].message.content or "").strip()
        parsed = _safe_json_parse(content_str)
        decision = parsed.get("decision", "").strip().upper()
        reason = parsed.get("reason", "").strip() or "Audited by Context Auditor."
        if decision not in ["SUFFICIENT", "TRY_FALLBACK", "NOT_FOUND"]:
            decision = ""
    except Exception as exc:
        decision = ""
        reason = f"Audit exception fallback: {exc}"

    # If not the last candidate and decision is NOT_FOUND, switch to TRY_FALLBACK to check next candidate
    if not is_last_candidate and decision == "NOT_FOUND":
        decision = "TRY_FALLBACK"
        reason += " (Checking remaining candidate sources)."

    # Deterministic fallback check if LLM returned empty or ambiguous decision
    cand_score = float(candidate.get("score", 0.0))
    if not decision:
        q_words = [w.lower() for w in re.findall(r"\w+", raw_query) if len(w) > 3 and w.lower() not in STOPWORDS]
        matches = [w for w in q_words if w in content.lower()]
        cand_url = candidate.get("url", "").lower()
        is_single_profile = ("/faculty/" in cand_url and not cand_url.endswith("/faculty/"))
        is_broad = any(w in raw_query.lower() for w in ["faculty details", "courses offered", "programmes", "programs"])
        if is_broad and is_single_profile and not is_last_candidate:
            decision = "TRY_FALLBACK"
            reason = "Auditor fallback: general query detected; inspecting broader candidate context."
        elif len(matches) >= 2 and len(content) > 300 and cand_score >= 0.18:
            decision = "SUFFICIENT"
            reason = f"Heuristic acceptance: verified factual match on {', '.join(matches[:3])}."
        elif is_last_candidate:
            decision = "NOT_FOUND"
            reason = "Heuristic fallback: exhausted candidates without sufficient match."
        else:
            decision = "TRY_FALLBACK"
            reason = "Heuristic fallback: checking next candidate source."

    # For the multi-document bundle or last candidate, accept if relevant keywords match
    if "Multi-Document Synthesis" in source_name or is_last_candidate:
        q_words = [w.lower() for w in re.findall(r"\w+", raw_query) if len(w) > 3 and w.lower() not in STOPWORDS]
        matches = [w for w in q_words if w in content.lower()]
        if len(matches) >= 2 and len(content) > 200 and cand_score >= 0.18:
            decision = "SUFFICIENT"
            reason = f"Conclusive acceptance on knowledge base bundle ({', '.join(matches[:3])})."
        elif is_last_candidate and decision == "TRY_FALLBACK":
            decision = "NOT_FOUND"
            reason += " (Exhausted all candidate sources)."

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else "Groq Cloud"
    trace_item: AgentTraceItem = {
        "agent": f"Agent 2 (ContextAuditor - Candidate #{idx+1}) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "candidate_source": source_name,
            "decision": decision,
            "reason": reason,
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "audit_decision": decision,
        "audit_reason": reason,
        "vetted_context": content if decision == "SUFFICIENT" else "",
        "agent_trace": current_trace,
    }


def route_after_context_audit(state: BMURAGState) -> str:
    """Conditional Edge: Routes to synthesis, fallback iteration, or refusal."""
    decision = state.get("audit_decision", "NOT_FOUND")
    idx = state.get("current_candidate_idx", 0)
    candidates = state.get("candidates", [])

    if decision == "SUFFICIENT":
        return "synthesize"
    if idx < len(candidates) - 1:
        return "advance_candidate"
    return "handle_not_found"


def advance_candidate_node(state: BMURAGState) -> Dict[str, Any]:
    """Autonomous Loop Step: Increments candidate index to inspect next source."""
    new_idx = state.get("current_candidate_idx", 0) + 1
    return {"current_candidate_idx": new_idx}


# =====================================================================
# Node 5: Authoritative Not Found Handler
# =====================================================================

def handle_not_found_node(state: BMURAGState) -> Dict[str, Any]:
    """Handles cases where evidence was verified to be missing in BMU records."""
    raw_query = state.get("raw_query", "")
    reason = state.get("audit_reason", "Official information not found in verified records.")

    msg = (
        "I could not find sufficiently relevant information in official BML Munjal University "
        "records for this query. Please check the official BMU portal (https://www.bmu.edu.in) "
        "or contact the admissions office at info@bmu.edu.in / 1800-103-6888."
    )
    trace_item: AgentTraceItem = {
        "agent": "RefusalHandler (Records Not Found)",
        "latency_ms": 0.0,
        "details": {
            "reason": reason,
            "action": "Authoritative BMU Helpdesk Referral",
        },
    }
    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "final_answer": msg,
        "agent_trace": current_trace,
    }


# =====================================================================
# Agent 3: Academic Response Synthesizer Node
# =====================================================================

def response_synthesizer_node(state: BMURAGState) -> Dict[str, Any]:
    """
    Agent 3: Synthesizes high-quality, grounded academic guidance
    strictly adhering to the vetted context.
    """
    start_t = time.perf_counter()
    raw_query = state.get("raw_query", "")
    vetted_context = state.get("vetted_context", "")
    provider = (state.get("llm_provider") or "groq").lower()

    client, model_name = get_llm_client("synthesis", provider=provider)

    prompt = f"""You are the official BMU Academic Assistant for BML Munjal University.

Answer the student's question clearly, accurately, and concisely based STRICTLY on the provided vetted context.
Guidelines:
1. State facts directly. If fees, dates, or eligibility criteria are mentioned, format them with clean bullet points.
2. If the user asks an overview question (e.g. regarding faculty details, courses offered, academic schools, or campus facilities), synthesize a comprehensive, well-structured response (grouping by School, Category, or visiting/permanent faculty) reflecting all verified facts present across the vetted context.
3. Do not assume, speculate, or extrapolate facts not present in the context.
4. If the context does not explicitly mention a detail, state clearly that it is not specified in the current records.
5. Do not use markdown headers (# or ##), keep paragraphs tidy, polite, and well-structured using bold titles and clean bullet points.

Context:
{vetted_context}

Question:
{raw_query}
"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are the official BMU Academic Assistant. Answer strictly from context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        answer = response.choices[0].message.content or "Not found in context"
    except Exception as exc:
        if provider == "local":
            print(f"[Local LLM Fallback]: Local model unavailable ({exc}). Seamlessly routing to Groq Cloud synthesis.")
            fallback_client, fallback_model = get_groq_client("synthesis")
            try:
                fb_resp = fallback_client.chat.completions.create(
                    model=fallback_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are the official BMU Academic Assistant. Answer strictly from context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=2048,
                )
                answer = fb_resp.choices[0].message.content or "Not found in context"
                provider = "groq (fallback)"
            except Exception as inner_exc:
                answer = f"Unable to generate response due to service error: {inner_exc}"
        else:
            answer = f"Unable to generate response due to service error: {exc}"

    latency_ms = round((time.perf_counter() - start_t) * 1000, 2)
    provider_tag = "Local Ollama" if provider == "local" else ("Groq Cloud (Local Fallback)" if "fallback" in provider else "Groq Cloud")

    trace_item: AgentTraceItem = {
        "agent": f"Agent 3 (ResponseSynthesizer) [{provider_tag}]",
        "latency_ms": latency_ms,
        "details": {
            "provider": provider,
            "model": model_name,
            "answer_length": len(answer),
        },
    }

    current_trace = list(state.get("agent_trace", []))
    current_trace.append(trace_item)

    return {
        "final_answer": answer,
        "agent_trace": current_trace,
    }


# =====================================================================
# Build and Compile LangGraph Workflow
# =====================================================================

def create_bmu_rag_graph():
    workflow = StateGraph(BMURAGState)

    # Add Nodes
    workflow.add_node("query_specialist", query_specialist_node)
    workflow.add_node("retrieval_tool", retrieval_tool_node)
    workflow.add_node("context_auditor", context_auditor_node)
    workflow.add_node("advance_candidate", advance_candidate_node)
    workflow.add_node("handle_not_found", handle_not_found_node)
    workflow.add_node("response_synthesizer", response_synthesizer_node)

    # Set Edges
    workflow.set_entry_point("query_specialist")
    workflow.add_edge("query_specialist", "retrieval_tool")
    workflow.add_edge("retrieval_tool", "context_auditor")

    workflow.add_conditional_edges(
        "context_auditor",
        route_after_context_audit,
        {
            "synthesize": "response_synthesizer",
            "advance_candidate": "advance_candidate",
            "handle_not_found": "handle_not_found",
        },
    )

    # Self-correction loop: fallback -> re-audit next candidate
    workflow.add_edge("advance_candidate", "context_auditor")

    # Terminations
    workflow.add_edge("response_synthesizer", END)
    workflow.add_edge("handle_not_found", END)

    return workflow.compile()


# Global compiled app
bmu_rag_app = create_bmu_rag_graph()
