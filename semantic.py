import os
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from jev_guardrails import jev_guardrail
from bmu_rag_graph import bmu_rag_app, BMURAGState

load_dotenv()

app = FastAPI(title="BMU Multi-Agent Semantic QA API (LangGraph + JEV)", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")


class RankedUrl(BaseModel):
    url: str
    score: float


class GuardrailMeta(BaseModel):
    status: str = Field(..., description="passed | blocked | flagged")
    action: str = Field(..., description="allow | block | flag_disclaimer | retrieval_gated")
    reason: str
    engine: str = "jev"
    input_checks: dict[str, Any] = Field(default_factory=dict)
    output_checks: dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    query: str
    answer: str
    top_results: list[RankedUrl]
    guardrail: Optional[GuardrailMeta] = None
    agent_trace: Optional[list[dict[str, Any]]] = None


def _build_answer(query: str) -> AskResponse:
    # =========================================================================
    # 1. JEV INPUT GUARDRAILS (Prompt Injection, Toxicity, Academic Domain)
    # =========================================================================
    input_guard = jev_guardrail.validate_input(query)
    if not input_guard.allowed:
        return AskResponse(
            query=query,
            answer=input_guard.suggested_response or "Query blocked by security guardrails.",
            top_results=[],
            guardrail=GuardrailMeta(
                status="blocked",
                action="block",
                reason=input_guard.reason,
                engine=input_guard.engine,
                input_checks={
                    "category": input_guard.category,
                    "injection_probability": input_guard.injection_probability,
                    "toxicity_probability": input_guard.toxicity_probability,
                    "latency_ms": input_guard.latency_ms,
                },
            ),
            agent_trace=[],
        )

    # =========================================================================
    # 2. LANGGRAPH MULTI-AGENT WORKFLOW
    #    Agent 1 (QuerySpecialist) -> HybridRetrievalTool -> Agent 2 (ContextAuditor)
    #    [Autonomous Loop] -> Agent 3 (ResponseSynthesizer)
    # =========================================================================
    initial_state: BMURAGState = {
        "raw_query": query,
        "optimized_query": query,
        "keywords": [],
        "intent_category": input_guard.category,
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

    graph_result = bmu_rag_app.invoke(initial_state)
    raw_answer = graph_result.get("final_answer", "Not found in context")
    top_results_data = graph_result.get("top_results", [])
    vetted_context = graph_result.get("vetted_context", "")
    agent_trace = graph_result.get("agent_trace", [])

    ranked_urls = [
        RankedUrl(url=item["url"], score=float(item["score"]))
        for item in top_results_data
        if item.get("url")
    ]

    # If evidence was not found, return authoritative message without output guard check
    if graph_result.get("audit_decision") == "NOT_FOUND" or not vetted_context:
        return AskResponse(
            query=query,
            answer=raw_answer,
            top_results=ranked_urls,
            guardrail=GuardrailMeta(
                status="passed",
                action="retrieval_gated",
                reason=graph_result.get("audit_reason", "Official information not found in verified records."),
                engine=input_guard.engine,
                input_checks={
                    "category": input_guard.category,
                    "latency_ms": input_guard.latency_ms,
                },
            ),
            agent_trace=agent_trace,
        )

    # =========================================================================
    # 3. JEV OUTPUT GUARDRAILS (Grounding / Hallucination & Safety Verification)
    # =========================================================================
    source_url = ranked_urls[0].url if ranked_urls else ""
    output_guard = jev_guardrail.validate_output(
        query=query,
        context=vetted_context,
        answer=raw_answer,
        source_url=source_url,
    )

    return AskResponse(
        query=query,
        answer=output_guard.final_answer,
        top_results=ranked_urls,
        guardrail=GuardrailMeta(
            status="flagged" if output_guard.action == "flag_disclaimer" else ("blocked" if output_guard.action == "block" else "passed"),
            action=output_guard.action,
            reason="Verified grounded in context" if output_guard.is_grounded else "Low context grounding detected; verification disclaimer appended",
            engine=output_guard.engine,
            input_checks={
                "category": input_guard.category,
                "latency_ms": input_guard.latency_ms,
            },
            output_checks={
                "is_grounded": output_guard.is_grounded,
                "grounding_probability": output_guard.grounding_probability,
                "is_safe": output_guard.is_safe,
                "latency_ms": output_guard.latency_ms,
            },
        ),
        agent_trace=agent_trace,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/agents/status")
def agents_status() -> dict[str, Any]:
    return {
        "status": "active",
        "orchestrator": "LangGraph StateGraph",
        "agents": [
            {
                "name": "Agent 1: Query Specialist",
                "role": "Academic terminology expansion & query vector optimization",
                "model": os.getenv("GROQ_MODEL_QUERY_AGENT", "openai/gpt-oss-20b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_QUERY_AGENT")),
            },
            {
                "name": "Deterministic Tool: Hybrid Retrieval & Extractor",
                "role": "SentenceTransformer dense embeddings & ChromaDB hybrid search",
                "type": "Standalone FastAPI Retrieval Microservice",
                "endpoint": os.getenv("RETRIEVAL_SERVICE_URL", "http://127.0.0.1:8001"),
            },
            {
                "name": "Agent 2: Context Auditor",
                "role": "Information sufficiency auditor & autonomous fallback decider",
                "model": os.getenv("GROQ_MODEL_CONTEXT_AGENT", "openai/gpt-oss-20b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_CONTEXT_AGENT")),
            },
            {
                "name": "Agent 3: Academic Response Synthesizer",
                "role": "High-capacity grounded academic guidance generator",
                "model": os.getenv("GROQ_MODEL_SYNTHESIS_AGENT", "openai/gpt-oss-120b"),
                "key_configured": bool(os.getenv("GROQ_API_KEY_SYNTHESIS_AGENT")),
            },
        ],
    }


@app.get("/guardrails/status")
def guardrails_status() -> dict[str, Any]:
    return {
        "status": "active",
        "has_live_jev_client": jev_guardrail.has_live_client,
        "engine": "jev-cloud" if jev_guardrail.has_live_client else "jev-system-one-local",
        "active_guardrails": [
            "Input Prompt Injection Detection (Noul)",
            "Input Content Toxicity & Abuse Detection (Noul)",
            "Input Academic Domain Scope Classification (Choice)",
            "Retrieval Score Cutoff Gating (Deterministic)",
            "Output Factual Grounding & Hallucination Guardrail (Noul)",
            "Output Content Safety Verification (Noul)",
        ],
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        return _build_answer(request.query)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("semantic:app", host="127.0.0.1", port=8000, reload=False)