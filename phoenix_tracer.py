"""
Phoenix OpenInference Tracing & Observability Setup
====================================================
Initializes OpenInference tracing and OpenTelemetry exporters to stream
traces to the local Arize Phoenix server on http://localhost:6006.

Auto-instruments:
- Groq LLM API calls (Agent 1, Agent 2, Agent 3)
- LangGraph / LangChain graph workflows and executions
- HTTP endpoints and spans
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("phoenix_tracer")

_is_initialized = False
_tracer = None

def init_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """
    Initializes Phoenix tracer provider with OpenInference auto-instrumentation.
    Safe to call multiple times across services.
    """
    global _is_initialized, _tracer
    if _is_initialized:
        return True

    # Check config
    proj = project_name or os.getenv("PHOENIX_PROJECT") or os.getenv("PHOENIX_PROJECT_NAME", "bmu-website-chatbot")
    ep = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
    # Quick probe: only register if Phoenix server is running to prevent export retry delays
    try:
        import urllib.request
        base_ep = ep.replace("/v1/traces", "")
        with urllib.request.urlopen(base_ep, timeout=0.4):
            pass
    except Exception:
        logger.info("[Phoenix Tracing] Phoenix server is offline. Skipping trace export.")
        return False

    try:
        from phoenix.otel import register
        from opentelemetry import trace

        # Register phoenix OpenTelemetry provider with auto instrumentation for Groq/LangChain
        register(
            project_name=proj,
            endpoint=ep,
            auto_instrument=True,
            set_global_tracer_provider=True,
        )
        _tracer = trace.get_tracer("bmu_rag_chatbot")
        _is_initialized = True
        logger.info(f"[Phoenix Tracing] Connected to {ep} (Project: {proj})")
        return True
    except Exception as e:
        logger.warning(f"[Phoenix Tracing] Initialization notice: {e}. Running without active trace export.")
        return False


class _DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_attribute(self, *args, **kwargs):
        pass


class _DummyTracer:
    def start_as_current_span(self, *args, **kwargs):
        return _DummySpan()


def get_tracer():
    """Returns the global OpenTelemetry tracer for custom spans, or a no-op fallback."""
    global _tracer
    if not _is_initialized:
        init_phoenix_tracing()
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("bmu_rag_chatbot")
        except Exception:
            _tracer = _DummyTracer()
    return _tracer

