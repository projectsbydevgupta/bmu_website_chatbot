"""
Phoenix OpenInference Tracing & Observability Setup
====================================================
Initializes OpenInference tracing and OpenTelemetry exporters to stream
traces to Arize Phoenix Cloud (or local server on port 6006).

Supports:
- Arize Phoenix Cloud (OTLP HTTP with project routing & Bearer auth)
- Local Arize Phoenix Server (http://localhost:6006)
- Pure OpenTelemetry standard fallback (zero hard dependency on phoenix package)
- Auto-instrumentation for LangChain if available
"""

import os
import logging
from typing import Optional

logger = logging.getLogger("phoenix_tracer")

_is_initialized = False
_tracer = None
_provider = None

def init_phoenix_tracing(
    project_name: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> bool:
    """
    Initializes Phoenix tracer provider with OpenTelemetry HTTP exporter.
    Safe to call multiple times across services.
    """
    global _is_initialized, _tracer, _provider
    if _is_initialized:
        return True

    proj = project_name or os.getenv("PHOENIX_PROJECT") or os.getenv("PHOENIX_PROJECT_NAME", "bmu-website-chatbot")
    ep = endpoint or os.getenv("PHOENIX_COLLECTOR_ENDPOINT") or os.getenv("PHOENIX_ENDPOINT", "http://localhost:6006")
    api_key = os.getenv("PHOENIX_API_KEY")

    # Sanitize endpoint for HTTP OTLP exporter
    collector_ep = ep.strip().strip("'\"")
    if "arize.com" in collector_ep and not collector_ep.endswith("/v1/traces"):
        collector_ep = collector_ep.rstrip("/") + "/v1/traces"

    # Sanitize OTEL_EXPORTER_OTLP_HEADERS
    raw_otel = os.getenv("OTEL_EXPORTER_OTLP_HEADERS", "").strip().strip("'\"")
    if raw_otel and not raw_otel.startswith("Authorization="):
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Bearer {raw_otel}"

    is_cloud_phoenix = bool(api_key or "arize.com" in collector_ep)
    if not is_cloud_phoenix:
        # Probe local server
        try:
            import urllib.request
            base_ep = collector_ep.replace("/v1/traces", "")
            with urllib.request.urlopen(base_ep, timeout=0.4):
                pass
        except Exception:
            logger.info("[Phoenix Tracing] Local Phoenix server is offline. Skipping trace export.")
            return False

    # Strategy 1: Attempt native phoenix.otel register
    try:
        from phoenix.otel import register
        from opentelemetry import trace

        register_kwargs = {
            "project_name": proj,
            "endpoint": collector_ep,
            "auto_instrument": True,
            "set_global_tracer_provider": True,
        }
        if api_key:
            register_kwargs["headers"] = {
                "Authorization": f"Bearer {api_key}",
                "api_key": api_key,
            }
        _provider = register(**register_kwargs)
        _tracer = trace.get_tracer("bmu_rag_chatbot")
        _is_initialized = True
        logger.info(f"[Phoenix Tracing] Connected via phoenix.otel to {collector_ep} (Project: {proj})")
        return True
    except Exception as e:
        logger.debug(f"[Phoenix Tracing] phoenix.otel register not used: {e}. Falling back to standard OpenTelemetry.")

    # Strategy 2: Native OpenTelemetry standard provider (zero phoenix dependency)
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource

        headers = {}
        if api_key:
            headers["api_key"] = api_key
            headers["authorization"] = f"Bearer {api_key}"

        resource = Resource.create({
            "service.name": proj,
            "project.name": proj,
            "openinference.project.name": proj,
        })

        _provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=collector_ep, headers=headers)

        # BatchSpanProcessor with quick 1s schedule delay for responsive streaming
        processor = BatchSpanProcessor(
            exporter,
            schedule_delay_millis=1000,
            max_export_batch_size=10,
        )
        _provider.add_span_processor(processor)
        trace.set_tracer_provider(_provider)

        # Attempt LangChain auto-instrumentation if present
        try:
            from openinference.instrumentation.langchain import LangChainInstrumentor
            LangChainInstrumentor().instrument(tracer_provider=_provider)
        except Exception:
            pass

        _tracer = trace.get_tracer("bmu_rag_chatbot")
        _is_initialized = True
        logger.info(f"[Phoenix Tracing] Connected via standard OpenTelemetry OTLP to {collector_ep} (Project: {proj})")
        return True
    except Exception as e:
        logger.warning(f"[Phoenix Tracing] Initialization notice: {e}. Running without active trace export.")
        return False


def flush_traces():
    """Flushes active span batches to Phoenix collector."""
    global _provider
    if _provider and hasattr(_provider, "force_flush"):
        try:
            _provider.force_flush(timeout_millis=2000)
        except Exception:
            pass


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
