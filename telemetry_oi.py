"""OpenTelemetry setup for agent_oi.py, the OpenInference variant of the agent.

feature: telemetry (OpenInference) — the counterpart of telemetry.py. Differences:
  - spans come from OpenInference: its OpenAI instrumentor wraps the openai SDK and makes
    one span per model call; agent_oi.py adds AGENT and TOOL spans through OpenInference's tracer
  - OpenInference writes its own attributes (llm.*, input.value, tool.name ...). With
    enable_genai_semconv it ALSO writes gen_ai.* on each span (both sets, not either/or)
  - spans only: OpenInference has no metrics, so this setup installs no meter provider
  - service.name is agent1-oi, so it can be told apart from agent1 in Dynatrace

Same environment as telemetry.py: OTEL_EXPORTER_OTLP_ENDPOINT / OTEL_EXPORTER_OTLP_HEADERS.
Prompt and result capture: OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false hides
them, as in agent.py; OpenInference's own OPENINFERENCE_HIDE_* variables work too.
"""

import os

# OpenInference (pip: openinference-instrumentation, openinference-instrumentation-openai,
# in .venv): Arize's OpenTelemetry instrumentation for LLM apps. TracerProvider is the OTel
# SDK one plus OpenInference's config (masking, gen_ai.* conversion); OpenAIInstrumentor
# patches openai.OpenAI.request so every SDK call becomes a span.
from openinference.instrumentation import TraceConfig, TracerProvider
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor

from telemetry import TreePrinter

SERVICE_NAME = "agent1-oi"

CAPTURE_CONTENT = (
    os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true").lower()
    == "true"
)

# Unset fields are read from OPENINFERENCE_* env vars by TraceConfig itself.
_hide = {} if CAPTURE_CONTENT else {"hide_inputs": True, "hide_outputs": True}
CONFIG = TraceConfig(enable_genai_semconv=True, **_hide)

_instrumentor = OpenAIInstrumentor()


def setup():
    """Install the tracer provider and instrument the openai SDK. Call once, before get_tracer()."""
    provider = TracerProvider(resource=Resource.create({"service.name": SERVICE_NAME}), config=CONFIG)
    provider.add_span_processor(SimpleSpanProcessor(TreePrinter()))
    otlp = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))
    if otlp:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    # Global too, so spans made through the plain OTel API (the weather tools' GET spans,
    # from instrumentation.py) land in the same traces.
    trace.set_tracer_provider(provider)
    _instrumentor.instrument(tracer_provider=provider, config=CONFIG)

    target = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] if otlp else "terminal only"
    print(f"[telemetry-oi] exporting to: {target}")


def get_tracer():
    """OpenInference's tracer: start_as_current_span() takes openinference_span_kind=."""
    return trace.get_tracer_provider().get_tracer(SERVICE_NAME)


def shutdown():
    """Flush everything still buffered. Without this, the last spans are lost on exit."""
    _instrumentor.uninstrument()
    trace.get_tracer_provider().shutdown()
