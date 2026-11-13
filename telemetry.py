"""OpenTelemetry setup for the agent.

feature: telemetry — everything in this module exists so the agent can emit traces
and metrics; the agent itself only calls setup(), get_tracer()/get_meter() and shutdown().

Where telemetry goes:
  - always: a compact span tree printed to the terminal after each turn
  - if OTEL_EXPORTER_OTLP_ENDPOINT is set: traces + metrics over OTLP/HTTP (e.g. Dynatrace)
  - if CONSOLE_METRICS=1: raw metric dump on exit (noisy, for curiosity)

Dynatrace:
  export OTEL_EXPORTER_OTLP_ENDPOINT=https://<env-id>.live.dynatrace.com/api/v2/otlp
  export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Api-Token%20<token>"
  (token scopes: openTelemetryTrace.ingest, metrics.ingest)
"""

import os
from collections import defaultdict

# OpenTelemetry (pip: opentelemetry-api / opentelemetry-sdk /
# opentelemetry-exporter-otlp-proto-http, installed in .venv): the vendor-neutral
# tracing + metrics standard. The API creates spans/instruments, the SDK decides
# where they go, the OTLP exporter ships them to a backend.
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

SERVICE_NAME = "agent1"

# Attributes worth showing in the terminal tree, with a short label each.
# The full set (including message content) goes to the OTLP backend.
TREE_ATTRS = {
    "gen_ai.usage.input_tokens": "in",
    "gen_ai.usage.output_tokens": "out",
    "gen_ai.response.finish_reasons": "finish",
    "agent1.reply.tool_calls": "calls",
    "agent1.turn.outcome": "outcome",
    "agent1.turn.model_calls": "model_calls",
    "agent1.turn.tool_calls": "tool_calls",
    "agent1.turn.rate_limit_retries": "retries",
    "agent1.turn.rate_limit_wait_s": "waited_s",
    "http.response.status_code": "status",  # feature: external tools
    "error.type": "error",
}


class TreePrinter(SpanExporter):
    """Prints each finished trace as an indented tree.

    Spans end child-first, so we hold them per trace until the root span
    (the one without a parent) arrives, then print the whole tree at once.
    """

    def __init__(self):
        self.pending = defaultdict(list)

    def export(self, spans):
        for span in spans:
            self.pending[span.context.trace_id].append(span)
            if span.parent is None:
                self._print_tree(self.pending.pop(span.context.trace_id))
        return SpanExportResult.SUCCESS

    def _print_tree(self, spans):
        children = defaultdict(list)
        for span in spans:
            parent_id = span.parent.span_id if span.parent else None
            children[parent_id].append(span)

        print("\n--- trace ---")

        def walk(parent_id, depth):
            for span in sorted(children[parent_id], key=lambda s: s.start_time):
                secs = (span.end_time - span.start_time) / 1e9
                shown = {
                    label: span.attributes[key]
                    for key, label in TREE_ATTRS.items()
                    if key in span.attributes
                }
                # Sequence attributes (e.g. finish_reasons) arrive as tuples; print them plainly.
                for label, value in shown.items():
                    if isinstance(value, tuple):
                        shown[label] = ",".join(str(v) for v in value)
                details = " ".join(f"{k}={v}" for k, v in shown.items())
                print(f"{'  ' * depth}{span.name}  {secs:.2f}s  {details}")
                walk(span.context.span_id, depth + 1)

        walk(None, 0)
        print("-------------")


def setup(service_name=SERVICE_NAME):
    """Install tracer + meter providers. Call once, before get_tracer()/get_meter().
    agent_tl.py passes its own service name; everything else is shared."""
    resource = Resource.create({"service.name": service_name})
    otlp = bool(os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"))

    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(SimpleSpanProcessor(TreePrinter()))
    if otlp:
        # Endpoint and auth header are read from OTEL_EXPORTER_OTLP_* env vars.
        tracer_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(tracer_provider)

    readers = []
    if otlp:
        # Dynatrace only accepts delta temporality over OTLP; the SDK default is cumulative.
        delta = {
            Counter: AggregationTemporality.DELTA,
            Histogram: AggregationTemporality.DELTA,
        }
        readers.append(
            PeriodicExportingMetricReader(OTLPMetricExporter(preferred_temporality=delta))
        )
    if os.environ.get("CONSOLE_METRICS"):
        # Very long interval: in practice this only exports on shutdown().
        readers.append(
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(), export_interval_millis=3_600_000
            )
        )
    metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=readers))

    target = os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] if otlp else "terminal only"
    print(f"[telemetry] exporting to: {target}")


def get_tracer():
    return trace.get_tracer(SERVICE_NAME)


def get_meter():
    return metrics.get_meter(SERVICE_NAME)


def shutdown():
    """Flush everything still buffered. Without this, the last spans/metrics are lost on exit."""
    trace.get_tracer_provider().shutdown()
    metrics.get_meter_provider().shutdown()
