"""What the agent's telemetry records: spans and metrics, applied as decorators.

feature: telemetry — this module keeps instrumentation out of the agent code. The agent
marks three kinds of function; everything about spans, attributes and metrics lives here:

  @agent_turn(name)            fn(messages, question) -> outcome   → "invoke_agent <name>" span
  @model_call(provider, model, url)  fn(messages) -> Ollama reply → "chat <model>" span
  @tool_execution              fn(name, args, call_id) -> result   → "execute_tool <name>" span

A decorator only sees a function's inputs, return value and exceptions, so that is the
contract: e.g. a failing tool must *raise* inside the decorated function, or its span
stays green.

Names follow the OpenTelemetry GenAI semantic conventions (gen_ai.*); agent- and
Ollama-specific extras live under agent1.* / ollama.*. Where telemetry is *sent* is
decided in telemetry.py.
"""

import contextvars
import functools
import json
import os
import time
import urllib.parse
import uuid

# OpenTelemetry API (pip: opentelemetry-api, in .venv): creates spans and metric
# instruments. Before telemetry.setup() runs these are proxies, so creating them at
# import time is safe — they start recording once a provider is installed.
from opentelemetry import metrics, trace
from opentelemetry.trace import SpanKind, Status, StatusCode

tracer = trace.get_tracer("agent1")
meter = metrics.get_meter("agent1")

# Prompts, tool arguments and results on spans. On by default in this local learning
# setup; in production you'd usually default it off, since spans then carry whatever
# the user and tools said.
CAPTURE_CONTENT = (
    os.environ.get("OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "true").lower()
    == "true"
)

# One process = one chat session here, so one conversation id ties all its turns together.
CONVERSATION_ID = str(uuid.uuid4())

# Bucket boundaries are the ones the GenAI semconv recommends; the SDK defaults
# (0, 5, 10 ... 10000) are sized for milliseconds and would put every call in one bucket.
op_duration = meter.create_histogram(
    "gen_ai.client.operation.duration",
    unit="s",
    description="Duration of one model call",
    explicit_bucket_boundaries_advisory=[
        0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56, 5.12, 10.24, 20.48, 40.96, 81.92
    ],
)
token_usage = meter.create_histogram(
    "gen_ai.client.token.usage",
    unit="{token}",
    description="Tokens per model call",
    explicit_bucket_boundaries_advisory=[
        1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864
    ],
)
# Agent-level metrics: questions no per-call metric answers — how often does a turn
# fail, how many model calls does one take, which tools misfire.
turns_counter = meter.create_counter(
    "agent1.turns", unit="{turn}", description="User turns, by outcome"
)
turn_model_calls = meter.create_histogram(
    "agent1.turn.model_calls", unit="{call}", description="Model calls needed per turn"
)
tool_calls_counter = meter.create_counter(
    "agent1.tool.calls", unit="{call}", description="Tool calls, by tool and outcome"
)

# Per-turn totals. model_call/tool_execution add to the stats of the turn they run in,
# so the root span can report the whole turn without the agent passing anything around.
_turn_stats = contextvars.ContextVar("agent1_turn_stats", default=None)


def _add_to_turn(**counts):
    stats = _turn_stats.get()
    if stats is not None:
        for key, n in counts.items():
            stats[key] += n


def _mark_error(span, error_type, exc=None):
    if exc is not None:
        span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, error_type))
    span.set_attribute("error.type", error_type)


def agent_turn(agent_name):
    """Root span for one user turn. The function returns the outcome: answered | gave_up | outage."""

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(messages, question):
            stats = {"model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0}
            token = _turn_stats.set(stats)
            with tracer.start_as_current_span(f"invoke_agent {agent_name}") as span:
                span.set_attributes(
                    {
                        "gen_ai.operation.name": "invoke_agent",
                        "gen_ai.agent.name": agent_name,
                        "gen_ai.conversation.id": CONVERSATION_ID,
                        # History this turn starts with; all of it is re-sent on every call.
                        "agent1.turn.history_messages": len(messages),
                    }
                )
                if CAPTURE_CONTENT:
                    span.set_attribute(
                        "gen_ai.input.messages",
                        json.dumps([{"role": "user", "content": question}]),
                    )
                try:
                    outcome = fn(messages, question)
                finally:
                    _turn_stats.reset(token)

                span.set_attributes(
                    {
                        "agent1.turn.outcome": outcome,
                        "agent1.turn.model_calls": stats["model_calls"],
                        "agent1.turn.tool_calls": stats["tool_calls"],
                        "gen_ai.usage.input_tokens": stats["input_tokens"],
                        "gen_ai.usage.output_tokens": stats["output_tokens"],
                    }
                )
                # A turn can fail with every child span green — this is the agent-level error.
                if outcome != "answered":
                    _mark_error(span, outcome)
                turns_counter.add(1, {"outcome": outcome})
                turn_model_calls.record(stats["model_calls"], {"outcome": outcome})
                return outcome

        return wrapper

    return decorate


def model_call(provider, model, url):
    """CLIENT span for one model call. Reads the reply in Ollama's shape."""
    server = urllib.parse.urlparse(url)

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(messages):
            metric_attrs = {
                "gen_ai.operation.name": "chat",
                "gen_ai.provider.name": provider,
                "gen_ai.request.model": model,
            }
            with tracer.start_as_current_span(f"chat {model}", kind=SpanKind.CLIENT) as span:
                span.set_attributes(metric_attrs)
                span.set_attribute("server.address", server.hostname)
                span.set_attribute("server.port", server.port or 80)
                if CAPTURE_CONTENT:
                    # Ollama's message format, not the semconv parts structure — close enough to read.
                    span.set_attribute("gen_ai.input.messages", json.dumps(messages))

                started = time.monotonic()
                try:
                    reply = fn(messages)
                except Exception as e:
                    _mark_error(span, type(e).__name__, e)
                    op_duration.record(
                        time.monotonic() - started,
                        {**metric_attrs, "error.type": type(e).__name__},
                    )
                    raise
                op_duration.record(time.monotonic() - started, metric_attrs)

                input_tokens = reply.get("prompt_eval_count", 0)
                output_tokens = reply.get("eval_count", 0)
                span.set_attributes(
                    {
                        "gen_ai.response.model": reply.get("model", model),
                        "gen_ai.usage.input_tokens": input_tokens,
                        "gen_ai.usage.output_tokens": output_tokens,
                        # Ollama reuses its cache for an unchanged prefix of the conversation.
                        "gen_ai.usage.cache_read.input_tokens": reply.get(
                            "prompt_eval_cached_count", 0
                        ),
                        "gen_ai.response.finish_reasons": [reply.get("done_reason", "")],
                    }
                )
                # Ollama's timing breakdown (ns → s): loading the model vs reading the prompt vs generating.
                for field in ("load_duration", "prompt_eval_duration", "eval_duration"):
                    span.set_attribute(f"ollama.{field}_s", reply.get(field, 0) / 1e9)
                if CAPTURE_CONTENT:
                    span.set_attribute("gen_ai.output.messages", json.dumps([reply["message"]]))

                token_usage.record(input_tokens, {**metric_attrs, "gen_ai.token.type": "input"})
                token_usage.record(output_tokens, {**metric_attrs, "gen_ai.token.type": "output"})
                _add_to_turn(model_calls=1, input_tokens=input_tokens, output_tokens=output_tokens)
                return reply

        return wrapper

    return decorate


def tool_execution(fn):
    """Span for one tool call. A tool failure must raise out of fn to be recorded as an error."""

    @functools.wraps(fn)
    def wrapper(name, args, call_id):
        with tracer.start_as_current_span(f"execute_tool {name}") as span:
            span.set_attributes(
                {
                    "gen_ai.operation.name": "execute_tool",
                    "gen_ai.tool.name": name,
                    "gen_ai.tool.type": "function",
                    "gen_ai.tool.call.id": call_id or "",
                }
            )
            if CAPTURE_CONTENT:
                span.set_attribute("gen_ai.tool.call.arguments", json.dumps(args))
            _add_to_turn(tool_calls=1)
            try:
                result = fn(name, args, call_id)
            except Exception as e:
                _mark_error(span, type(e).__name__, e)
                tool_calls_counter.add(1, {"gen_ai.tool.name": name, "outcome": type(e).__name__})
                raise
            if CAPTURE_CONTENT:
                span.set_attribute("gen_ai.tool.call.result", str(result))
            tool_calls_counter.add(1, {"gen_ai.tool.name": name, "outcome": "ok"})
            return result

    return wrapper
