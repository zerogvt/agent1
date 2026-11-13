"""Step 5, OpenLLMetry variant: the same agent as agent.py, traced with Traceloop's OpenLLMetry.

The agent loop, limits, tools and configuration (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
LLM_MAX_TOKENS) are agent.py's. What differs is how the model is called and traced:

  - model calls go through the openai SDK; OpenLLMetry's OpenAI instrumentor turns each one
    into an "openai.chat" span and records the gen_ai.client.* metrics
  - the turn and each tool call get spans shaped like the ones traceloop-sdk's @agent / @tool
    decorators make (traceloop.span.kind, traceloop.entity.*). The SDK itself is not used:
    it pulls in ~60 packages, instrumentors for ~40 libraries, and sends usage telemetry to
    Traceloop unless told not to
  - exporting, delta temporality and the terminal tree are telemetry.py's, as for agent.py

  agent1-tl.agent            one user turn
    openai.chat              one model call, from the instrumentor
    geocode.tool             one tool call: traceloop.entity.input / output
      GET                    the tool's outbound HTTP call (plain OTel, from instrumentation.py)
    chat <model>

Run:  .venv/bin/python agent_tl.py
      DEBUG=1 .venv/bin/python agent_tl.py   # also dump raw request/response JSON
"""

import json
import os
import readline

# openai (pip: openai, in .venv): OpenAI's official Python SDK. It speaks the same
# OpenAI-compatible API agent.py calls with urllib, so it works with Ollama and Groq too.
import openai
# OpenLLMetry (pip: opentelemetry-instrumentation-openai + opentelemetry-semantic-conventions-ai,
# in .venv): Traceloop's instrumentor for the openai SDK. Despite the name it is not part of
# the OpenTelemetry project. It writes gen_ai.* span attributes and gen_ai.client.* metrics.
from opentelemetry.instrumentation.openai import OpenAIInstrumentor
from opentelemetry.semconv_ai import SpanAttributes, TraceloopSpanKindValues
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# The agent's tools, limits and settings are shared with agent.py.
from agent import (
    DEBUG,
    DEFAULT_RATE_LIMIT_WAIT_S,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_RATE_LIMIT_RETRIES,
    MAX_RATE_LIMIT_WAIT_S,
    MAX_REPLY_TOKENS,
    MAX_STEPS,
    MAX_TOOL_CALLS,
    MODEL,
    MODEL_TIMEOUT_S,
    TOOLS,
    TOOLS_FUNC,
    UnknownToolError,
    load_history,
    save_history,
    wait_for_rate_limit,
)
from instrumentation import CAPTURE_CONTENT, CONVERSATION_ID
import telemetry

AGENT_NAME = "agent1-tl"
SERVICE_NAME = "agent1-tl"

# OpenLLMetry's own switch for prompts/results on spans; follow agent.py's variable unless set.
os.environ.setdefault("TRACELOOP_TRACE_CONTENT", "true" if CAPTURE_CONTENT else "false")

client = openai.OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "unused",  # the SDK insists on a key; Ollama ignores it
    timeout=MODEL_TIMEOUT_S,
    max_retries=0,  # 429s are retried below, the same way as agent.py; don't stack the SDK's own
)
tracer = trace.get_tracer(SERVICE_NAME)  # a proxy until telemetry.setup() installs the provider
_instrumentor = OpenAIInstrumentor()


def chat(messages):
    """One model call. The OpenLLMetry instrumentor records it; nothing to do here."""
    if DEBUG:
        print("=== REQUEST ===")
        print(json.dumps(messages, indent=2))
    reply = client.chat.completions.create(
        model=MODEL, tools=TOOLS, messages=messages, max_tokens=MAX_REPLY_TOKENS
    )
    if DEBUG:
        print("=== RESPONSE ===")
        print(reply.model_dump_json(indent=2))
    return reply


def retry_after_seconds(e):
    """As agent.py's, but the header comes from the SDK's response object."""
    try:
        return max(0.0, float(e.response.headers.get("retry-after")))
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT_WAIT_S


def chat_with_retry(messages):
    """chat(), with agent.py's 429 policy. Every attempt is its own chat span."""
    for retry in range(1, MAX_RATE_LIMIT_RETRIES + 2):
        try:
            return chat(messages)
        except openai.RateLimitError as e:
            if retry > MAX_RATE_LIMIT_RETRIES:
                raise
            wait = retry_after_seconds(e)
            if wait > MAX_RATE_LIMIT_WAIT_S:
                print(f"!! rate limited; server asks for {wait:.0f}s, more than {MAX_RATE_LIMIT_WAIT_S}s")
                raise
            print(f"!! rate limited, waiting {wait:.1f}s (retry {retry}/{MAX_RATE_LIMIT_RETRIES})")
            # agent.py's: sleeps, and adds a rate_limit_wait event to the current (turn) span.
            wait_for_rate_limit(wait)


def _entity_span(name, kind):
    """A span as traceloop-sdk's decorators make it: "<name>.<kind>", traceloop.span.kind,
    traceloop.entity.name; tool spans also get gen_ai.tool.name."""
    span = tracer.start_span(f"{name}.{kind.value}")
    span.set_attribute(SpanAttributes.TRACELOOP_SPAN_KIND, kind.value)
    span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_NAME, name)
    return span


def execute_tool(name, args, call_id):
    """Run one tool inside a TOOL span. Raises on failure.
    The SDK names the span after the decorated function (one name for all tools);
    here it is named after the tool, so the tree stays readable."""
    span = _entity_span(name, TraceloopSpanKindValues.TOOL)
    span.set_attribute("gen_ai.tool.name", name)
    with trace.use_span(span, end_on_exit=True):  # records the exception and ERROR status on raise
        if CAPTURE_CONTENT:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps({"args": [], "kwargs": json.loads(args or "{}")}))
        fn = TOOLS_FUNC.get(name)
        if fn is None:
            raise UnknownToolError(f"unknown tool: {name}")
        result = fn(**json.loads(args or "{}"))
        if CAPTURE_CONTENT:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(str(result)))
        return result


def run_tool(call):
    """Run one tool call and return its result as text. Errors go back to the model as text too."""
    name = call.function.name
    args = call.function.arguments
    print(f"model wants {name} (args: {args})")
    try:
        return str(execute_tool(name, args, call.id))
    except Exception as e:
        return f"error: {e}"


def agent_loop(messages):
    """agent.py's loop, with the SDK's errors. Returns (outcome, answer or None)."""
    tool_rounds = 0
    tool_calls = 0
    while True:
        try:
            reply = chat_with_retry(messages)
        except openai.APITimeoutError:  # a subclass of APIConnectionError, so it goes first
            print(f"!! model did not answer within {MODEL_TIMEOUT_S}s")
            return "timeout", None
        except openai.APIConnectionError as e:
            print(f"!! could not reach {LLM_BASE_URL}: {e}")
            return "outage", None
        except openai.APIStatusError as e:
            print(f"!! {LLM_BASE_URL} answered {e.status_code}: {str(e.message)[:500]}")
            return ("rate_limited" if e.status_code == 429 else "api_error"), None
        choice = reply.choices[0]
        if choice.finish_reason == "length":
            print(f"!! reply cut off at {MAX_REPLY_TOKENS} tokens")
            return "truncated", None
        message = choice.message
        # Re-send only the standard fields, as plain dicts (as in agent.py).
        sent = {"role": message.role, "content": message.content}
        if message.tool_calls:
            sent["tool_calls"] = [tc.model_dump(exclude_none=True) for tc in message.tool_calls]
        messages.append(sent)

        tc = message.tool_calls
        if not tc:
            print(message.content)
            return "answered", message.content

        if tool_rounds == MAX_STEPS:
            print(f"!! gave up after {MAX_STEPS} rounds")
            return "gave_up", None
        if tool_calls + len(tc) > MAX_TOOL_CALLS:
            print(f"!! gave up: {tool_calls + len(tc)} tool calls requested, limit is {MAX_TOOL_CALLS}")
            return "gave_up", None
        tool_rounds += 1
        tool_calls += len(tc)

        for call in tc:
            messages.append({"role": "tool", "tool_call_id": call.id, "content": run_tool(call)})


def run_turn(messages, question):
    """One user turn, inside an AGENT span. Rolled back unless answered."""
    span = _entity_span(AGENT_NAME, TraceloopSpanKindValues.AGENT)
    # The SDK puts gen_ai.agent.name on every span inside the agent; only the root gets it here.
    span.set_attribute("gen_ai.agent.name", AGENT_NAME)
    # The SDK's equivalent is Traceloop.set_association_properties({"session_id": ...}).
    span.set_attribute(f"{SpanAttributes.TRACELOOP_ASSOCIATION_PROPERTIES}.session_id", CONVERSATION_ID)
    with trace.use_span(span, end_on_exit=True):
        if CAPTURE_CONTENT:
            span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_INPUT, json.dumps({"args": [question], "kwargs": {}}))
        turn_start = len(messages)
        messages.append({"role": "user", "content": question})
        outcome, answer = agent_loop(messages)
        # No OpenLLMetry attribute for how a turn ended; kept under agent1.*, as in agent.py.
        span.set_attribute("agent1.turn.outcome", outcome)
        if outcome == "answered":
            if CAPTURE_CONTENT:
                span.set_attribute(SpanAttributes.TRACELOOP_ENTITY_OUTPUT, json.dumps(answer or ""))
        else:
            del messages[turn_start:]
            span.set_attribute("error.type", outcome)
            span.set_status(Status(StatusCode.ERROR, outcome))
        return outcome


def main():
    load_history()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they help; otherwise answer from your own knowledge.",
        }
    ]
    telemetry.setup(SERVICE_NAME)  # same providers and exporters as agent.py
    _instrumentor.instrument()  # uses the global tracer and meter providers set up above
    try:
        while True:
            try:
                question = input(">> ")
            except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
                print()
                break
            if question.strip() in ["quit", "exit"]:
                last = readline.get_current_history_length()
                if last and readline.get_history_item(last) == question:
                    readline.remove_history_item(last - 1)
                break
            if not question.strip():
                continue
            run_turn(messages, question)
    finally:
        save_history()
        _instrumentor.uninstrument()
        telemetry.shutdown()  # flush, or the last turns never arrive


if __name__ == "__main__":
    main()
