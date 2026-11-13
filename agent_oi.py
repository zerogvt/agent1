"""Step 5, OpenInference variant: the same agent as agent.py, traced with OpenInference.

The agent loop, limits, tools and configuration (LLM_BASE_URL, LLM_API_KEY, LLM_MODEL,
LLM_MAX_TOKENS) are agent.py's; what differs is how the model is called and traced:

  - model calls go through the openai SDK (not urllib), because OpenInference instruments
    SDKs: its OpenAI instrumentor turns each call into a "ChatCompletion" span by itself
  - the turn and each tool call get OpenInference AGENT / TOOL spans, made here by hand
  - no metrics: OpenInference is spans only (see telemetry_oi.py)

  invoke_agent agent1-oi     one user turn (AGENT): question, answer, outcome
    ChatCompletion           one model call (LLM), from the instrumentor
    execute_tool geocode     one tool call (TOOL): arguments, result, error
      GET                    the tool's outbound HTTP call (plain OTel, from instrumentation.py)
    ChatCompletion

Run:  .venv/bin/python agent_oi.py
      DEBUG=1 .venv/bin/python agent_oi.py   # also dump raw request/response JSON
"""

import json
import readline

# openai (pip: openai, in .venv): OpenAI's official Python SDK. It speaks the same
# OpenAI-compatible API agent.py calls with urllib, so it works with Ollama and Groq too.
import openai
# OpenInference semantic conventions (pip: openinference-semantic-conventions, installed
# with the instrumentor): attribute names and span kinds.
from openinference.instrumentation import using_session
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes, ToolCallAttributes
from opentelemetry.trace import Status, StatusCode

# The agent's tools, limits and settings are shared with agent.py.
from agent import (
    DEBUG,
    LLM_API_KEY,
    LLM_BASE_URL,
    MAX_RATE_LIMIT_RETRIES,
    MAX_RATE_LIMIT_WAIT_S,
    DEFAULT_RATE_LIMIT_WAIT_S,
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
from instrumentation import CONVERSATION_ID
import telemetry_oi

AGENT_NAME = "agent1-oi"
TOOL_DESCRIPTIONS = {t["function"]["name"]: t["function"]["description"] for t in TOOLS}

client = openai.OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "unused",  # the SDK insists on a key; Ollama ignores it
    timeout=MODEL_TIMEOUT_S,
    max_retries=0,  # 429s are retried below, the same way as agent.py; don't stack the SDK's own
)
tracer = None  # set in main(), after telemetry_oi.setup()


def chat(messages):
    """One model call. The OpenAI instrumentor records it; nothing to do here."""
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
    """chat(), with agent.py's 429 policy. Every attempt is its own ChatCompletion span."""
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


def execute_tool(name, args, call_id):
    """Run one tool inside an OpenInference TOOL span. Raises on failure."""
    with tracer.start_as_current_span(
        f"execute_tool {name}", openinference_span_kind=OpenInferenceSpanKindValues.TOOL
    ) as span:
        span.set_attributes(
            {
                SpanAttributes.TOOL_NAME: name,
                SpanAttributes.TOOL_DESCRIPTION: TOOL_DESCRIPTIONS.get(name, ""),
                ToolCallAttributes.TOOL_CALL_ID: call_id or "",
            }
        )
        # input.value / output.value are hidden by the TraceConfig when content capture is off.
        span.set_input(args or "{}")
        if not telemetry_oi.CONFIG.hide_inputs:
            # OpenInference's gen_ai conversion takes gen_ai.tool.call.arguments from
            # tool.parameters, which OpenInference defines as the tool's JSON *schema*.
            # Set it directly so Dynatrace shows the call's actual arguments.
            span.set_attribute("gen_ai.tool.call.arguments", args or "{}")
        fn = TOOLS_FUNC.get(name)
        try:
            if fn is None:
                raise UnknownToolError(f"unknown tool: {name}")
            result = fn(**json.loads(args or "{}"))
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            raise  # the span records the exception and sets ERROR status on its way out
        span.set_output(str(result))
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
    """One user turn, inside an OpenInference AGENT span. Rolled back unless answered."""
    # session.id goes on every span started inside, the SDK's too; it becomes gen_ai.conversation.id.
    with using_session(CONVERSATION_ID), tracer.start_as_current_span(
        f"invoke_agent {AGENT_NAME}", openinference_span_kind=OpenInferenceSpanKindValues.AGENT
    ) as span:
        span.set_attribute(SpanAttributes.AGENT_NAME, AGENT_NAME)
        span.set_input(question)
        turn_start = len(messages)
        messages.append({"role": "user", "content": question})
        outcome, answer = agent_loop(messages)
        # No OpenInference attribute for how a turn ended; kept under agent1.*, as in agent.py.
        span.set_attribute("agent1.turn.outcome", outcome)
        if outcome == "answered":
            span.set_output(answer or "")
        else:
            del messages[turn_start:]
            span.set_attribute("error.type", outcome)
            span.set_status(Status(StatusCode.ERROR, outcome))
        return outcome


def main():
    global tracer
    load_history()
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they help; otherwise answer from your own knowledge.",
        }
    ]
    telemetry_oi.setup()
    tracer = telemetry_oi.get_tracer()
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
        telemetry_oi.shutdown()  # flush, or the last turns never arrive


if __name__ == "__main__":
    main()
