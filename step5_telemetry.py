"""Step 5: the agent loop, with traces and metrics.

Same agent as step 4. Telemetry is attached with three decorators (see instrumentation.py)
so the agent code stays readable; every user turn becomes a trace:

  invoke_agent agent1      one user turn: outcome, model calls, tool calls, tokens
    chat qwen2.5:3b        one model call: tokens, finish reason, Ollama timings
    execute_tool add       one tool call: arguments, result, error
    chat qwen2.5:3b

Run:  .venv/bin/python step5_telemetry.py
      DEBUG=1 .venv/bin/python step5_telemetry.py     # also dump raw request/response JSON
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false ...   # keep prompts/results off spans
See telemetry.py for exporting to Dynatrace.
"""

import datetime
import json
import os
import urllib.error
import urllib.request

# feature: telemetry — local modules: telemetry.py (where data goes),
# instrumentation.py (the decorators that produce it).
import instrumentation as instr
import telemetry

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEBUG = os.environ.get("DEBUG")
MAX_STEPS = 5  # tool rounds per turn
# One reply can hold many tool calls (a small model can repeat a block of calls
# dozens of times in one reply), so rounds alone don't bound the work.
MAX_TOOL_CALLS = 10  # tool calls per turn
MAX_REPLY_TOKENS = 512  # per model call; Ollama's default is "until the context is full"
MODEL_TIMEOUT_S = 120  # per model call; without it a runaway reply hangs the agent
AGENT_NAME = "agent1"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get the current local date and time",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add 2 numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply 2 numbers",
            "parameters": {
                "type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"],
            },
        },
    },
]


def get_time():
    return datetime.datetime.now()


def add(a, b):
    return a + b


def multiply(a, b):
    return a * b


TOOLS_FUNC = {"get_time": get_time, "add": add, "multiply": multiply}


class UnknownToolError(Exception):
    pass


@instr.model_call("ollama", MODEL, OLLAMA_URL)  # feature: telemetry
def chat(messages):
    """POST the conversation to Ollama's /api/chat and return the parsed JSON reply."""
    payload = {
        "model": MODEL,
        "tools": TOOLS,
        "messages": messages,
        "stream": False,  # one complete JSON answer instead of a token-by-token stream
        "options": {"num_predict": MAX_REPLY_TOKENS},
    }
    body = json.dumps(payload).encode()

    if DEBUG:
        print("=== REQUEST ===")
        print(json.dumps(payload, indent=2))

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT_S) as resp:
        return json.load(resp)


@instr.tool_execution  # feature: telemetry
def execute_tool(name, args, call_id):
    """Run one tool. Raises on failure (unknown tool, bad arguments, tool error)."""
    fn = TOOLS_FUNC.get(name)
    if fn is None:
        raise UnknownToolError(f"unknown tool: {name}")
    return fn(**args)


def run_tool(call):
    """Run one tool call and return its result as text. Errors go back to the model as text too."""
    name = call.get("function").get("name")
    args = call.get("function").get("arguments")
    print(f"model wants {name} (args: {args})")
    try:
        return str(execute_tool(name, args, call.get("id")))
    except Exception as e:
        return f"error: {e}"


def agent_loop(messages):
    """The step 4 loop. Returns how the turn ended:
    answered | gave_up | outage | timeout | truncated."""
    tool_rounds = 0
    tool_calls = 0
    while True:
        try:
            reply = chat(messages)
        except urllib.error.URLError as e:
            print(f"!! could not reach {OLLAMA_URL}: {e.reason}")
            return "outage"
        except TimeoutError:
            # Raised as-is (not wrapped in URLError) when Ollama is reachable but slow.
            print(f"!! model did not answer within {MODEL_TIMEOUT_S}s")
            return "timeout"
        if reply.get("done_reason") == "length":
            # Cut off by num_predict: possibly mid tool call, so it is not an answer.
            print(f"!! reply cut off at {MAX_REPLY_TOKENS} tokens")
            return "truncated"
        messages.append(reply["message"])

        tc = reply["message"].get("tool_calls")
        if not tc:
            print(reply["message"]["content"])
            return "answered"

        if tool_rounds == MAX_STEPS:
            print(f"!! gave up after {MAX_STEPS} rounds")
            return "gave_up"
        if tool_calls + len(tc) > MAX_TOOL_CALLS:
            print(f"!! gave up: {tool_calls + len(tc)} tool calls requested, limit is {MAX_TOOL_CALLS}")
            return "gave_up"
        tool_rounds += 1
        tool_calls += len(tc)

        for call in tc:
            messages.append(
                {"role": "tool", "tool_name": call.get("function").get("name"), "content": run_tool(call)}
            )


@instr.agent_turn(AGENT_NAME)  # feature: telemetry
def run_turn(messages, question):
    """One user turn. On any outcome but "answered" the turn is rolled back
    (tools that already ran are NOT undone)."""
    turn_start = len(messages)
    messages.append({"role": "user", "content": question})
    outcome = agent_loop(messages)
    if outcome != "answered":
        del messages[turn_start:]
    return outcome


def main():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they help; otherwise answer from your own knowledge.",
        }
    ]
    telemetry.setup()  # feature: telemetry
    try:
        while True:
            try:
                question = input(">> ")
            except (EOFError, KeyboardInterrupt):  # Ctrl+D / Ctrl+C
                print()
                break
            if question.strip() in ["quit", "exit"]:
                break
            if not question.strip():
                continue
            run_turn(messages, question)
    finally:
        telemetry.shutdown()  # feature: telemetry — flush, or the last turns never arrive


if __name__ == "__main__":
    main()
