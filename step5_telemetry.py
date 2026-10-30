"""Step 5: the agent loop, with traces and metrics.

Same agent as step 4. Telemetry is attached with three decorators (see instrumentation.py)
so the agent code stays readable; every user turn becomes a trace:

  invoke_agent agent1      one user turn: outcome, model calls, tool calls, tokens
    chat qwen2.5:3b        one model call: tokens, finish reason
    execute_tool add       one tool call: arguments, result, error
    chat qwen2.5:3b

Run:  .venv/bin/python step5_telemetry.py
      DEBUG=1 .venv/bin/python step5_telemetry.py     # also dump raw request/response JSON
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false ...   # keep prompts/results off spans
See telemetry.py for exporting to Dynatrace.

Unlike steps 1-4 (Ollama's own /api/chat), this step speaks the OpenAI-compatible
chat completions API, so the same code runs against local Ollama or a hosted service:

  LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=... LLM_MODEL=... .venv/bin/python step5_telemetry.py
"""

import datetime
import json
import os
import urllib.error
import urllib.parse
import urllib.request

# feature: telemetry — local modules: telemetry.py (where data goes),
# instrumentation.py (the decorators that produce it).
import instrumentation as instr
import telemetry

# Defaults to local Ollama, which serves the OpenAI-compatible API under /v1 and ignores the key.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen2.5:3b")
_host = urllib.parse.urlparse(LLM_BASE_URL).hostname
# gen_ai.provider.name on spans and metrics; guessed from the URL unless set.
PROVIDER = os.environ.get("LLM_PROVIDER") or (
    "ollama" if _host in ("localhost", "127.0.0.1") else _host
)
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


@instr.model_call(PROVIDER, MODEL, LLM_BASE_URL)  # feature: telemetry
def chat(messages):
    """POST the conversation to /chat/completions and return the parsed JSON reply."""
    payload = {
        "model": MODEL,
        "tools": TOOLS,
        "messages": messages,
        "stream": False,  # one complete JSON answer instead of a token-by-token stream
        "max_tokens": MAX_REPLY_TOKENS,
    }
    body = json.dumps(payload).encode()

    if DEBUG:
        print("=== REQUEST ===")
        print(json.dumps(payload, indent=2))

    # Some hosted APIs sit behind Cloudflare, which refuses urllib's default
    # "Python-urllib/3.x" User-Agent with a 403 (error code 1010).
    headers = {"Content-Type": "application/json", "User-Agent": "agent1/0.1"}
    if LLM_API_KEY:
        headers["Authorization"] = f"Bearer {LLM_API_KEY}"
    req = urllib.request.Request(f"{LLM_BASE_URL}/chat/completions", data=body, headers=headers)
    with urllib.request.urlopen(req, timeout=MODEL_TIMEOUT_S) as resp:
        reply = json.load(resp)

    if DEBUG:
        print("=== RESPONSE ===")
        print(json.dumps(reply, indent=2))
    return reply


@instr.tool_execution  # feature: telemetry
def execute_tool(name, args, call_id):
    """Run one tool. args is the JSON string the model sent.
    Raises on failure (unknown tool, bad arguments, tool error)."""
    fn = TOOLS_FUNC.get(name)
    if fn is None:
        raise UnknownToolError(f"unknown tool: {name}")
    return fn(**json.loads(args or "{}"))


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
    answered | gave_up | outage | timeout | rate_limited | api_error | truncated."""
    tool_rounds = 0
    tool_calls = 0
    while True:
        try:
            reply = chat(messages)
        except urllib.error.HTTPError as e:
            # Reachable, but the request was refused. HTTPError is a URLError, so it goes first.
            print(f"!! {LLM_BASE_URL} answered {e.code}: {e.read().decode(errors='replace')[:500]}")
            return "rate_limited" if e.code == 429 else "api_error"
        except urllib.error.URLError as e:
            print(f"!! could not reach {LLM_BASE_URL}: {e.reason}")
            return "outage"
        except TimeoutError:
            # Raised as-is (not wrapped in URLError) when the server is reachable but slow.
            print(f"!! model did not answer within {MODEL_TIMEOUT_S}s")
            return "timeout"
        choice = reply["choices"][0]
        if choice.get("finish_reason") == "length":
            # Cut off by max_tokens: possibly mid tool call, so it is not an answer.
            print(f"!! reply cut off at {MAX_REPLY_TOKENS} tokens")
            return "truncated"
        message = choice["message"]
        # Re-send only the standard fields: extras such as a reasoning model's
        # "reasoning" text can be refused by strict servers when sent back.
        messages.append({k: message[k] for k in ("role", "content", "tool_calls") if k in message})

        tc = message.get("tool_calls")
        if not tc:
            print(message.get("content"))
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
            messages.append({"role": "tool", "tool_call_id": call.get("id"), "content": run_tool(call)})


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
