"""Step 4: the agent loop.

The model may call tools several rounds in a row: call, see the result, call again,
until it answers in text. MAX_STEPS caps the rounds; a turn that hits the cap or
loses the model mid-way is rolled back.

Run:  .venv/bin/python step4_agent.py
      DEBUG=1 .venv/bin/python step4_agent.py   # also dump raw request/response JSON
"""

import json
import os
import urllib.error
import urllib.request
import datetime

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEBUG = os.environ.get("DEBUG")
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

MAX_STEPS = 5


def chat(messages):
    """POST the conversation to Ollama's /api/chat and return the parsed JSON reply."""
    payload = {
        "model": MODEL,
        "tools": TOOLS,
        "messages": messages,
        "stream": False,  # one complete JSON answer instead of a token-by-token stream
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
    with urllib.request.urlopen(req) as resp:
        return json.load(resp)


def main():
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use tools when they help; otherwise answer from your own knowledge.",
        }
    ]
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

        # A conversation is just a list of messages, each with a role and content.
        turn_start = len(messages)
        messages.append({"role": "user", "content": question})
        try:
            reply = chat(messages)
        except urllib.error.URLError as e:
            del messages[turn_start:]
            print(f"!! could not reach {OLLAMA_URL}: {e.reason}")
            continue
        messages.append(reply["message"])

        tc = reply["message"].get("tool_calls")
        step = 0
        while tc:
            step += 1
            if step > MAX_STEPS:
                print(f"!! gave up after {MAX_STEPS} rounds")
                del messages[turn_start:]
                break
            for call in tc:
                name = call.get("function").get("name")
                args = call.get("function").get("arguments")
                print(f"model wants {name} (args: {args})")
                fn = TOOLS_FUNC.get(name)
                try:
                    tool_res = fn(**args) if fn else f"error: unknown tool: {name}"
                except Exception as e:
                    tool_res = f"error: {e}"
                messages.append(
                    {"role": "tool", "tool_name": name, "content": str(tool_res)}
                )
            try:
                reply = chat(messages)
            except urllib.error.URLError as e:
                print(f"!! could not reach {OLLAMA_URL}: {e.reason}")
                del messages[turn_start:]
                break
            messages.append(reply["message"])
            tc = reply["message"].get("tool_calls")
        else:
            print(reply["message"]["content"])


if __name__ == "__main__":
    main()
