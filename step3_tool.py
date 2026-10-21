"""Step 3"""

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
    }
]


def get_time():
    return datetime.datetime.now()


TOOLS_FUNC = {"get_time": get_time}


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
        if DEBUG:
            print("\n=== RAW RESPONSE ===")
            print(json.dumps(reply, indent=2))
        if tc:
            for call in tc:
                name = call.get("function").get("name")
                print(f"model wants {name}")
                fn = TOOLS_FUNC.get(name)
                tool_res = fn() if fn else f"error: unknown tool: {name}"
                messages.append(
                    {"role": "tool", "tool_name": name, "content": str(tool_res)}
                )
            try:
                reply = chat(messages)
            except urllib.error.URLError as e:
                print(f"!! could not reach {OLLAMA_URL}: {e.reason}")
                del messages[turn_start:]
                continue
            messages.append(reply["message"])
        print(reply["message"]["content"])


if __name__ == "__main__":
    main()
