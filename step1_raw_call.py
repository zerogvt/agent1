"""Step 1: one raw call to a local LLM.

No tools, no loop, no memory. We send one message and print exactly what
went over the wire in both directions.

Run:  .venv/bin/python step1_raw_call.py "Why is the sky blue? One sentence."
"""
import json
import os
import sys
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def chat(messages):
    """POST the conversation to Ollama's /api/chat and return the parsed JSON reply."""
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,  # one complete JSON answer instead of a token-by-token stream
    }
    body = json.dumps(payload).encode()

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
    question = " ".join(sys.argv[1:]) or "Why is the sky blue? One sentence."

    # A conversation is just a list of messages, each with a role and content.
    messages = [{"role": "user", "content": question}]

    reply = chat(messages)

    print("\n=== RAW RESPONSE ===")
    print(json.dumps(reply, indent=2))

    print("\n=== WHAT THE MODEL SAID ===")
    print(reply["message"]["content"])


if __name__ == "__main__":
    main()
