"""Step 2: chat with memory.

The model itself is stateless. It "remembers" only because we keep the whole
conversation in a list and send all of it on every turn.

Run:  .venv/bin/python step2_chat.py
      DEBUG=1 .venv/bin/python step2_chat.py   # also dump raw request/response JSON
"""
import json
import os
import urllib.error
import urllib.request

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
DEBUG = os.environ.get("DEBUG")


def chat(messages):
  """POST the conversation to Ollama's /api/chat and return the parsed JSON reply."""
  payload = {
    "model": MODEL,
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
  messages = [{"role": "system", "content": "You are a grumpy pirate. Answer in at most two sentences"}]
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
    messages.append({"role": "user", "content": question})

    try:
      reply = chat(messages)
    except urllib.error.URLError as e:
      # Drop the unanswered question so history stays user/assistant pairs, and keep chatting.
      messages.pop()
      print(f"!! could not reach {OLLAMA_URL}: {e.reason}")
      continue
    messages.append(reply["message"])

    if DEBUG:
      print("\n=== RAW RESPONSE ===")
      print(json.dumps(reply, indent=2))

    print("\n=== WHAT THE MODEL SAID ===")
    print(reply["message"]["content"])


if __name__ == "__main__":
  main()
