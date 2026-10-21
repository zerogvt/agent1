# agent1

A very small AI agent built step by step to learn how agents work. It runs on a local model through Ollama and uses no agent framework. Each step is its own file, so you can diff one step against the next.

| Step | File | What it adds |
|---|---|---|
| 1 | `step1_raw_call.py` | One raw HTTP call to the model; prints the request and response JSON |
| 2 | `step2_chat.py` | Chat with memory: the client re-sends the whole history on every turn |
| 3 | `step3_tool.py` | The first tool: the model asks for a call, our code runs it |
| 4 | `step4_agent.py` | The agent loop: several tool rounds, a step cap, errors returned as data, rollback |
| 5 | `step5_telemetry.py` | Traces and metrics (OpenTelemetry, GenAI semantic conventions) |

Step 5 uses two helper modules: `instrumentation.py` (what is recorded) and `telemetry.py` (where it is sent).

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) (or plain `venv` + `pip`)
- [Ollama](https://ollama.com), running locally, with a model that supports tool calling

## Setup

**1. Install Ollama and pull the model**

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:3b
```

If systemd isn't running (common under WSL), start the server yourself in another terminal with `ollama serve`. To check it's up:

```bash
curl -s http://localhost:11434/api/tags
```

**2. Create the virtualenv and install the dependencies**

```bash
uv venv
uv pip install -r requirements.txt
```

Steps 1–4 use only the standard library. `requirements.txt` holds the OpenTelemetry packages for step 5.

## Running

```bash
.venv/bin/python step1_raw_call.py "Why is the sky blue? One sentence."
.venv/bin/python step2_chat.py        # steps 2-5 are interactive; type quit or press Ctrl+D to exit
.venv/bin/python step4_agent.py
.venv/bin/python step5_telemetry.py
```

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to use; it must support tools |
| `DEBUG` | unset | `DEBUG=1` dumps the raw request and response JSON (steps 2–5) |

Some prompts to try with step 4 or 5:

- `What time is it?` (one tool call)
- `What is (3 + 4) * 5? Use the tools.` (does the model chain the two calls, or guess both at once?)
- `What is the capital of France?` (no tools needed)

## Telemetry (step 5)

By default, step 5 prints a span tree after each turn:

```
invoke_agent agent1  16.51s  in=548 out=62 outcome=answered model_calls=2 tool_calls=1
  chat qwen2.5:3b  12.51s  in=246 out=34 finish=stop
  execute_tool add  0.00s
  chat qwen2.5:3b  4.00s  in=302 out=28 finish=stop
```

### Sending to Dynatrace

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<env-id>.live.dynatrace.com/api/v2/otlp
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Api-Token%20<token>"   # %20, not a space
.venv/bin/python step5_telemetry.py
```

- **Token scopes:** `openTelemetryTrace.ingest`, `metrics.ingest`.
- **Allow-list the span attributes.** Dynatrace drops custom span attributes by default. Add `gen_ai.*`, `agent1.*` and `ollama.*` to the allow-list, or the traces arrive without them, even though the metrics look fine.
- **Exit with `quit` or Ctrl+D.** Spans are sent in batches and flushed on exit, so killing the process loses the last turns.
- **Delta temporality.** Metrics are exported with delta temporality, which Dynatrace requires; this is already set up in `telemetry.py`.

### Other telemetry switches

| Variable | Default | Purpose |
|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `true` | Puts prompts, tool arguments and results on spans. Set it to `false` to keep content out of the backend. |
| `CONSOLE_METRICS` | unset | `CONSOLE_METRICS=1` dumps the raw metrics on exit |

## Known limitations

- **Small CPU model.** `qwen2.5:3b` on a CPU is slow, often 5–20 s per call, and sometimes calls tools sloppily. That is useful for learning, but the results aren't reliable.
- **Rollback only affects the history.** When a turn fails (cap reached or Ollama unreachable), it's removed from the history, but tools that already ran are not undone.
- **No quality signal.** A confident wrong answer shows up as `outcome=answered` with every span green.
- **Message format.** `gen_ai.input.messages` / `gen_ai.output.messages` hold Ollama's message format, not the semantic-conventions "parts" structure.
