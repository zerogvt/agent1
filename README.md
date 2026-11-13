# agent1

A very small AI agent built step by step to learn how agents work. It runs on a local model through Ollama (step 5 can also use a free hosted model) and uses no agent framework. Each step is its own file, so you can diff one step against the next.

| Step | File | What it adds |
|---|---|---|
| 1 | `step1_raw_call.py` | One raw HTTP call to the model; prints the request and response JSON |
| 2 | `step2_chat.py` | Chat with memory: the client re-sends the whole history on every turn |
| 3 | `step3_tool.py` | The first tool: the model asks for a call, our code runs it |
| 4 | `step4_agent.py` | The agent loop: several tool rounds, a step cap, errors returned as data, rollback |
| 5 | `agent.py` | Traces and metrics (OpenTelemetry, GenAI semantic conventions), plus limits that stop a runaway model. Uses the OpenAI-compatible API, so it runs on Ollama or a hosted model |

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

Steps 1–4 use only the standard library. `requirements.txt` holds the OpenTelemetry packages for step 5, plus `openai` and the instrumentors for `agent_oi.py` (OpenInference) and `agent_tl.py` (OpenLLMetry).

## Running

```bash
.venv/bin/python step1_raw_call.py "Why is the sky blue? One sentence."
.venv/bin/python step2_chat.py        # steps 2-5 are interactive; type quit or press Ctrl+D to exit
# agent.py keeps prompt history: press Up/Down to recall earlier prompts, also from past sessions (saved in .agent1_history)
.venv/bin/python step4_agent.py
.venv/bin/python agent.py
```

Environment variables (all optional):

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server (steps 1–4) |
| `OLLAMA_MODEL` | `qwen2.5:3b` | Model to use; it must support tools (steps 1–4) |
| `LLM_BASE_URL` | `http://localhost:11434/v1` | OpenAI-compatible API (step 5). The default is local Ollama |
| `LLM_API_KEY` | unset | API key, sent as `Authorization: Bearer` (step 5). Ollama doesn't need one |
| `LLM_MODEL` | `qwen2.5:3b` | Model to use; it must support tools (step 5) |
| `LLM_PROVIDER` | guessed from the URL | `gen_ai.provider.name` on spans and metrics (step 5): `ollama` for localhost, otherwise the host name |
| `LLM_MAX_TOKENS` | `1000` | Maximum tokens per model reply (step 5). Reasoning models such as gpt-oss also spend this budget on their hidden reasoning; too low and turns end `truncated` |
| `DEBUG` | unset | `DEBUG=1` dumps the raw request and response JSON (steps 2–5) |

### Using a free hosted model (step 5)

A small model on a laptop CPU is slow. Step 5 speaks the OpenAI-compatible chat completions API, so it also runs against hosted services with a free tier, such as [Groq](https://console.groq.com):

```bash
export LLM_BASE_URL=https://api.groq.com/openai/v1
export LLM_API_KEY=<your key>          # create one at console.groq.com; don't commit it
export LLM_MODEL=llama-3.3-70b-versatile   # any model from the provider's list that supports tool calling
.venv/bin/python agent.py
```

- **Your prompts and tool results leave your machine.** Don't let the agent read anything sensitive.
- **Free tiers have rate limits.** A turn makes several model calls. When the server answers 429, the agent waits as long as the `Retry-After` header says and tries again (see the limits below). Only when that fails does the turn end with outcome `rate_limited`.
- **`max_tokens` seems to count against Groq's tokens-per-minute limit.** A 429 reported `Requested 2397` with `max_tokens` at 2000, so each call appears to reserve its whole reply budget up front. Groq's docs don't say so; this is inferred from that one message. A smaller `LLM_MAX_TOKENS` means fewer 429s.

Some prompts to try with step 4 or 5:

- `What time is it?` (one tool call)
- `What is (3 + 4) * 5? Use the tools.` (does the model chain the two calls, or guess both at once?)
- `What is the capital of France?` (no tools needed)

### External tools: weather (step 5)

Step 5 also has two tools that call [Open-Meteo](https://open-meteo.com), a free weather service that needs no key or signup (free for non-commercial use):

| Tool | Calls | Returns |
|---|---|---|
| `geocode(name, country_code?)` | `geocoding-api.open-meteo.com` | Up to 5 places matching the name, with coordinates |
| `get_weather(latitude, longitude, days?)` | `api.open-meteo.com` | Current weather and a daily forecast for 1–7 days, in the place's local time |

The model has to chain them: first geocode the place name, then pass the coordinates to `get_weather`. Each outbound request shows up as a `GET` span under its `execute_tool` span, with the HTTP status. When a request fails (unknown place, bad arguments, service down, or no reply within `EXTERNAL_TIMEOUT_S` = 10 s), the error goes back to the model as text, like any other tool error.

- **Place names and coordinates leave your machine** and go to Open-Meteo.
- **The tool results go straight into the model's context.** Here that's trusted weather data, but in general a tool that reads external content is a way in for prompt injection.

Prompts to try:

- `Should I take an umbrella in Thessaloniki tomorrow?` (geocode, then get_weather)
- `What's the weather in Springfield?` (several matches: does the model pick one or ask?)
- `Is it warmer in Athens or Oslo right now?` (two places: four tool calls)
- `What's the weather in Xyzzyville?` (no match: does the model admit it, or make up a forecast?)

## Telemetry (step 5)

By default, step 5 prints a span tree after each turn:

```
invoke_agent agent1  16.51s  in=548 out=62 outcome=answered model_calls=2 tool_calls=1
  chat qwen2.5:3b  12.51s  in=246 out=34 finish=stop calls=1
  execute_tool add  0.00s
  chat qwen2.5:3b  4.00s  in=302 out=28 finish=stop calls=0
```

`calls=` is the number of tool calls in that one model reply (`agent1.reply.tool_calls`).

### Limits and turn outcomes

Small models can get stuck in a loop. For example, `qwen2.5:3b` has been seen repeating the same block of tool calls dozens of times in a single reply. Step 5 caps each turn with these constants at the top of `agent.py`:

| Constant | Value | What it limits |
|---|---|---|
| `MAX_STEPS` | 15 | Tool rounds per turn (model replies that ask for tools) |
| `MAX_TOOL_CALLS` | 10 | Tool calls per turn; one reply can contain many. Checked before any of the reply's calls run |
| `MAX_REPLY_TOKENS` | 1000 | Tokens per model reply, sent as `max_tokens`. Set it with `LLM_MAX_TOKENS`. Ollama's default is "until the context is full" |
| `MODEL_TIMEOUT_S` | 120 | Seconds to wait for each model call |
| `MAX_RATE_LIMIT_RETRIES` | 3 | Retries of one model call after HTTP 429. Each attempt is its own `chat` span |
| `MAX_RATE_LIMIT_WAIT_S` | 30 | Longest single wait. If `Retry-After` asks for more (say, a daily limit), the turn ends `rate_limited` |
| `DEFAULT_RATE_LIMIT_WAIT_S` | 5 | Wait used when the 429 has no usable `Retry-After` header |

Every turn ends with one of these outcomes, recorded as `agent1.turn.outcome` on the root span and in the `agent1.turns` metric:

| Outcome | Meaning |
|---|---|
| `answered` | The model replied with text, which is **not** necessarily a correct answer (see below) |
| `gave_up` | Hit `MAX_STEPS` or `MAX_TOOL_CALLS` |
| `outage` | Model server unreachable |
| `timeout` | Server reachable, but no reply within `MODEL_TIMEOUT_S` |
| `rate_limited` | Server answered HTTP 429 (free tier limit reached), and retrying didn't help or would have meant too long a wait |
| `api_error` | Server answered with another HTTP error, such as a bad key (401) or an unknown model. The response body is printed |
| `truncated` | The reply hit `MAX_REPLY_TOKENS`. It may stop in the middle of a tool call, so it is not used as an answer |

Every outcome other than `answered` rolls the turn back out of the history.

When a turn waited on 429s, the root span also gets `agent1.turn.rate_limit_retries` and `agent1.turn.rate_limit_wait_s` (shown as `retries=` and `waited_s=` in the tree), plus one `rate_limit_wait` event per wait. Waiting doesn't count towards `MODEL_TIMEOUT_S`.

The limits are sized for the toy tools here. A real task may need higher values, and the outcome metric shows how often turns hit them.

### Sending to Dynatrace

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT=https://<env-id>.live.dynatrace.com/api/v2/otlp
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Api-Token%20<token>"   # %20, not a space
.venv/bin/python agent.py
```

- **Token scopes:** `openTelemetryTrace.ingest`, `metrics.ingest`.
- **Allow-list the span attributes.** Dynatrace drops custom span attributes by default. Add `gen_ai.*` and `agent1.*` to the allow-list, or the traces arrive without them, even though the metrics look fine.
- **Exit with `quit` or Ctrl+D.** Spans are sent in batches and flushed on exit, so killing the process loses the last turns.
- **Delta temporality.** Metrics are exported with delta temporality, which Dynatrace requires; this is already set up in `telemetry.py`.

### Other telemetry switches

| Variable | Default | Purpose |
|---|---|---|
| `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` | `true` | Puts prompts, tool arguments and results on spans. Set it to `false` to keep content out of the backend. |
| `CONSOLE_METRICS` | unset | `CONSOLE_METRICS=1` dumps the raw metrics on exit |

## OpenInference variant: `agent_oi.py`

The same agent, with the same tools, limits and `LLM_*` settings, traced with [OpenInference](https://github.com/Arize-ai/openinference) instead of the hand-written `instrumentation.py`. It exists to compare the two in Dynatrace. Both work; use either.

```bash
.venv/bin/python agent_oi.py          # same env vars as agent.py, including OTEL_EXPORTER_OTLP_*
```

- **Model calls go through the `openai` SDK**, because OpenInference instruments SDKs, not raw HTTP. Its `OpenAIInstrumentor` makes one `ChatCompletion` span per call. The SDK's own retries are off (`max_retries=0`); 429s are retried as in `agent.py`.
- **Turn and tool spans** are OpenInference `AGENT` and `TOOL` spans, created in `agent_oi.py`. The weather tools' `GET` spans still come from `instrumentation.py`.
- **`gen_ai.*` is on** (`TraceConfig(enable_genai_semconv=True)` in `telemetry_oi.py`). Every span then carries both families: `gen_ai.*` and OpenInference's own `llm.*`, `input.value`, `output.value`. On a model span, `input.value` is the whole request, tool definitions included, so spans are much bigger than `agent.py`'s.
- **Messages use the semantic-conventions "parts" format**, unlike `agent.py`.
- **Service name `agent1-oi`**, agent name `agent1-oi`.
- **Token scope:** `openTelemetryTrace.ingest` is enough. There are no metrics.

What you lose compared with `agent.py`:

| | `agent.py` | `agent_oi.py` |
|---|---|---|
| Metrics (token usage, duration, turns, tool calls) | yes | none: OpenInference is spans only |
| Turn totals (tokens, model calls, tool calls, retries, waited) | on the turn span | not recorded (only `agent1.turn.outcome` is kept) |
| `error.type` on a refused model call (e.g. `429`) | yes | no: ERROR status and an exception event only |
| `gen_ai.provider.name` | `ollama`, or the host | `groq` for Groq; `openai` for Ollama (OpenInference only knows hosted APIs by host name) |
| Tool calls per reply (`agent1.reply.tool_calls`) | yes | no |

One workaround in `agent_oi.py`: OpenInference fills `gen_ai.tool.call.arguments` from `tool.parameters`, which it defines as the tool's JSON *schema*. So the call's real arguments are set on `gen_ai.tool.call.arguments` directly.

To keep prompts and results off spans, use `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false` as with `agent.py`; OpenInference's `OPENINFERENCE_HIDE_INPUTS` / `OPENINFERENCE_HIDE_OUTPUTS` also work.

## OpenLLMetry variant: `agent_tl.py`

The same agent again, traced with Traceloop's [OpenLLMetry](https://github.com/traceloop/openllmetry), for a three-way comparison in Dynatrace.

```bash
.venv/bin/python agent_tl.py          # same env vars as agent.py, including OTEL_EXPORTER_OTLP_*
```

- **Only the OpenAI instrumentor is used** (`opentelemetry-instrumentation-openai`, Traceloop's despite the name), not the full `traceloop-sdk`. The SDK pulls in about 60 packages and instrumentors for about 40 libraries, and sends anonymous usage telemetry to Traceloop unless `TRACELOOP_TELEMETRY=false`.
- **Model calls** go through the `openai` SDK, as in `agent_oi.py`. Each becomes an `openai.chat` span with `gen_ai.*` attributes only (no second attribute family), messages in the "parts" format, and `gen_ai.client.token.usage` / `gen_ai.client.operation.duration` metrics.
- **Turn and tool spans** are shaped like the ones `traceloop-sdk`'s `@agent` / `@tool` decorators make: `<name>.agent` / `<name>.tool`, with `traceloop.span.kind`, `traceloop.entity.name`, `traceloop.entity.input` / `output`. They carry no `gen_ai.operation.name`, so they don't show up as `invoke_agent` / `execute_tool`.
- **Exporting** is `telemetry.py`'s (delta temporality, terminal tree), with service name `agent1-tl`. Token scopes as for `agent.py`.
- **Content:** `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false` sets OpenLLMetry's `TRACELOOP_TRACE_CONTENT=false`.

## Comparing the three

| | `agent.py` (hand-written) | `agent_oi.py` (OpenInference) | `agent_tl.py` (OpenLLMetry) |
|---|---|---|---|
| Model call | `urllib` | `openai` SDK | `openai` SDK |
| Model span name | `chat <model>` | `ChatCompletion` | `openai.chat` |
| Attribute families | `gen_ai.*` (+ `agent1.*`) | `gen_ai.*` + `llm.*`, `input.value` … | `gen_ai.*` |
| Message format | OpenAI messages | semconv "parts" | semconv "parts" |
| Turn / tool spans as `invoke_agent` / `execute_tool` | yes | yes | no (`traceloop.span.kind`) |
| Tool call arguments / result | `gen_ai.tool.call.*` | `gen_ai.tool.call.*` | `traceloop.entity.input` / `output` |
| `gen_ai.conversation.id` | every turn | every span | no (`traceloop.association.properties.session_id` on the turn) |
| `gen_ai.provider.name` (Ollama / Groq) | `ollama` / `api.groq.com` | `openai` / `groq` | `openai` / `openai` |
| `error.type` on a 429 | `429` | none | `RateLimitError` |
| Token usage / duration metrics | yes | none | yes |
| Turn, tool and outcome metrics | yes | none | none |
| Turn totals on the turn span | yes | no | no |

All three keep `agent1.turn.outcome` on the turn span, and the weather tools' `GET` spans come from `instrumentation.py` in all three.

## Known limitations

- **Small CPU model.** `qwen2.5:3b` on a CPU is slow, often 5–20 s per call, and sometimes calls tools sloppily. That is useful for learning, but the results aren't reliable.
- **Rollback only affects the history.** When a turn fails (cap reached or model server unreachable), it's removed from the history, but tools that already ran are not undone.
- **No quality signal.** `answered` only means the model stopped asking for tools. With `(10+20)*30 + ((20+30)*(30+40))`, `qwen2.5:3b` guessed intermediate values it hadn't computed yet, then ended with "Let's perform step 2." and made no further call. That was recorded as `outcome=answered` with every span green. A bigger model (`LLM_MODEL=qwen2.5:7b`, or a hosted one) or a stricter system prompt helps. Detecting the failure needs a real quality check, such as an LLM judge or user feedback.
- **Limits stop the agent; they don't make it work.** They turn a hang into a quick, visible failure. They don't make the model plan better.
- **Timeouts are per call, not per turn.** A turn can still take up to `MAX_STEPS + 1` calls × `MODEL_TIMEOUT_S`.
- **Message format.** `gen_ai.input.messages` / `gen_ai.output.messages` hold the OpenAI message format, not the semantic-conventions "parts" structure.
