# Next time

Parked 2026-11-12. Branch `openai-compatible-api` at `b1bf730`. Vasilis handles commits, branches and PRs himself. **The work from 2026-11-12 is uncommitted** in `agent.py`, `instrumentation.py`, `telemetry.py` and `README.md`.

## Where we are
- This is a learning project: a minimal agent, built by hand step by step, with no framework and raw `urllib`. Vasilis wrote steps 2–4 from specs, with reviews from Claude. Claude wrote step 5 (`agent.py`: the agent plus telemetry) at Vasilis's request.
- Step 5 speaks the OpenAI-compatible `/chat/completions` API. **It works against Groq with `LLM_MODEL=openai/gpt-oss-20b`.** Configuration: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_PROVIDER`, `LLM_MAX_TOKENS`.

### Done 2026-11-12 (uncommitted)
- **Weather tools:** `geocode` and `get_weather` call Open-Meteo, with no key needed. Outbound calls get `GET` spans through `@instr.http_call`. Claude tested both tools live, outside the agent loop; they work. In the README, the "no match" example is now `Xyzzyville`, because Open-Meteo does find an "Atlantis".
- **Retry on 429:** `chat_with_retry()` waits as long as the `Retry-After` header says (5 s if the header is missing or not a number), retries up to 3 times per call, and gives up if a single wait would exceed 30 s. In the trace, each attempt is its own `chat` span. The turn span gets one `rate_limit_wait` event per wait, plus `agent1.turn.rate_limit_retries` and `agent1.turn.rate_limit_wait_s` (shown as `retries=` and `waited_s=` in the tree). This was tested only with simulated 429s, not against Groq.
- **`LLM_MAX_TOKENS` default is now 1000** (it was 2000). It's a guess that Groq counts `max_tokens` against its tokens-per-minute limit of 8000: a 429 showed `Requested 2397` while `max_tokens` was 2000. Groq's docs don't say so.
- **`MAX_STEPS` is now 15** (it was 5). However, `MAX_TOOL_CALLS` is still 10, so a turn gives up after 10 rounds at most. Claude asked whether to raise it, to about 30, and got no answer yet.

## Pick up with
1. **Live run on Groq** of the weather prompts in the README. Check that retries happen, whether `Requested` in a 429 dropped by about 1000, and whether turns end `truncated` now that the token limit is 1000.
2. Decide on `MAX_TOOL_CALLS`. Raising it to about 30 would let a turn actually reach 15 rounds.
3. **Dynatrace: no data seen yet.** `OTEL_EXPORTER_OTLP_ENDPOINT` must be `https://nzu34348.live.dynatrace.com/api/v2/otlp`. Without the `/api/v2/otlp` suffix, the exporter reports SUCCESS even though nothing arrives. Then run, `quit`, and look for service `agent1`. On the tenant, the `gen_ai.*` and `agent1.*` span attributes still need to be allow-listed.
4. Offered, not yet decided: make `telemetry.py` warn at startup when a Dynatrace endpoint doesn't end in `/api/v2/otlp`.
5. **The false "answered":** on `(10+20)*30 + ((20+30)*(30+40))`, qwen2.5:3b guessed intermediate values, stopped early, and the turn was recorded as `answered`. Re-test it with gpt-oss-20b.
6. **Step 6:** real tools on a real task, such as summarising the files in a folder. Rollback doesn't undo what tools did, so side effects matter there. Add a span per tool round (`run_round()` plus a decorator), as agreed. The weather tools were a first go at external tools.
7. **Step 7:** guardrails, human approval before risky tools, and prompt injection. Tool results such as weather data go straight into the model's context.

## Possible follow-ups (offered, not asked for)
- A deadline per turn: the timeout is per call, and with `MAX_STEPS` at 15 a turn can take much longer.
- Apply the step 5 limits to `step4_agent.py`. It was left as a lesson snapshot.
- Remove the `anthropic` package's leftover dependencies from `.venv`.
- Optional comparison: the `ollama` library with OpenLLMetry, diffing its spans against the hand-written ones.

## Environment notes
- **Untracked dotfiles in the repo** (`.bashrc`, `.gitconfig`, `.mcp.json`, `.claude/`, `.idea`, `.vscode`, …), probably from the sandbox. They haven't been looked into. Check them before running any `git add .`.
- Claude's sandbox can reach Open-Meteo (with the domain allowed), but not `localhost` (Ollama) or Groq with Vasilis's key, so live agent runs need Vasilis. Env vars (`LLM_*`, `OTEL_*`) live only in his terminal; keys are never pasted into the chat.
- `uv` inside the sandbox needs `UV_CACHE_DIR="$TMPDIR/uv-cache"`.
- Use 4-space indentation (global CLAUDE.md).
