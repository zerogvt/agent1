# Next time

Parked 2026-11-02. Current branch: `openai-compatible-api` at `b8d51ee`, pushed to `origin` (github.com/zerogvt/agent1). `main` is still at `0c1d6c6`. **The PR isn't open yet:** open it from https://github.com/zerogvt/agent1/pull/new/openai-compatible-api (Claude drafted a description in the session).

## Where we are
- This is a learning project: a minimal agent, built by hand step by step, with no framework and raw `urllib`. Vasilis wrote steps 2–4 from specs, with reviews from Claude. Claude wrote step 5 (telemetry) at Vasilis's request.
- **This session (branch `openai-compatible-api`):** step 5 now speaks the OpenAI-compatible `/chat/completions` API, so it runs on a hosted free tier. Local Ollama was too slow.
  - Configuration: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`, and optional `LLM_PROVIDER`. The default is local Ollama at `http://localhost:11434/v1`. Steps 1–4 still use Ollama's `/api/chat` with `OLLAMA_*`.
  - **Works against Groq with `LLM_MODEL=openai/gpt-oss-20b`.** Vasilis confirmed it. The key does *not* list `llama-3.3-70b-versatile` (404), even though Groq's docs still list it. The reason is unknown.
  - New outcomes: `rate_limited` (429) and `api_error` (other HTTP errors). A custom `User-Agent` is sent, because Cloudflare blocks `Python-urllib` with 403 / error 1010. Only `role`, `content` and `tool_calls` are re-sent in the history. The `ollama.*` span attributes are gone, and HTTP errors are recorded as `error.type` = the status code.
- **Dynatrace: not seen working yet.** The cause was found: `OTEL_EXPORTER_OTLP_ENDPOINT` was set to `https://nzu34348.live.dynatrace.com` without `/api/v2/otlp`. The exporter counts any 2xx as success, so a login-page redirect showed as `SUCCESS`. Vasilis has to set it to `https://nzu34348.live.dynatrace.com/api/v2/otlp`, then run, `quit`, and look for service `agent1`. The token header (122 chars) looked right.
  - Check script: `/tmp/claude-1000/-home-v-git-linux-agent1/57c253f2-8fe7-4ee1-b613-21420b39f833/scratchpad/otlp_check.py`. It sends one test span and one test metric and now warns about a missing `/api/v2/otlp`. It's in the scratchpad, so it may be gone. Recreate it if needed.
  - Still to do on the tenant: allow-list the `gen_ai.*` and `agent1.*` span attributes. `ollama.*` is no longer needed.

## Pick up with
1. Confirm that traces and metrics appear in Dynatrace with the fixed endpoint.
2. Open the PR for `openai-compatible-api` and merge it.
3. Offered, not yet decided: make `telemetry.py` warn at startup when a Dynatrace endpoint doesn't end in `/api/v2/otlp`.
4. **The false "answered"** (from before). On `(10+20)*30 + ((20+30)*(30+40))`, qwen2.5:3b guesses intermediate values and stops early, and it's recorded as `answered`. Re-test with gpt-oss-20b on Groq, which may not show it. Other options: a stricter system prompt, or an "incomplete answer" check, with an LLM judge as the proper version.
5. **Step 6:** real tools and a real task, e.g. reading and summarising files in a folder. Side effects matter there, because rollback doesn't undo what tools did. Add a per-round span (`run_round()` plus a decorator), as agreed.
6. **Step 7:** guardrails, human approval before risky tools, and prompt injection.

## Possible follow-ups (offered, not asked for)
- Apply the step 5 limits to `step4_agent.py`. It was left as a lesson snapshot.
- Remove the `anthropic` package's leftover dependencies from `.venv`.
- A deadline per turn: the timeout is per call.
- Optional comparison: the `ollama` library with OpenLLMetry, diffing its spans against the hand-written ones.

## Environment notes
- **Untracked dotfiles in the repo** (`.bashrc`, `.gitconfig`, `.mcp.json`, `.claude/`, `.idea`, `.vscode`, …) appeared during this session, probably from the sandbox. They're not committed and haven't been looked into. Check them before running any `git add .`.
- Claude's sandbox: `localhost` (Ollama) is unreachable, and running outside the sandbox was refused, so live runs need Vasilis. GitHub can be reached, but `git push` can't use the credential store, so Vasilis pushes with `! git push`.
- `!` commands run in Claude's shell and don't see the env vars of Vasilis's own terminal (`LLM_*`, `OTEL_*`). Keys stay in that terminal and are never pasted into the chat.
- `uv` inside the sandbox needs `UV_CACHE_DIR="$TMPDIR/uv-cache"`.
- Use 4-space indentation (global CLAUDE.md).
