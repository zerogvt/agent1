"""Step 5: the agent loop, with traces and metrics.

Same agent as step 4. Telemetry is attached with decorators (see instrumentation.py)
so the agent code stays readable; every user turn becomes a trace:

  invoke_agent agent1      one user turn: outcome, model calls, tool calls, tokens
    chat qwen2.5:3b        one model call: tokens, finish reason
    execute_tool geocode   one tool call: arguments, result, error
      GET                  the tool's outbound HTTP call: server, status (external tools only)
    chat qwen2.5:3b

feature: external tools — geocode and get_weather call Open-Meteo over HTTP.

Run:  .venv/bin/python agent.py
      DEBUG=1 .venv/bin/python agent.py     # also dump raw request/response JSON
      OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=false ...   # keep prompts/results off spans
See telemetry.py for exporting to Dynatrace.

Unlike steps 1-4 (Ollama's own /api/chat), this step speaks the OpenAI-compatible
chat completions API, so the same code runs against local Ollama or a hosted service:

  LLM_BASE_URL=https://api.groq.com/openai/v1 LLM_API_KEY=... LLM_MODEL=... .venv/bin/python agent.py
"""

import datetime
import json
import os
# feature: prompt history — readline is Python's standard-library wrapper around GNU readline
# (Linux/macOS). Importing it is enough to give input() arrow-key history and line editing.
import readline
import time
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
MAX_STEPS = 15  # tool rounds per turn
# One reply can hold many tool calls (a small model can repeat a block of calls
# dozens of times in one reply), so rounds alone don't bound the work.
MAX_TOOL_CALLS = 10  # tool calls per turn
# Per model call, sent as max_tokens. Ollama's default is "until the context is full".
# Kept modest: Groq seems to count it against the tokens-per-minute limit up front.
MAX_REPLY_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "1000"))
MODEL_TIMEOUT_S = 120  # per model call; without it a runaway reply hangs the agent
# On HTTP 429, wait as long as the server's Retry-After header says, then try again.
MAX_RATE_LIMIT_RETRIES = 3  # per model call
MAX_RATE_LIMIT_WAIT_S = 30  # a longer wait (e.g. a daily limit) ends the turn instead
DEFAULT_RATE_LIMIT_WAIT_S = 5  # when the 429 has no usable Retry-After header
AGENT_NAME = "agent1"

# feature: external tools — Open-Meteo (open-meteo.com): free weather and geocoding
# APIs, no key, free for non-commercial use. Requests go out over plain urllib.
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
EXTERNAL_TIMEOUT_S = 10  # per HTTP call a tool makes; a slow service should not stall the turn
MAX_FORECAST_DAYS = 7

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
    # feature: external tools — two tools backed by Open-Meteo. The model has to chain
    # them: geocode a place name, then pass the coordinates to get_weather.
    {
        "type": "function",
        "function": {
            "name": "geocode",
            "description": (
                "Find places by name and return up to 5 matches, each with latitude, longitude, "
                "country and region. Use it to get coordinates for get_weather. If several "
                "places match, pick the likeliest or ask the user."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Place name only, e.g. 'Thessaloniki'"},
                    "country_code": {
                        "type": "string",
                        "description": "Optional ISO 3166-1 alpha-2 code to narrow the search, e.g. 'GR'",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Current weather and a daily forecast for a latitude/longitude. "
                "Temperatures in °C, precipitation in mm, wind in km/h, times in local time."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "latitude": {"type": "number"},
                    "longitude": {"type": "number"},
                    "days": {
                        "type": "integer",
                        "description": f"Forecast days including today, 1-{MAX_FORECAST_DAYS}. Default 3",
                    },
                },
                "required": ["latitude", "longitude"],
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


# feature: external tools — everything from here to TOOLS_FUNC.

# WMO weather interpretation codes, as listed in Open-Meteo's docs. Text reads better
# than a bare number, for the model and in the trace.
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog",
    51: "light drizzle", 53: "moderate drizzle", 55: "dense drizzle",
    56: "light freezing drizzle", 57: "dense freezing drizzle",
    61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    66: "light freezing rain", 67: "heavy freezing rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 77: "snow grains",
    80: "slight rain showers", 81: "moderate rain showers", 82: "violent rain showers",
    85: "slight snow showers", 86: "heavy snow showers",
    95: "thunderstorm", 96: "thunderstorm with slight hail", 99: "thunderstorm with heavy hail",
}


@instr.http_call  # feature: telemetry
def http_get_json(url):
    """GET a URL and parse the JSON body. Raises on HTTP errors, network errors and timeouts,
    so the tool fails and run_tool hands the error to the model as text."""
    req = urllib.request.Request(url, headers={"User-Agent": "agent1/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=EXTERNAL_TIMEOUT_S) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as e:
        # Open-Meteo explains a refused request in the body ({"reason": "Latitude must be ..."}).
        # Put that in the message, so the model can fix its arguments instead of seeing "Bad Request".
        try:
            e.msg = json.loads(e.read()).get("reason") or e.msg
        except ValueError:
            pass
        raise


def geocode(name, country_code=None):
    params = {"name": name, "count": 5, "language": "en", "format": "json"}
    if country_code:
        params["countryCode"] = country_code
    _, body = http_get_json(f"{GEOCODING_URL}?{urllib.parse.urlencode(params)}")
    matches = body.get("results") or []  # no "results" key at all when nothing matches
    if not matches:
        raise LookupError(f"no place found named {name!r}")
    # Keep only what the model needs: fewer tokens and less to misread.
    keep = ("name", "admin1", "country", "latitude", "longitude", "population")
    return json.dumps([{k: m[k] for k in keep if k in m} for m in matches])


def get_weather(latitude, longitude, days=3):
    days = int(days)
    if not 1 <= days <= MAX_FORECAST_DAYS:
        raise ValueError(f"days must be 1-{MAX_FORECAST_DAYS}, got {days}")
    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "current": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,"
        "precipitation_sum,precipitation_probability_max",
        "timezone": "auto",  # the place's local time, so "today" means the place's today
        "forecast_days": days,
    }
    _, body = http_get_json(f"{FORECAST_URL}?{urllib.parse.urlencode(params)}")

    current = dict(body["current"])
    current.pop("interval", None)  # the API's update interval in seconds; noise for the model
    current["weather"] = WEATHER_CODES.get(current.pop("weather_code"), "unknown")
    # The API returns daily data as columns (one list per variable). Small models
    # read one record per day more reliably, so turn the columns into rows.
    daily = body["daily"]
    forecast = []
    for i, date in enumerate(daily["time"]):
        day = {key: values[i] for key, values in daily.items() if key != "time"}
        day["weather"] = WEATHER_CODES.get(day.pop("weather_code"), "unknown")
        forecast.append({"date": date, **day})
    return json.dumps({"timezone": body.get("timezone"), "current": current, "daily": forecast})


TOOLS_FUNC = {
    "get_time": get_time,
    "add": add,
    "multiply": multiply,
    "geocode": geocode,  # feature: external tools
    "get_weather": get_weather,  # feature: external tools
}


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


def retry_after_seconds(e):
    """Seconds the server asks us to wait, from the Retry-After header of a 429.
    The header may also be an HTTP date; we don't handle that and use the default instead."""
    try:
        return max(0.0, float(e.headers.get("Retry-After")))
    except (TypeError, ValueError):
        return DEFAULT_RATE_LIMIT_WAIT_S


@instr.rate_limit_wait  # feature: telemetry
def wait_for_rate_limit(seconds):
    time.sleep(seconds)


def chat_with_retry(messages):
    """chat(), but a 429 is retried after the wait the server asks for, within limits.
    Every attempt is its own model call (and span); the last error is raised as is."""
    for retry in range(1, MAX_RATE_LIMIT_RETRIES + 2):
        try:
            return chat(messages)
        except urllib.error.HTTPError as e:
            if e.code != 429 or retry > MAX_RATE_LIMIT_RETRIES:
                raise
            wait = retry_after_seconds(e)
            if wait > MAX_RATE_LIMIT_WAIT_S:
                print(f"!! rate limited; server asks for {wait:.0f}s, more than {MAX_RATE_LIMIT_WAIT_S}s")
                raise
            print(f"!! rate limited, waiting {wait:.1f}s (retry {retry}/{MAX_RATE_LIMIT_RETRIES})")
            wait_for_rate_limit(wait)


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
            reply = chat_with_retry(messages)
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


HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent1_history")
HISTORY_LENGTH = 1000


def load_history():
    """Arrow-up recalls prompts from earlier sessions too, not only this one."""
    readline.set_history_length(HISTORY_LENGTH)
    try:
        readline.read_history_file(HISTORY_FILE)
    except FileNotFoundError:
        pass


def save_history():
    try:
        readline.write_history_file(HISTORY_FILE)
    except OSError as e:
        print(f"(could not save prompt history: {e})")


def main():
    load_history()
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
                # readline has already recorded it; drop it so arrow-up starts at a real prompt.
                last = readline.get_current_history_length()
                if last and readline.get_history_item(last) == question:
                    readline.remove_history_item(last - 1)  # get is 1-based, remove is 0-based
                break
            if not question.strip():
                continue
            run_turn(messages, question)
    finally:
        save_history()
        telemetry.shutdown()  # feature: telemetry — flush, or the last turns never arrive


if __name__ == "__main__":
    main()
