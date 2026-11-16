"""One interface over every LLM backend this project can run against.

The app was written against Ollama's native API, which is right for local use and
wrong for a free CPU-only host: Qwen3-4B on two shared vCPUs is most of a minute
per answer. Every hosted option that costs nothing (Groq, Gemini, OpenRouter)
speaks the OpenAI chat-completions dialect, and so does Ollama's /v1 endpoint, so
there are exactly two adapters here:

- ``ollama``  - Ollama's native /api/generate and /api/chat. Kept native rather
  than going through /v1 because it is the only way to pass ``think: false`` to
  Qwen3 and the sampler options (``num_predict``, ``repeat_penalty``) the README's
  measured results were produced with.
- ``openai``  - any OpenAI-compatible endpoint; ``groq``, ``gemini`` and
  ``openrouter`` are presets of it (base URL, default model, key variable).

Structured output is where providers genuinely differ, and the answer schema is
load-bearing (README Finding 3: without it the model invented keys and the user
saw a blank card; Finding 7: the citation ``enum`` makes an unretrieved authority
impossible in that field). So JSON mode degrades down a ladder instead of
failing: strict json_schema -> best-effort json_schema -> json_object -> plain
prompt. The first rung a provider rejects is remembered for the process. Losing
the constrained enum is covered downstream: generate.py filters citations to the
retrieved set in code, so that guarantee no longer depends on the provider.

Every call reports token usage and latency into the request's Trace (see
telemetry.py), which is where p95 latency and tokens/query come from.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

# --------------------------------------------------------------------------- config

PRESETS: dict[str, dict[str, str]] = {
    # Free tier, email signup, no card. gpt-oss-20b supports strict json_schema
    # (including enum) and tool calling on Groq.
    "groq": {"base_url": "https://api.groq.com/openai/v1",
             "model": "openai/gpt-oss-20b", "key_env": "GROQ_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "model": "gemini-2.5-flash", "key_env": "GEMINI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "openai/gpt-oss-20b:free", "key_env": "OPENROUTER_API_KEY"},
    # Generic OpenAI-compatible endpoint: LLM_BASE_URL and LLM_MODEL required.
    "openai": {"base_url": "", "model": "", "key_env": "LLM_API_KEY"},
}

JSON_LADDER = ["schema_strict", "schema", "json_object", "none"]


@dataclass
class LLMConfig:
    provider: str = "ollama"          # "ollama" or an OpenAI-compatible preset
    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    api_key: str = ""
    timeout: float = 600.0
    json_mode: str = "schema_strict"  # starting rung of JSON_LADDER
    reasoning_effort: str = ""        # sent only when set (gpt-oss: "low")
    max_retries: int = 4
    # USD per million tokens, for the cost-per-query estimate. Free tiers cost
    # nothing, so this is "what it would cost at list price", and says so.
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0

    @property
    def is_ollama(self) -> bool:
        return self.provider == "ollama"

    @property
    def hosted(self) -> bool:
        return not self.is_ollama

    def public(self) -> dict:
        """Safe to expose on /health: no key."""
        return {"provider": self.provider, "model": self.model, "hosted": self.hosted}


def config_from_env(env: dict[str, str] | None = None) -> LLMConfig:
    # Blank values (".env" files with "RAG_TIMEOUT=") mean "unset", not "".
    env = {k: v for k, v in (os.environ if env is None else env).items() if v.strip()}
    provider = env.get("LLM_PROVIDER", "ollama").strip().lower()
    timeout = float(env.get("RAG_TIMEOUT", "600"))

    if provider == "ollama":
        return LLMConfig(
            provider="ollama",
            # RAG_MODEL is the original variable (README model comparison); keep it.
            model=env.get("LLM_MODEL") or env.get("RAG_MODEL", "qwen3:4b"),
            base_url=env.get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
            timeout=timeout,
            json_mode="schema_strict",
            # Local: there are no rate limits to wait out, and a refused connection
            # (Ollama not running) should surface quickly rather than after backoff.
            max_retries=int(env.get("LLM_MAX_RETRIES", "1")),
            price_in_per_m=float(env.get("LLM_PRICE_IN_PER_M", "0")),
            price_out_per_m=float(env.get("LLM_PRICE_OUT_PER_M", "0")),
        )

    if provider not in PRESETS:
        raise ValueError(f"Unknown LLM_PROVIDER {provider!r}; expected ollama or one of "
                         f"{', '.join(PRESETS)}")
    preset = PRESETS[provider]
    model = env.get("LLM_MODEL") or preset["model"]
    base_url = (env.get("LLM_BASE_URL") or preset["base_url"]).rstrip("/")
    if not base_url or not model:
        raise ValueError(f"LLM_PROVIDER={provider} needs LLM_BASE_URL and LLM_MODEL set")
    api_key = env.get("LLM_API_KEY") or env.get(preset["key_env"], "")
    # gpt-oss is a reasoning model; "low" keeps hidden reasoning tokens (which are
    # billed and count against free-tier token quotas) to a minimum.
    default_effort = "low" if "gpt-oss" in model else ""
    return LLMConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        # Hosted inference is seconds, not minutes; a long timeout there only
        # turns a stuck request into a long stuck request.
        timeout=float(env.get("RAG_TIMEOUT", "60")),
        json_mode=env.get("LLM_JSON_MODE", "schema_strict"),
        reasoning_effort=env.get("LLM_REASONING_EFFORT", default_effort),
        max_retries=int(env.get("LLM_MAX_RETRIES", "4")),
        price_in_per_m=float(env.get("LLM_PRICE_IN_PER_M", "0")),
        price_out_per_m=float(env.get("LLM_PRICE_OUT_PER_M", "0")),
    )


_lock = threading.Lock()
_config: LLMConfig | None = None
_client: httpx.Client | None = None
_transport: httpx.BaseTransport | None = None
# Retry waits go through this so tests don't actually sleep.
_sleep = time.sleep


def get_config() -> LLMConfig:
    global _config
    with _lock:
        if _config is None:
            _config = config_from_env()
        return _config


def configure(config: LLMConfig | None = None,
              transport: httpx.BaseTransport | None = None) -> LLMConfig:
    """Replace the active config (tests, eval CLI). None re-reads the environment."""
    global _config, _client, _transport
    with _lock:
        _config = config or config_from_env()
        _transport = transport
        if _client is not None:
            _client.close()
        _client = None
        return _config


def _http() -> httpx.Client:
    """One pooled client per process: keep-alive saves a TLS handshake per call,
    which is a visible fraction of a hosted call's latency."""
    global _client
    cfg = get_config()
    with _lock:
        if _client is None:
            _client = httpx.Client(timeout=cfg.timeout, transport=_transport)
        return _client


# --------------------------------------------------------------------------- results

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResult:
    text: str = ""
    parsed: dict | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    latency_s: float = 0.0
    model: str = ""


class LLMError(RuntimeError):
    pass


# --------------------------------------------------------------------------- HTTP

RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}


def _retry_wait(resp: httpx.Response | None, attempt: int) -> float:
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra), 60.0)
            except ValueError:
                pass
    return min(2.0 ** attempt, 30.0)


def _post(url: str, payload: dict, headers: dict | None = None) -> httpx.Response:
    """POST with retries on rate limits and transient failures.

    Free tiers rate-limit per minute, and a 62-question eval in CI hits that
    ceiling by design, so a 429 is routine rather than exceptional: wait as long
    as the provider asks (Retry-After) and go again.
    """
    cfg = get_config()
    last: httpx.Response | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            resp = _http().post(url, json=payload, headers=headers or {})
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            if attempt >= cfg.max_retries:
                raise LLMError(f"LLM request failed: {exc}") from exc
            _sleep(_retry_wait(None, attempt))
            continue
        if resp.status_code in RETRYABLE and attempt < cfg.max_retries:
            last = resp
            _sleep(_retry_wait(resp, attempt))
            continue
        return resp
    assert last is not None
    return last


def _auth_headers(cfg: LLMConfig) -> dict:
    return {"Authorization": f"Bearer {cfg.api_key}"} if cfg.api_key else {}


def _record(trace, stage: str, result: LLMResult) -> None:
    if trace is not None:
        trace.record_llm(stage, result.model, result.usage.prompt_tokens,
                         result.usage.completion_tokens, result.latency_s)


# --------------------------------------------------------------------------- JSON parsing

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def parse_json_object(text: str) -> dict | None:
    """Parse a model's JSON reply, tolerating the wrappers models add when they
    aren't decoding under a schema (code fences, a sentence before the object)."""
    if not text:
        return None
    stripped = _FENCE.sub("", text.strip()).strip()
    for candidate in (stripped, _outermost_object(stripped)):
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _outermost_object(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if 0 <= start < end else None


def _strict_schema(schema: dict) -> dict:
    """Strict json_schema requires additionalProperties: false and every property
    required, at every object level. Our schemas already require everything; add
    the flag rather than making every caller remember it."""
    schema = json.loads(json.dumps(schema))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node.setdefault("additionalProperties", False)
                if "properties" in node:
                    node["required"] = list(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)
    return schema


def _response_format(mode: str, schema: dict, name: str) -> dict | None:
    if mode == "schema_strict":
        return {"type": "json_schema",
                "json_schema": {"name": name, "schema": _strict_schema(schema), "strict": True}}
    if mode == "schema":
        return {"type": "json_schema",
                "json_schema": {"name": name, "schema": schema, "strict": False}}
    if mode == "json_object":
        return {"type": "json_object"}
    return None


_UNSUPPORTED_HINTS = ("response_format", "json_schema", "strict", "schema", "json mode",
                      "json_object", "not supported")


def _is_format_rejection(resp: httpx.Response) -> bool:
    if resp.status_code not in (400, 422):
        return False
    body = resp.text.lower()
    # Groq returns 400 json_validate_failed when the *generation* didn't fit the
    # schema - that's a bad sample, not an unsupported feature, so don't downgrade.
    if "json_validate_failed" in body:
        return False
    return any(h in body for h in _UNSUPPORTED_HINTS)


# --------------------------------------------------------------------------- public API

def complete_json(system: str, user: str, schema: dict, *, name: str = "answer",
                  temperature: float = 0.1, max_tokens: int = 900,
                  repeat_penalty: float | None = None,
                  trace=None, stage: str = "generation") -> LLMResult:
    """One structured completion. ``parsed`` is None when the reply wasn't a JSON
    object; callers already have an honest fallback for that (generate.py)."""
    cfg = get_config()
    if cfg.is_ollama:
        result = _ollama_generate(cfg, system, user, schema, temperature, max_tokens,
                                  repeat_penalty)
    else:
        result = _openai_json(cfg, system, user, schema, name, temperature, max_tokens)
    _record(trace, stage, result)
    return result


def chat(messages: list[dict], tools: list[dict] | None = None, *,
         temperature: float = 0.0, max_tokens: int = 800,
         trace=None, stage: str = "agent") -> LLMResult:
    """A chat turn with optional tool calling. ``messages`` use the OpenAI shape
    (role/content/tool_calls/tool_call_id); the Ollama adapter translates."""
    cfg = get_config()
    if cfg.is_ollama:
        result = _ollama_chat(cfg, messages, tools, temperature, max_tokens)
    else:
        result = _openai_chat(cfg, messages, tools, temperature, max_tokens)
    _record(trace, stage, result)
    return result


# --------------------------------------------------------------------------- Ollama

def _ollama_generate(cfg: LLMConfig, system: str, user: str, schema: dict,
                     temperature: float, max_tokens: int,
                     repeat_penalty: float | None) -> LLMResult:
    options: dict = {"temperature": temperature, "num_predict": max_tokens}
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty
    payload = {"model": cfg.model, "system": system, "prompt": user, "stream": False,
               "think": False, "format": schema, "options": options}
    t0 = time.perf_counter()
    resp = _post(f"{cfg.base_url}/api/generate", payload)
    _raise_for(resp)
    data = resp.json()
    text = data.get("response", "")
    return LLMResult(
        text=text, parsed=parse_json_object(text),
        usage=Usage(data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0),
        latency_s=time.perf_counter() - t0, model=cfg.model)


def _to_ollama_messages(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            m["tool_calls"] = [
                {"function": {"name": tc["function"]["name"],
                              "arguments": _as_dict(tc["function"].get("arguments"))}}
                for tc in m["tool_calls"]
            ]
        m.pop("tool_call_id", None)
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def _ollama_chat(cfg: LLMConfig, messages: list[dict], tools: list[dict] | None,
                 temperature: float, max_tokens: int) -> LLMResult:
    payload: dict = {"model": cfg.model, "messages": _to_ollama_messages(messages),
                     "stream": False, "think": False,
                     "options": {"temperature": temperature, "num_predict": max_tokens}}
    if tools:
        payload["tools"] = tools
    t0 = time.perf_counter()
    resp = _post(f"{cfg.base_url}/api/chat", payload)
    _raise_for(resp)
    data = resp.json()
    msg = data.get("message", {}) or {}
    calls = [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}",
                      name=tc.get("function", {}).get("name", ""),
                      arguments=_as_dict(tc.get("function", {}).get("arguments")))
             for tc in msg.get("tool_calls") or []]
    return LLMResult(
        text=msg.get("content", "") or "", tool_calls=calls,
        usage=Usage(data.get("prompt_eval_count", 0) or 0, data.get("eval_count", 0) or 0),
        latency_s=time.perf_counter() - t0, model=cfg.model)


# --------------------------------------------------------------------------- OpenAI-compatible

def _openai_payload(cfg: LLMConfig, messages: list[dict], temperature: float,
                    max_tokens: int) -> dict:
    payload: dict = {"model": cfg.model, "messages": messages,
                     "temperature": temperature, "max_tokens": max_tokens}
    if cfg.reasoning_effort:
        payload["reasoning_effort"] = cfg.reasoning_effort
        # Reasoning tokens are drawn from the same budget as the answer; without
        # headroom a terse cap truncates the JSON mid-object.
        payload["max_tokens"] = max_tokens + 1024
    return payload


def _usage(data: dict) -> Usage:
    u = data.get("usage") or {}
    return Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0))


def _raise_for(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        raise LLMError(f"LLM provider returned HTTP {resp.status_code}: {resp.text[:300]}")


def _openai_json(cfg: LLMConfig, system: str, user: str, schema: dict, name: str,
                 temperature: float, max_tokens: int) -> LLMResult:
    global _config
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    t0 = time.perf_counter()
    rung = JSON_LADDER.index(cfg.json_mode) if cfg.json_mode in JSON_LADDER else 0
    validate_retries = 1
    while True:
        mode = JSON_LADDER[rung]
        payload = _openai_payload(cfg, messages, temperature, max_tokens)
        fmt = _response_format(mode, schema, name)
        if fmt:
            payload["response_format"] = fmt
        if mode == "none":
            payload["messages"] = [messages[0], {"role": "user", "content": user + (
                "\n\nReply with ONLY a JSON object matching this schema:\n" + json.dumps(schema))}]
        resp = _post(f"{cfg.base_url}/chat/completions", payload, _auth_headers(cfg))

        if _is_format_rejection(resp) and rung < len(JSON_LADDER) - 1:
            rung += 1
            # Remember for the rest of the process: every request would otherwise
            # pay a rejected round-trip first.
            with _lock:
                if _config is cfg:
                    cfg.json_mode = JSON_LADDER[rung]
            continue
        if resp.status_code == 400 and "json_validate_failed" in resp.text and validate_retries:
            validate_retries -= 1
            continue
        if resp.status_code == 400 and "json_validate_failed" in resp.text:
            # A sample that failed the schema twice: report it as unusable output
            # rather than an outage, so generate.py shows its honest fallback.
            return LLMResult(text="", parsed=None, latency_s=time.perf_counter() - t0,
                             model=cfg.model)
        _raise_for(resp)
        data = resp.json()
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        return LLMResult(text=text, parsed=parse_json_object(text), usage=_usage(data),
                         latency_s=time.perf_counter() - t0, model=data.get("model", cfg.model))


def _openai_chat(cfg: LLMConfig, messages: list[dict], tools: list[dict] | None,
                 temperature: float, max_tokens: int) -> LLMResult:
    payload = _openai_payload(cfg, messages, temperature, max_tokens)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    t0 = time.perf_counter()
    resp = _post(f"{cfg.base_url}/chat/completions", payload, _auth_headers(cfg))
    _raise_for(resp)
    data = resp.json()
    msg = (data.get("choices") or [{}])[0].get("message", {}) or {}
    calls = [ToolCall(id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                      name=tc.get("function", {}).get("name", ""),
                      arguments=_as_dict(tc.get("function", {}).get("arguments")))
             for tc in msg.get("tool_calls") or []]
    return LLMResult(text=msg.get("content") or "", tool_calls=calls, usage=_usage(data),
                     latency_s=time.perf_counter() - t0, model=data.get("model", cfg.model))


def _as_dict(arguments: Any) -> dict:
    """Tool arguments arrive as a JSON string (OpenAI) or an object (Ollama)."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str) and arguments.strip():
        try:
            value = json.loads(arguments)
            return value if isinstance(value, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


# --------------------------------------------------------------------------- warm-up

def warm_up() -> dict:
    """Load the model ahead of the first question; report whether it is on GPU.

    Only meaningful for Ollama: a hosted provider has nothing to load, and
    probing it would spend free-tier quota on every container restart.

    For Ollama, zero bytes in VRAM means it fell back to CPU, which is a ~10x
    slowdown (measured on this project: 5s -> 58s per answer). Ollama does that
    silently, and it can happen for transient reasons, so record it for the UI.
    """
    cfg = get_config()
    if cfg.hosted:
        return {"ready": True, "gpu": None}
    try:
        _http().post(f"{cfg.base_url}/api/generate", timeout=180.0, json={
            "model": cfg.model, "prompt": "ok", "stream": False,
            "think": False, "options": {"num_predict": 1}})
        ps = _http().get(f"{cfg.base_url}/api/ps", timeout=10.0).json()
        loaded = [m for m in ps.get("models", []) if m.get("name") == cfg.model]
        return {"ready": True, "gpu": bool(loaded and loaded[0].get("size_vram", 0) > 0)}
    except Exception:  # noqa: BLE001 - warming is an optimisation, never fatal
        return {"ready": False, "gpu": None}
