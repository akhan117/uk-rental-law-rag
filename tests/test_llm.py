"""Provider adapters: config, structured-output ladder, retries, tool calls."""
from __future__ import annotations

import json

import httpx
import pytest

import llm
from conftest import chat_reply, json_reply
from telemetry import Trace

SCHEMA = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}


# --------------------------------------------------------------------------- config

def test_groq_preset_defaults_and_key():
    cfg = llm.config_from_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "gk"})
    assert cfg.base_url == "https://api.groq.com/openai/v1"
    assert cfg.model == "openai/gpt-oss-20b"
    assert cfg.api_key == "gk"
    assert cfg.reasoning_effort == "low"   # gpt-oss: keep hidden reasoning tokens down
    assert cfg.hosted


def test_generic_key_overrides_preset_key():
    cfg = llm.config_from_env({"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "a",
                               "LLM_API_KEY": "b", "LLM_MODEL": "gemini-x"})
    assert cfg.api_key == "b" and cfg.model == "gemini-x" and cfg.reasoning_effort == ""


def test_ollama_keeps_original_rag_model_variable():
    cfg = llm.config_from_env({"RAG_MODEL": "qwen3:8b", "OLLAMA_URL": "http://o:11434/"})
    assert cfg.provider == "ollama" and cfg.model == "qwen3:8b"
    assert cfg.base_url == "http://o:11434" and not cfg.hosted


def test_bad_provider_and_incomplete_generic_config_raise():
    with pytest.raises(ValueError):
        llm.config_from_env({"LLM_PROVIDER": "nope"})
    with pytest.raises(ValueError):
        llm.config_from_env({"LLM_PROVIDER": "openai"})


def test_public_config_never_exposes_key():
    cfg = llm.config_from_env({"LLM_PROVIDER": "groq", "GROQ_API_KEY": "secret"})
    assert "secret" not in json.dumps(cfg.public())


# --------------------------------------------------------------------------- JSON parsing

@pytest.mark.parametrize("text", [
    '{"a": "x"}',
    '```json\n{"a": "x"}\n```',
    'Here is the answer:\n{"a": "x"}\nHope that helps.',
])
def test_parse_json_object_tolerates_wrappers(text):
    assert llm.parse_json_object(text) == {"a": "x"}


@pytest.mark.parametrize("text", ["", "not json", "[1, 2]"])
def test_parse_json_object_rejects_non_objects(text):
    assert llm.parse_json_object(text) is None


def test_strict_schema_adds_additional_properties_everywhere():
    nested = {"type": "object", "properties": {
        "a": {"type": "string"},
        "b": {"type": "object", "properties": {"c": {"type": "string"}}}}}
    strict = llm._strict_schema(nested)
    assert strict["additionalProperties"] is False
    assert strict["required"] == ["a", "b"]
    assert strict["properties"]["b"]["additionalProperties"] is False
    assert "additionalProperties" not in nested   # input untouched


# --------------------------------------------------------------------------- OpenAI-compatible

def test_complete_json_sends_strict_schema_and_records_usage(provider):
    fake = provider(json_reply({"a": "x"}, usage=(120, 30)))
    trace = Trace()
    result = llm.complete_json("sys", "user", SCHEMA, name="t", trace=trace)
    assert result.parsed == {"a": "x"}
    body = fake.requests[0]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["strict"] is True
    assert body["messages"][0] == {"role": "system", "content": "sys"}
    assert trace.tokens_in == 120 and trace.tokens_out == 30
    assert trace.llm_calls[0].stage == "generation"


def test_reasoning_models_get_token_headroom(provider):
    fake = provider(json_reply({"a": "x"}), reasoning_effort="low")
    llm.complete_json("s", "u", SCHEMA, max_tokens=900)
    assert fake.requests[0]["reasoning_effort"] == "low"
    assert fake.requests[0]["max_tokens"] == 900 + 1024


def test_json_mode_ladder_degrades_and_remembers(provider):
    reject = httpx.Response(400, json={"error": {"message": "response_format json_schema "
                                                            "is not supported for this model"}})
    fake = provider(reject, reject, json_reply({"a": "x"}), json_reply({"a": "y"}))
    assert llm.complete_json("s", "u", SCHEMA).parsed == {"a": "x"}
    modes = [b.get("response_format", {}).get("type") for b in fake.requests]
    assert modes == ["json_schema", "json_schema", "json_object"]
    assert fake.requests[0]["response_format"]["json_schema"]["strict"] is True
    assert fake.requests[1]["response_format"]["json_schema"]["strict"] is False
    # The next call starts at the rung that worked - no repeated rejections.
    llm.complete_json("s", "u", SCHEMA)
    assert fake.requests[3]["response_format"]["type"] == "json_object"


def test_last_rung_puts_schema_in_prompt(provider):
    fake = provider(json_reply({"a": "x"}), json_mode="none")
    llm.complete_json("s", "u", SCHEMA)
    assert "response_format" not in fake.requests[0]
    assert '"required": ["a"]' in fake.requests[0]["messages"][1]["content"]


def test_schema_validation_failure_is_bad_output_not_outage(provider):
    bad = httpx.Response(400, json={"error": {"code": "json_validate_failed",
                                              "message": "json_validate_failed"}})
    fake = provider(bad, bad)
    result = llm.complete_json("s", "u", SCHEMA)
    assert result.parsed is None
    assert len(fake.requests) == 2   # retried once, didn't downgrade the mode
    assert llm.get_config().json_mode == "schema_strict"


def test_rate_limit_waits_for_retry_after(provider, monkeypatch):
    waits = []
    monkeypatch.setattr(llm, "_sleep", waits.append)
    limited = httpx.Response(429, headers={"retry-after": "3"}, json={"error": "slow down"})
    provider(limited, json_reply({"a": "x"}))
    assert llm.complete_json("s", "u", SCHEMA).parsed == {"a": "x"}
    assert waits == [3.0]


def test_persistent_server_error_raises_llm_error(provider):
    provider(httpx.Response(503, text="down"))
    with pytest.raises(llm.LLMError, match="503"):
        llm.complete_json("s", "u", SCHEMA)


def test_auth_header_sent(provider):
    seen = {}
    original = provider(json_reply({"a": "x"})).handler

    def handler(request):
        seen["auth"] = request.headers.get("authorization")
        return original(request)

    llm.configure(llm.get_config(), transport=httpx.MockTransport(handler))
    llm.complete_json("s", "u", SCHEMA)
    assert seen["auth"] == "Bearer k"


def test_chat_parses_tool_calls(provider):
    fake = provider(chat_reply(tool_calls=[("get_provision", {"act": "HA 1988", "provision": "s.8"})]))
    result = llm.chat([{"role": "user", "content": "q"}], tools=[{"type": "function"}])
    assert result.tool_calls[0].name == "get_provision"
    assert result.tool_calls[0].arguments == {"act": "HA 1988", "provision": "s.8"}
    assert fake.requests[0]["tool_choice"] == "auto"


# --------------------------------------------------------------------------- Ollama

def test_ollama_generate_payload_and_usage(provider):
    fake = provider({"response": '{"a": "x"}', "prompt_eval_count": 50, "eval_count": 7},
                    provider="ollama", base_url="http://ollama.test", model="qwen3:4b")
    trace = Trace()
    result = llm.complete_json("sys", "user", SCHEMA, max_tokens=900, repeat_penalty=1.15,
                               trace=trace)
    body = fake.requests[0]
    assert fake.paths[0] == "/api/generate"
    assert body["think"] is False and body["format"] == SCHEMA
    assert body["options"] == {"temperature": 0.1, "num_predict": 900, "repeat_penalty": 1.15}
    assert result.parsed == {"a": "x"} and trace.tokens_in == 50 and trace.tokens_out == 7


def test_ollama_chat_translates_tool_messages(provider):
    fake = provider({"message": {"content": "", "tool_calls": [
        {"function": {"name": "search_law", "arguments": {"query": "deposit"}}}]},
        "prompt_eval_count": 5, "eval_count": 2},
        provider="ollama", base_url="http://ollama.test", model="qwen3:4b")
    history = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "search_law", "arguments": '{"query": "rent"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
    ]
    result = llm.chat(history, tools=[{"type": "function"}])
    sent = fake.requests[0]["messages"]
    assert fake.paths[0] == "/api/chat"
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == {"query": "rent"}
    assert sent[1]["content"] == "" and "tool_call_id" not in sent[2]
    assert result.tool_calls[0].arguments == {"query": "deposit"}
    assert result.tool_calls[0].id.startswith("call_")


def test_hosted_warm_up_makes_no_call():
    # The autouse fixture's transport fails on any request.
    assert llm.warm_up() == {"ready": True, "gpu": None}
