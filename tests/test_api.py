"""End-to-end through FastAPI with a fake retriever and a scripted LLM."""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import guard
import main
import tools
from conftest import ANSWER, CHUNKS, MANIFEST, FakeRetriever, chat_reply, json_reply


@pytest.fixture
def client(monkeypatch):
    def make(distance=0.2, cache_size=0, rate=0, cap=0):
        retriever = FakeRetriever(distance=distance)
        main.set_retriever(retriever, tools.Corpus(CHUNKS, manifest=MANIFEST))
        monkeypatch.setattr(main, "CACHE", guard.AnswerCache(cache_size, 3600))
        monkeypatch.setattr(main, "RATE_LIMIT", guard.RateLimiter(rate))
        monkeypatch.setattr(main, "DAILY_CAP", guard.DailyCap(cap))
        monkeypatch.setattr(main, "METRICS", main.Metrics())
        return TestClient(main.app)
    yield make
    main.set_retriever(None)


def _events(text: str) -> list[tuple[str, dict]]:
    out = []
    for frame in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in frame.split("\n"))
        out.append((lines["event"], json.loads(lines["data"])))
    return out


def test_rag_answer_with_meta(client, provider):
    provider(json_reply(ANSWER, usage=(900, 100)))
    r = client().post("/query", json={"question": "How much notice for a rent increase?"})
    assert r.status_code == 200
    body = r.json()
    assert not body["abstained"] and "two months" in body["short_answer"]
    assert body["citations"] == ["Housing Act 1988, s.13(2)"]
    assert body["unsupported_citations"] == []
    meta = body["meta"]
    assert meta["mode"] == "rag" and meta["tokens_in"] == 900 and meta["llm_calls"] == 1
    assert {"retrieval", "generation", "verify"} <= set(meta["stages_ms"])


def test_unsupported_citation_is_flagged(client, provider):
    provider(json_reply(dict(ANSWER, applicable_law="Housing Act 1988, s.999 applies.")))
    body = client().post("/query", json={"question": "rent increase notice?"}).json()
    assert body["unsupported_citations"] == ["housing act 1988|s999"]
    assert "Automated check" in body["uncertainties"]


def test_keyword_gate_abstains_without_calling_the_model(client):
    # The autouse fixture fails the test on any LLM call.
    body = client().post("/query", json={"question": "Eviction notice in Scotland?"}).json()
    assert body["abstained"] and "Scotland" in body["scope_note"]
    assert body["meta"]["llm_calls"] == 0


def test_low_confidence_rewrites_then_abstains(client, provider):
    fake = provider(json_reply({"legal_question": "x", "search_queries": ["mortgage interest rate"]}))
    body = client(distance=0.6).post("/query", json={"question": "best mortgage rate?"}).json()
    assert body["abstained"] and len(fake.requests) == 1     # the rewrite only
    assert body["meta"]["stages_ms"].get("rewrite") is not None


def test_stream_emits_sources_before_answer(client, provider):
    provider(json_reply(ANSWER))
    r = client().post("/query/stream", json={"question": "rent increase notice?"})
    kinds = [k for k, _ in _events(r.text)]
    assert kinds == ["sources", "answer"]


def test_stream_reports_provider_failure_as_error_event(client, provider):
    import httpx
    provider(httpx.Response(503, text="down"))
    r = client().post("/query/stream", json={"question": "rent increase notice?"})
    kind, data = _events(r.text)[-1]
    assert kind == "error" and "503" in data["detail"]


def test_agent_mode_streams_steps(client, provider):
    provider(
        chat_reply(tool_calls=[("get_provision", {"act": "Housing Act 1988", "provision": "s.13(2)"})]),
        chat_reply("Found it."),
        json_reply(ANSWER))
    r = client(distance=0.6).post("/query/stream", json={"question": "rent increase notice?",
                                                         "mode": "agent"})
    events = _events(r.text)
    assert [k for k, _ in events] == ["step", "sources", "answer"]
    answer = events[-1][1]
    # One exact lookup passes the gate on its own, even though similarity
    # distances (0.6 here) would have abstained.
    assert not answer["abstained"]
    assert answer["agent_steps"][0]["tool"] == "get_provision"
    assert answer["meta"]["tool_calls"] == 1 and answer["meta"]["llm_calls"] == 3


def test_agent_repeal_reaches_the_answer_via_fallback(client, provider):
    provider(
        chat_reply(tool_calls=[("get_provision", {"act": "Housing Act 1988", "provision": "s.21"})]),
        chat_reply("Repealed."),
        json_reply(ANSWER))
    body = client().post("/query", json={"question": "How long is a s.21 notice?",
                                         "mode": "agent"}).json()
    # No evidence from the agent -> ordinary retrieval; the repeal it found is
    # still stated, independently of the model.
    assert not body["abstained"]
    assert "Housing Act 1988, s.21" in body["uncertainties"]


def test_cache_serves_repeat_without_llm(client, provider):
    fake = provider(json_reply(ANSWER))
    c = client(cache_size=10)
    first = c.post("/query", json={"question": "Rent increase notice?"}).json()
    second = c.post("/query", json={"question": "  rent increase NOTICE "}).json()
    assert len(fake.requests) == 1
    assert second["meta"]["cached"] and not first["meta"]["cached"]
    assert second["short_answer"] == first["short_answer"]


def test_rate_limit_and_daily_cap(client, provider):
    provider(json_reply(ANSWER))
    c = client(rate=1)
    assert c.post("/query", json={"question": "a rent question"}).status_code == 200
    assert c.post("/query", json={"question": "another rent question"}).status_code == 429
    c = client(cap=1)
    assert c.post("/query", json={"question": "a rent question"}).status_code == 200
    r = c.post("/query", json={"question": "another rent question"})
    assert r.status_code == 429 and "daily" in r.json()["detail"]


def test_question_length_is_bounded(client):
    assert client().post("/query", json={"question": "x" * 2001}).status_code == 422
    assert client().post("/query", json={"question": ""}).status_code == 422


def test_health_and_metrics(client, provider):
    provider(json_reply(ANSWER, usage=(900, 100)))
    c = client()
    c.post("/query", json={"question": "rent increase notice?"})
    health = c.get("/health").json()
    assert health["provider"] == "groq" and health["corpus"]["acts"] == 3
    assert "api_key" not in health
    prom = c.get("/metrics").text
    assert 'rag_requests_total{mode="rag",outcome="answered"} 1' in prom
    assert 'rag_llm_tokens_total{direction="in"} 900' in prom
    summary = c.get("/metrics/summary").json()
    assert summary["all"]["answered"] == 1 and summary["all"]["tokens_in_avg"] == 900
