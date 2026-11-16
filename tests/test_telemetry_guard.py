"""Telemetry aggregation, demo guards, and eval scoring/gating."""
from __future__ import annotations

import pytest

import guard
import scoring
from telemetry import Metrics, Trace, percentile, summarize


def test_percentile_interpolates():
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile(list(range(1, 101)), 95) == pytest.approx(95.05)


def test_trace_accumulates_stages_and_finishes_once():
    t = Trace(mode="agent")
    with t.stage("retrieval"):
        pass
    with t.stage("retrieval"):
        pass
    t.record_llm("generation", "m", 100, 10, 0.5)
    t.record_llm("agent_plan", "m", 50, 5, 0.2)
    m = Metrics()
    first = t.finish(m, abstained=False)
    second = t.finish(m, abstained=True)
    assert first is second and m.summary()["window"] == 1
    assert first["tokens_in"] == 150 and first["llm_calls"] == 2
    assert set(t.meta()["stages_ms"]) == {"retrieval"}


def _row(lat, mode="rag", **kw):
    return {"latency_s": lat, "mode": mode, "tokens_in": 1000, "tokens_out": 100,
            "llm_calls": 1, "stages_s": {"generation": lat * 0.9}, **kw}


def test_summarize_separates_modes_and_excludes_cached_from_token_averages():
    rows = [_row(1.0), _row(2.0, unsupported_citations=1), _row(0.01, cached=True,
            tokens_in=0, tokens_out=0), _row(5.0, mode="agent", tokens_in=3000)]
    s = summarize(rows, cost_fn=lambda i, o: i * 1e-6 + o * 1e-6)
    assert s["rag"]["answered"] == 2 and s["rag"]["cached"] == 1
    assert s["rag"]["tokens_in_avg"] == 1000 and s["agent"]["tokens_in_avg"] == 3000
    assert s["rag"]["citation_flag_rate"] == 0.5
    assert s["rag"]["est_cost_per_answer_usd"] == pytest.approx(0.0011)


def test_prometheus_exposition():
    m = Metrics()
    m.observe(_row(1.5))
    m.observe(_row(0.2, abstained=True))
    text = m.prometheus()
    assert 'rag_requests_total{mode="rag",outcome="answered"} 1' in text
    assert 'rag_requests_total{mode="rag",outcome="abstained"} 1' in text
    assert 'rag_request_latency_seconds_bucket{le="2"} 2' in text
    assert 'rag_request_latency_seconds_bucket{le="1"} 1' in text
    assert "rag_request_latency_seconds_count 2" in text


def test_rate_limiter_sliding_window():
    rl = guard.RateLimiter(2)
    assert rl.allow("a", now=0) and rl.allow("a", now=1)
    assert not rl.allow("a", now=2)
    assert rl.allow("b", now=2)            # per client
    assert rl.allow("a", now=60.5)         # window slid
    assert guard.RateLimiter(0).allow("a")  # disabled


def test_daily_cap_resets_each_day():
    cap = guard.DailyCap(1)
    assert cap.take("2026-01-01") and not cap.take("2026-01-01")
    assert cap.take("2026-01-02")


def test_answer_cache_lru_and_ttl(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(guard.time, "monotonic", lambda: clock[0])
    c = guard.AnswerCache(size=2, ttl_s=10)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1
    c.put("c", 3)                     # evicts b (least recently used)
    assert c.get("b") is None and c.get("c") == 3
    clock[0] = 11
    assert c.get("a") is None         # expired
    assert guard.AnswerCache.normalize_question(" Is THIS ok? ") == "is this ok"


# --------------------------------------------------------------------------- eval scoring

def _results():
    return [
        {"id": "q01", "type": "in_scope", "recall_hit": True, "citation_correct": True,
         "hallucinated_authorities": 0, "latency_s": 1.0, "llm_calls": 1,
         "tokens_in": 1000, "tokens_out": 100},
        {"id": "q02", "type": "in_scope", "recall_hit": False, "citation_correct": False,
         "hallucinated_authorities": 2, "latency_s": 3.0, "llm_calls": 1,
         "tokens_in": 1200, "tokens_out": 120},
        {"id": "q18", "type": "in_scope", "skipped": "expected evidence not in corpus"},
        {"id": "oos01", "type": "out_of_scope", "correctly_abstained": True},
        {"id": "ma01", "type": "must_answer", "correctly_answered": None},
    ]


def test_summarize_eval_results():
    s = scoring.summarize(_results())
    assert s["recall"] == {"n": 1, "d": 2, "rate": 0.5}
    assert s["hallucination"]["rate"] == 0.5 and s["skipped"] == 1
    assert s["must_answer"]["d"] == 0          # None = not scoreable, not a failure
    assert s["latency_p95_s"] == pytest.approx(2.9)
    assert scoring.fmt(s["recall"]) == "1/2 (50%)"


def test_check_flags_bounds_and_per_question_regressions():
    results = _results()
    s = scoring.summarize(results)
    spec = {"recall": {"min": 0.75}, "hallucination": {"max": 0.15},
            "oos_abstention": {"min_n": 1}, "must_answer": {"min": 1.0}}
    baseline = [{"id": "q02", "recall_hit": True}, {"id": "q01", "recall_hit": True}]
    failures = scoring.check(s, results, spec, baseline)
    assert any(f.startswith("recall:") for f in failures)
    assert any(f.startswith("hallucination:") for f in failures)
    assert any(f.startswith("q02:") for f in failures)
    assert not any(f.startswith(("q01:", "oos_abstention", "must_answer")) for f in failures)
