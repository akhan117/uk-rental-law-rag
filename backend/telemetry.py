"""Per-request tracing and rolling production metrics.

The offline eval says how good an answer is; this says what serving it cost:
latency per pipeline stage, LLM tokens, which gate fired, and - the metric that
mirrors the eval online - how often post-generation citation verification flagged
an answer. Each request produces one Trace, logged as a single JSON line
(`docker logs` and container hosts capture stdout) and folded into an in-memory
window that /metrics (Prometheus text) and /metrics/summary (JSON) read.

The trace is passed explicitly rather than held in a contextvar. The streaming
endpoint runs its generator one step at a time on threadpool workers, so a
contextvar set in one step isn't reliably visible in the next; an explicit
object can't go missing that way.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, field

log = logging.getLogger("rag.request")


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. 0.0 for an empty list."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    pos = (len(s) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


@dataclass
class LLMCall:
    stage: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


@dataclass
class Trace:
    mode: str = "rag"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    started: float = field(default_factory=time.perf_counter)
    stages: dict[str, float] = field(default_factory=dict)
    llm_calls: list[LLMCall] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)
    outcome: dict = field(default_factory=dict)
    finished: dict | None = None

    @contextmanager
    def stage(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - t0

    def record_llm(self, stage: str, model: str, prompt_tokens: int,
                   completion_tokens: int, latency_s: float) -> None:
        self.llm_calls.append(LLMCall(stage, model, prompt_tokens, completion_tokens, latency_s))

    @property
    def tokens_in(self) -> int:
        return sum(c.prompt_tokens for c in self.llm_calls)

    @property
    def tokens_out(self) -> int:
        return sum(c.completion_tokens for c in self.llm_calls)

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started

    def meta(self, provider: str = "", model: str = "", cached: bool = False) -> dict:
        """The per-response metadata block returned to the client."""
        return {
            "request_id": self.request_id,
            "mode": self.mode,
            "provider": provider,
            "model": model,
            "latency_ms": round(self.elapsed_s * 1000),
            "stages_ms": {k: round(v * 1000) for k, v in self.stages.items()},
            "llm_calls": len(self.llm_calls),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tool_calls": len(self.tool_calls),
            "cached": cached,
        }

    def finish(self, metrics: Metrics | None = None, **outcome) -> dict:
        """Close the trace once; later calls return the same summary."""
        if self.finished is not None:
            return self.finished
        self.outcome.update(outcome)
        summary = {
            "event": "rag_request",
            "request_id": self.request_id,
            "mode": self.mode,
            "latency_s": round(self.elapsed_s, 4),
            "stages_s": {k: round(v, 4) for k, v in self.stages.items()},
            "llm_calls": len(self.llm_calls),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tools": list(self.tool_calls),
            **self.outcome,
        }
        self.finished = summary
        if metrics is not None:
            metrics.observe(summary)
        log.info(json.dumps(summary, default=str))
        return summary


# Prometheus histogram buckets for end-to-end latency, in seconds. Spans a cached
# hit (~ms), a hosted answer (~1-5s) and a CPU-only local answer (~60s).
LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 21, 34, 60, 120)


class Metrics:
    """Thread-safe rolling window of request summaries plus lifetime counters."""

    def __init__(self, window: int = 2000, price_in_per_m: float = 0.0,
                 price_out_per_m: float = 0.0) -> None:
        self._lock = threading.Lock()
        self._recent: deque[dict] = deque(maxlen=window)
        self.started_at = time.time()
        self.price_in_per_m = price_in_per_m
        self.price_out_per_m = price_out_per_m
        self.requests_total: dict[tuple[str, str], int] = {}
        self.tokens_total = {"in": 0, "out": 0}
        self.latency_buckets = [0] * len(LATENCY_BUCKETS)
        self.latency_count = 0
        self.latency_sum = 0.0

    @staticmethod
    def outcome_label(s: dict) -> str:
        if s.get("error"):
            return "error"
        if s.get("cached"):
            return "cached"
        if s.get("abstained"):
            return "abstained"
        return "answered"

    def observe(self, s: dict) -> None:
        with self._lock:
            self._recent.append(s)
            key = (s.get("mode", "rag"), self.outcome_label(s))
            self.requests_total[key] = self.requests_total.get(key, 0) + 1
            self.tokens_total["in"] += s.get("tokens_in", 0)
            self.tokens_total["out"] += s.get("tokens_out", 0)
            lat = float(s.get("latency_s", 0.0))
            self.latency_count += 1
            self.latency_sum += lat
            for i, b in enumerate(LATENCY_BUCKETS):
                if lat <= b:
                    self.latency_buckets[i] += 1

    def cost_usd(self, tokens_in: float, tokens_out: float) -> float:
        return (tokens_in * self.price_in_per_m + tokens_out * self.price_out_per_m) / 1e6

    def summary(self) -> dict:
        with self._lock:
            recent = list(self._recent)
        return summarize(recent, self.cost_usd) | {
            "uptime_s": round(time.time() - self.started_at),
            "window": len(recent),
        }

    def prometheus(self) -> str:
        with self._lock:
            totals = dict(self.requests_total)
            tokens = dict(self.tokens_total)
            buckets = list(self.latency_buckets)
            count, total = self.latency_count, self.latency_sum
            recent = list(self._recent)
        flagged = sum(1 for s in recent if s.get("unsupported_citations"))
        answered = sum(1 for s in recent if self.outcome_label(s) == "answered")
        lines = [
            "# HELP rag_requests_total Queries served, by mode and outcome.",
            "# TYPE rag_requests_total counter",
        ]
        for (mode, outcome), n in sorted(totals.items()):
            lines.append(f'rag_requests_total{{mode="{mode}",outcome="{outcome}"}} {n}')
        lines += [
            "# HELP rag_llm_tokens_total LLM tokens consumed.",
            "# TYPE rag_llm_tokens_total counter",
            f'rag_llm_tokens_total{{direction="in"}} {tokens["in"]}',
            f'rag_llm_tokens_total{{direction="out"}} {tokens["out"]}',
            "# HELP rag_request_latency_seconds End-to-end query latency.",
            "# TYPE rag_request_latency_seconds histogram",
        ]
        for b, n in zip(LATENCY_BUCKETS, buckets):
            lines.append(f'rag_request_latency_seconds_bucket{{le="{b}"}} {n}')
        lines += [
            f'rag_request_latency_seconds_bucket{{le="+Inf"}} {count}',
            f"rag_request_latency_seconds_sum {total:.4f}",
            f"rag_request_latency_seconds_count {count}",
            "# HELP rag_citation_flag_ratio Share of recent answers where post-generation "
            "verification flagged an unsupported citation.",
            "# TYPE rag_citation_flag_ratio gauge",
            f"rag_citation_flag_ratio {flagged / answered if answered else 0:.4f}",
        ]
        return "\n".join(lines) + "\n"


def summarize(rows: list[dict], cost_fn=None) -> dict:
    """Aggregate request summaries (from the server or from an eval run)."""
    def block(subset: list[dict]) -> dict:
        lat = [float(r.get("latency_s", 0.0)) for r in subset]
        answered = [r for r in subset if not r.get("abstained") and not r.get("error")
                    and not r.get("cached")]
        tin = [r.get("tokens_in", 0) for r in answered]
        tout = [r.get("tokens_out", 0) for r in answered]
        stage_names = sorted({k for r in subset for k in r.get("stages_s", {})})
        out = {
            "requests": len(subset),
            "errors": sum(1 for r in subset if r.get("error")),
            "abstained": sum(1 for r in subset if r.get("abstained")),
            "cached": sum(1 for r in subset if r.get("cached")),
            "latency_p50_s": round(percentile(lat, 50), 3),
            "latency_p95_s": round(percentile(lat, 95), 3),
            "latency_p99_s": round(percentile(lat, 99), 3),
            "stage_p95_s": {
                name: round(percentile([r["stages_s"][name] for r in subset
                                        if name in r.get("stages_s", {})], 95), 3)
                for name in stage_names
            },
            "answered": len(answered),
            "tokens_in_avg": round(sum(tin) / len(tin), 1) if tin else 0,
            "tokens_out_avg": round(sum(tout) / len(tout), 1) if tout else 0,
            "llm_calls_avg": (round(sum(r.get("llm_calls", 0) for r in answered)
                                    / len(answered), 2) if answered else 0),
            "citation_flag_rate": (round(sum(1 for r in answered
                                             if r.get("unsupported_citations")) / len(answered), 3)
                                   if answered else 0),
        }
        if cost_fn is not None and answered:
            out["est_cost_per_answer_usd"] = round(
                cost_fn(out["tokens_in_avg"], out["tokens_out_avg"]), 6)
        return out

    result = {"all": block(rows)}
    for mode in sorted({r.get("mode", "rag") for r in rows}):
        result[mode] = block([r for r in rows if r.get("mode", "rag") == mode])
    return result
