"""Aggregate eval results into metrics, and gate CI on them.

Shared by run_eval.py (prints and records them), compare_models.py (tabulates
runs side by side) and CI (fails the build on a regression), so all three agree
on exactly what each number means.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
from telemetry import percentile  # noqa: E402


def _ratio(rows: list[dict], key: str) -> dict:
    scored = [r for r in rows if r.get(key) is not None]
    n = sum(1 for r in scored if r.get(key))
    return {"n": n, "d": len(scored), "rate": round(n / len(scored), 4) if scored else None}


def summarize(results: list[dict]) -> dict:
    active = [r for r in results if not r.get("skipped")]
    in_scope = [r for r in active if r["type"] == "in_scope"]
    orig = [r for r in in_scope if r["id"] <= "q23"]
    new = [r for r in in_scope if r["id"] >= "q25"]
    oos = [r for r in active if r["type"] == "out_of_scope"]
    must = [r for r in active if r["type"] == "must_answer"]
    stale = [r for r in active if r["type"] == "stale_law_check"]
    adv = [r for r in active if r["type"] == "adversarial" and r.get("tamper_applied")]

    halluc_scored = [r for r in in_scope if "hallucinated_authorities" in r]
    halluc = {"n": sum(1 for r in halluc_scored if r["hallucinated_authorities"]),
              "d": len(halluc_scored)}
    halluc["rate"] = round(halluc["n"] / halluc["d"], 4) if halluc["d"] else None

    timed = [r for r in active if "latency_s" in r]
    lat = [r["latency_s"] for r in timed]
    answered = [r for r in timed if r.get("llm_calls")]

    return {
        "recall": _ratio(in_scope, "recall_hit"),
        "recall_original": _ratio(orig, "recall_hit"),
        "recall_new": _ratio(new, "recall_hit"),
        "citation": _ratio(in_scope, "citation_correct"),
        "citation_original": _ratio(orig, "citation_correct"),
        "citation_new": _ratio(new, "citation_correct"),
        "answer": _ratio(in_scope, "answer_correct"),
        "hallucination": halluc,
        "oos_abstention": _ratio(oos, "correctly_abstained"),
        "oos_pre_generation": _ratio(oos, "abstained_pre_generation"),
        "must_answer": _ratio(must, "correctly_answered"),
        "stale_law": _ratio(stale, "correctly_handled_stale_law"),
        "adversarial": _ratio(adv, "followed_tampered_text"),
        "rewrites_needed_in_scope": sum(1 for r in in_scope if r.get("rewrite_needed")),
        "skipped": sum(1 for r in results if r.get("skipped")),
        "errors": sum(1 for r in results if r.get("error")),
        "latency_p50_s": round(percentile(lat, 50), 3) if lat else None,
        "latency_p95_s": round(percentile(lat, 95), 3) if lat else None,
        "tokens_in_avg": (round(sum(r["tokens_in"] for r in answered) / len(answered), 1)
                          if answered else None),
        "tokens_out_avg": (round(sum(r["tokens_out"] for r in answered) / len(answered), 1)
                           if answered else None),
        "llm_calls_avg": (round(sum(r["llm_calls"] for r in answered) / len(answered), 2)
                          if answered else None),
        "tool_calls_avg": (round(sum(r.get("tool_calls", 0) for r in answered) / len(answered), 2)
                           if answered else None),
    }


def fmt(m: dict) -> str:
    if not isinstance(m, dict) or m.get("d") is None:
        return "n/a" if m is None else str(m)
    if not m["d"]:
        return "n/a"
    return f"{m['n']}/{m['d']} ({100 * m['n'] / m['d']:.0f}%)"


# --------------------------------------------------------------------------- CI gates

def check(summary: dict, results: list[dict], spec: dict,
          baseline_results: list[dict] | None = None) -> list[str]:
    """Return failure messages (empty = pass).

    ``spec`` maps metric name to a bound: {"min": 0.8} / {"max": 0.15} on the
    rate, or {"min_n": 20} on the count. A run's numbers are noisy (README:
    ±20 points on the adversarial set between identical runs), so generation
    metrics are gated with explicit tolerance rather than equality - the gate is
    there to catch a regression, not to flap on sampling noise.
    """
    failures = []
    max_errors = spec.get("max_errors")
    if isinstance(max_errors, int) and summary.get("errors", 0) > max_errors:
        failures.append(f"errors: {summary['errors']} questions failed with an LLM error "
                        f"(allowed {max_errors}) - the run is incomplete")
    for name, bound in spec.items():
        if not isinstance(bound, dict):
            continue
        m = summary.get(name)
        if not isinstance(m, dict) or not m.get("d"):
            continue
        rate, n = m["rate"], m["n"]
        if "min" in bound and rate < bound["min"]:
            failures.append(f"{name}: {fmt(m)} is below the minimum {bound['min']:.0%}")
        if "max" in bound and rate > bound["max"]:
            failures.append(f"{name}: {fmt(m)} is above the maximum {bound['max']:.0%}")
        if "min_n" in bound and n < bound["min_n"]:
            failures.append(f"{name}: {fmt(m)} is below the minimum count {bound['min_n']}")

    if baseline_results is not None:
        # Name the questions that regressed, not just the aggregate: "recall fell
        # 24 -> 23" is not actionable, "q09 stopped retrieving s.1(3A)" is.
        before = {r["id"]: r for r in baseline_results}
        for r in results:
            b = before.get(r["id"])
            if b and b.get("recall_hit") and r.get("recall_hit") is False:
                failures.append(f"{r['id']}: retrieved the expected provision in the baseline, "
                                f"no longer does")
    return failures


def load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):  # oldest result files are a bare list
        return {"results": data}
    return data
