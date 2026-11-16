"""Side-by-side comparison of eval runs across models and modes.

The gold set is the fixed benchmark and the model is the variable, so swapping
the base model should be a configuration change with a measured result rather
than a rebuild (Docs.pdf §6). Reads every eval_results_<model>[_agent].json in
this directory; agent-mode runs appear as their own column, so RAG vs agent on
the same model is one table.

Usage: python compare_models.py [--markdown]
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring  # noqa: E402

ROWS = [
    ("recall@6 (all in-scope)", "recall"),
    ("citation (all in-scope)", "citation"),
    ("answer correctness", "answer"),
    ("recall@6 (original 23)", "recall_original"),
    ("citation (original 23)", "citation_original"),
    ("recall@6 (new 7)", "recall_new"),
    ("citation (new 7)", "citation_new"),
    ("hallucinated authority", "hallucination"),
    ("out-of-scope abstention", "oos_abstention"),
    ("answerable not over-gated", "must_answer"),
    ("stale-law handled", "stale_law"),
    ("adversarial grounding", "adversarial"),
    ("latency p50 (s)", "latency_p50_s"),
    ("latency p95 (s)", "latency_p95_s"),
    ("tokens in / answer", "tokens_in_avg"),
    ("tokens out / answer", "tokens_out_avg"),
    ("LLM calls / answer", "llm_calls_avg"),
    ("tool calls / answer", "tool_calls_avg"),
]


def load_runs() -> dict[str, dict]:
    """Historical per-model files (eval_results_*.json) plus archived runs in
    results/, which are labelled by file name since several share a model."""
    runs = {}
    for path in sorted(HERE.glob("eval_results_*.json")):
        data = scoring.load(path)
        if data.get("retrieval_only"):
            continue
        label = data.get("model", path.stem.replace("eval_results_", ""))
        if data.get("mode") == "agent":
            label += " [agent]"
        runs[label] = scoring.summarize(data["results"])
    for path in sorted((HERE / "results").glob("*.json")):
        data = scoring.load(path)
        if not data.get("retrieval_only"):
            runs[path.stem] = scoring.summarize(data["results"])
    return runs


def _cell(v) -> str:
    if isinstance(v, dict):
        return scoring.fmt(v)
    return "n/a" if v is None else str(v)


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No eval_results_<model>.json files found. Run run_eval.py first.")
        return
    models = list(runs)
    markdown = "--markdown" in sys.argv
    if markdown:
        print("| | " + " | ".join(models) + " |")
        print("|---|" + "---|" * len(models))
        for label, key in ROWS:
            print(f"| {label} | " + " | ".join(_cell(runs[m].get(key)) for m in models) + " |")
        return
    label_w = max(len(r[0]) for r in ROWS)
    col_w = max(16, max(len(m) for m in models) + 2)
    print(f"{'metric':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in models))
    print("-" * (label_w + col_w * len(models)))
    for label, key in ROWS:
        print(f"{label:<{label_w}}" + "".join(f"{_cell(runs[m].get(key)):>{col_w}}"
                                              for m in models))


if __name__ == "__main__":
    main()
