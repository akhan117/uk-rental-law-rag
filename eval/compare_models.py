"""Side-by-side comparison of eval runs across models.

The gold set is the fixed benchmark and the model is the variable, so swapping
the base model should be a configuration change with a measured result rather
than a rebuild (Docs.pdf §6). Reads every eval_results_<model>.json in this
directory.

Usage: python compare_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_runs() -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(HERE.glob("eval_results_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            runs[data.get("model", path.stem)] = data["results"]
        else:
            # Older baseline files are a bare list; infer the model from the name.
            runs[path.stem.replace("eval_results_", "")] = data
    return runs


def metrics(results: list[dict]) -> dict[str, str]:
    def pct(subset, key):
        scored = [r for r in subset if r.get(key) is not None]
        if not scored:
            return "n/a"
        n = sum(1 for r in scored if r.get(key))
        return f"{n}/{len(scored)} ({100 * n / len(scored):.0f}%)"

    in_scope = [r for r in results if r["type"] == "in_scope"]
    orig = [r for r in in_scope if r["id"] <= "q23"]
    new = [r for r in in_scope if r["id"] >= "q25"]
    oos = [r for r in results if r["type"] == "out_of_scope"]
    must = [r for r in results if r["type"] == "must_answer"]
    stale = [r for r in results if r["type"] == "stale_law_check"]
    adv = [r for r in results if r["type"] == "adversarial" and r.get("tamper_applied")]

    return {
        "recall@6 (all in-scope)": pct(in_scope, "recall_hit"),
        "citation (all in-scope)": pct(in_scope, "citation_correct"),
        "answer correctness": pct(in_scope, "answer_correct"),
        "recall@6 (original 23)": pct(orig, "recall_hit"),
        "citation (original 23)": pct(orig, "citation_correct"),
        "recall@6 (new 7)": pct(new, "recall_hit"),
        "citation (new 7)": pct(new, "citation_correct"),
        "out-of-scope abstention": pct(oos, "correctly_abstained"),
        "answerable not over-gated": pct(must, "correctly_answered"),
        "stale-law handled": pct(stale, "correctly_handled_stale_law"),
        "adversarial grounding": pct(adv, "followed_tampered_text"),
    }


def main() -> None:
    runs = load_runs()
    if not runs:
        print("No eval_results_<model>.json files found. Run run_eval.py first.")
        return

    models = list(runs)
    rows = {m: metrics(r) for m, r in runs.items()}
    label_w = max(len(k) for k in next(iter(rows.values())))
    col_w = max(16, max(len(m) for m in models) + 2)

    print(f"{'metric':<{label_w}}" + "".join(f"{m:>{col_w}}" for m in models))
    print("-" * (label_w + col_w * len(models)))
    for metric in next(iter(rows.values())):
        print(f"{metric:<{label_w}}" + "".join(f"{rows[m][metric]:>{col_w}}" for m in models))


if __name__ == "__main__":
    main()
