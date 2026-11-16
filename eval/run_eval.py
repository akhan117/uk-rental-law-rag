"""Score the RAG system against eval/gold_questions.jsonl:
- in_scope / stale_law_check: retrieval recall@k + whether the generated
  answer actually cites the expected provision (citation correctness).
- out_of_scope: did the scope gate correctly abstain.
- adversarial: tamper a specific retrieved chunk's numbers and check whether
  generation follows the retrieved (tampered) text rather than real-world
  memory - the grounding-override check that's the centerpiece of this demo.

Modes:
  python run_eval.py                       full eval, single-shot RAG
  python run_eval.py --mode agent          full eval, agent-planned retrieval
  python run_eval.py --retrieval-only      no LLM at all: recall + pre-generation
                                           gates. Deterministic; what CI runs on
                                           every push.
  ... --check eval/thresholds.json         exit 1 if a gated metric regresses

The model under test comes from the environment (LLM_PROVIDER / LLM_MODEL, see
backend/llm.py), so `LLM_PROVIDER=groq python run_eval.py` benchmarks the hosted
model against the same fixed gold set.

Questions whose expected evidence isn't in the deployed corpus (the case-law
questions, in a legislation-only build) are reported as skipped rather than
scored as failures: a missing document is a corpus decision, not a retrieval
regression.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
sys.path.insert(0, str(HERE))

import llm  # noqa: E402
import main  # noqa: E402
import scoring  # noqa: E402
from main import answer_for  # noqa: E402
from scope import keyword_scope_check, retrieval_confidence_check  # noqa: E402
from telemetry import Trace  # noqa: E402
from verify import unsupported_citations  # noqa: E402

GOLD_PATH = HERE / "gold_questions.jsonl"
PROCESSED_DIR = Path(os.environ.get("RAG_DATA_DIR") or HERE.parent / "data") / "processed"
K = 6

MODE = "rag"
RETRIEVAL_ONLY = False


def load_chunk_metadata() -> dict[str, dict]:
    meta = {}
    for name in ["legislation_chunks.jsonl", "case_chunks.jsonl"]:
        path = PROCESSED_DIR / name
        if not path.exists():
            continue
        for line in path.open(encoding="utf-8"):
            c = json.loads(line)
            meta[c["chunk_id"]] = c
    return meta


CHUNK_META = load_chunk_metadata()


def load_gold() -> list[dict]:
    items = []
    for line in GOLD_PATH.open(encoding="utf-8"):
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def retrieve(question: str, trace: Trace) -> tuple[list[dict], str]:
    """Delegate to backend/main.py's own retrieval rather than re-implementing it.

    An earlier copy here drifted from the real thing the moment the pipeline
    changed, which meant the eval was scoring a system that no longer shipped.
    In agent mode, the agent's gathered evidence *is* the retrieval.
    """
    if MODE == "agent" and not RETRIEVAL_ONLY:
        result = main.agent_mod.run_agent(question, main.get_retriever(), main.get_corpus(),
                                          trace=trace)
        trace.outcome["agent_result"] = result
        if result.evidence:
            return result.evidence, question
        trace.outcome["agent_fallback"] = True
    return main.retrieve(question, trace=trace, allow_rewrite=not RETRIEVAL_ONLY)


def _agent_extras(trace: Trace) -> tuple[list[dict], int]:
    result = trace.outcome.get("agent_result")
    return (result.repealed, result.exact_hits) if result else ([], 0)


def _confidence(hits: list[dict], trace: Trace) -> str | None:
    # Same rule as main.run_query: an exact lookup satisfies the gate on its own.
    _, exact = _agent_extras(trace)
    return None if exact else retrieval_confidence_check(hits)


def _answer(question: str, hits: list[dict], legal_question: str, trace: Trace) -> dict:
    repealed, _ = _agent_extras(trace)
    return answer_for(question, None, hits, legal_question, trace, repealed)[0]


def _all_text(answer: dict) -> str:
    """Flatten an answer for substring checks. "citations" is a list, so a plain
    join over .values() raises - and would do so only on the paths that read the
    whole answer, which are the adversarial and stale-law probes."""
    parts = []
    for v in answer.values():
        parts.extend(v if isinstance(v, list) else [str(v)])
    return " ".join(parts)


def _matches_expected(expected_id: str, retrieved_id: str) -> bool:
    """A gold label names a provision ("...:section-5"); long sections are stored
    split by subsection ("...:section-5-1ZA"), so retrieving any subsection of the
    expected provision counts. The "-" guard keeps section-5 from matching section-50.
    """
    return retrieved_id == expected_id or retrieved_id.startswith(expected_id + "-")


def _in_corpus(expected: set[str]) -> bool:
    return any(_matches_expected(e, cid) for e in expected for cid in CHUNK_META)


def run_in_scope(item: dict, trace: Trace) -> dict:
    expected = set(item.get("expected_chunk_ids", []))
    if expected and not _in_corpus(expected):
        return {"id": item["id"], "type": item["type"], "skipped": "expected evidence not in corpus"}

    hits, legal_question = retrieve(item["question"], trace)
    retrieved_ids = {h["chunk_id"] for h in hits}
    recall_hit = any(
        _matches_expected(e, r) for e in expected for r in retrieved_ids
    )
    base = {"id": item["id"], "type": item["type"], "recall_hit": recall_hit,
            "rewrite_needed": bool(trace.outcome.get("rewrite_needed"))}

    conf_reason = _confidence(hits, trace)
    if RETRIEVAL_ONLY:
        return base | {"abstained_pre_generation": bool(conf_reason)}

    if item["type"] == "stale_law_check":
        # Scored on whether the answer says the provision is no longer in force -
        # not on whether the system refused. Refusing was only ever a proxy for
        # "didn't state repealed law as current", and it is now the worse of the
        # two behaviours: telling a user that section 21 has been repealed is more
        # useful than declining to engage. Abstention still counts as a pass.
        if conf_reason:
            return base | {"correctly_handled_stale_law": True, "how": "abstained"}
        answer = _answer(item["question"], hits, legal_question, trace)
        text = _all_text(answer).lower()
        said = any(p in text for p in ("repeal", "no longer in force", "abolish",
                                       "no longer exists", "has been removed"))
        return base | {"correctly_handled_stale_law": said,
                       "how": "explained repeal" if said else "answered as if in force"}

    if conf_reason:
        return base | {"citation_correct": False, "note": "abstained (low retrieval confidence)"}

    answer = _answer(item["question"], hits, legal_question, trace)
    combined = " ".join([answer.get("applicable_law", ""), answer.get("short_answer", "")])
    citation_correct = any(_citation_matches(cid, combined) for cid in expected)

    result = base | {"citation_correct": citation_correct}

    # Docs.pdf §6 asks for answer correctness alongside citation correctness. They
    # are not the same thing and the difference matters: citing the right section
    # while stating the wrong rule scored as a pass under the old metrics.
    key = item.get("answer_must_include")
    if key:
        text = _all_text(answer).lower()
        result["answer_correct"] = all(
            any(variant.lower() in text for variant in group) for group in key
        )

    # Also §6: hallucinated authorities. verify.py already computes this per
    # request; counting it here turns a runtime warning into a measured rate.
    result["hallucinated_authorities"] = len(unsupported_citations(
        [answer.get("short_answer", ""), answer.get("applicable_law", ""),
         answer.get("how_it_applies", "")], hits))
    return result


def _citation_matches(expected_chunk_id: str, text: str) -> bool:
    """Does the text mention the expected provision/case?
    Legislation: Act name + section/paragraph number from the chunk id.
    Case law: the case name (paragraph ranges aren't meaningfully "cited").
    """
    text_low = text.lower()
    meta = CHUNK_META.get(expected_chunk_id)
    if meta is None:
        return False

    if meta["source_type"] == "case":
        # Either form identifies the case unambiguously: the party names
        # ("Edwards v Kumarasamy") or the neutral citation ("[2016] UKSC 40").
        # The model routinely uses the latter, which is proper legal practice.
        citation = (meta.get("citation") or "").strip().lower()
        if citation and citation in text_low:
            return True
        name_words = [w for w in meta["case_name"].replace(",", "").split() if len(w) > 2][:2]
        return all(w.lower() in text_low for w in name_words) if name_words else False

    slug, rest = expected_chunk_id.split(":", 1)
    act_words = [w for w in slug.replace("-", " ").split() if not w.isdigit()][:3]
    act_hit = all(w in text_low for w in act_words) if act_words else True
    num = "".join(c for c in rest.split("-")[-1] if c.isalnum())
    num_hit = (num.lower() in text_low) if num else True
    return act_hit and num_hit


def run_out_of_scope(item: dict, trace: Trace) -> dict:
    """Run the *whole* gate, both checks, exactly as backend/main.py does.

    Scoring only the keyword check made the metric look perfect while testing
    half the system: it can only ever catch the jurisdictions someone wrote a
    regex for, and says nothing about questions that are simply absent from the
    corpus ("how do I appeal a parking ticket?"), which the confidence check is
    what actually handles.
    """
    if keyword_scope_check(item["question"]):
        return {"id": item["id"], "type": item["type"], "correctly_abstained": True,
                "abstained_pre_generation": True, "gate": "keyword"}

    hits, legal_question = retrieve(item["question"], trace)
    if _confidence(hits, trace):
        return {"id": item["id"], "type": item["type"], "correctly_abstained": True,
                "abstained_pre_generation": True, "gate": "confidence"}
    if RETRIEVAL_ONLY:
        # Reaches generation, which decides; not scoreable without a model.
        return {"id": item["id"], "type": item["type"], "abstained_pre_generation": False}

    # Abstention is two-stage by design: the gate rejects questions with no
    # evidence, the generator rejects questions whose evidence doesn't address
    # them. Scoring only the gate would mark the second kind as a failure even
    # when the system behaved correctly - and would push the gate's threshold up
    # to compensate, which is exactly what made it refuse answerable questions.
    declined = _declined(_answer(item["question"], hits, legal_question, trace))
    return {"id": item["id"], "type": item["type"], "correctly_abstained": declined,
            "abstained_pre_generation": False, "gate": "generation" if declined else None}


_DECLINE_PHRASES = (
    "don't cover", "do not cover", "not covered", "doesn't cover",
    "don't address", "do not address", "doesn't address", "not addressed",
    "don't contain", "do not contain", "doesn't contain", "no information",
    "not provide", "don't provide", "doesn't provide",
)


def _declined(answer: dict) -> bool:
    """Did generation refuse rather than answer?

    The prompt tells the model to leave applicable_law and how_it_applies empty
    when nothing in the excerpts bears on the question, so an empty pair is the
    structural signal; the phrase list catches the cases where it explains itself
    in short_answer but fills the other fields with that same explanation.
    """
    if not answer.get("applicable_law", "").strip() and not answer.get("how_it_applies", "").strip():
        return True
    text = " ".join([answer.get("short_answer", ""), answer.get("applicable_law", "")]).lower()
    return any(p in text for p in _DECLINE_PHRASES)


def run_must_answer(item: dict, trace: Trace) -> dict:
    """The inverse guard: questions that must NOT be gated away.

    The scope gate is the easiest thing in the system to over-tighten - every
    abstention looks responsible in isolation - so these pin the other side.
    They mention a place or read like an edge case but are ordinary answerable
    questions.
    """
    if keyword_scope_check(item["question"]):
        return {"id": item["id"], "type": item["type"], "correctly_answered": False,
                "blocked_by": "keyword"}
    hits, _ = retrieve(item["question"], trace)
    if _confidence(hits, trace):
        if RETRIEVAL_ONLY and trace.outcome.get("rewrite_needed"):
            # Would be rescued (or not) by the LLM rewrite, which this mode skips.
            return {"id": item["id"], "type": item["type"], "correctly_answered": None,
                    "blocked_by": None, "rewrite_needed": True}
        return {"id": item["id"], "type": item["type"], "correctly_answered": False,
                "blocked_by": "confidence"}
    return {"id": item["id"], "type": item["type"], "correctly_answered": True,
            "blocked_by": None}


def run_adversarial(item: dict, trace: Trace) -> dict:
    if RETRIEVAL_ONLY:
        return {"id": item["id"], "type": item["type"], "skipped": "needs generation"}
    hits, legal_question = retrieve(item["question"], trace)
    # Tamper EVERY retrieved chunk containing the target text, not just the named
    # one. Other provisions and judgments often restate the same figure, so
    # tampering a single chunk leaves the real value visible elsewhere in context
    # and the probe stops measuring grounding - it just measures which of two
    # conflicting retrieved sources the model happened to prefer.
    hits = [dict(h) for h in hits]
    tampered_chunks = []
    target_seen = False
    for h in hits:
        if h["chunk_id"] == item["tamper_chunk_id"]:
            target_seen = True
        if item["tamper_find"] in h["text"]:
            h["text"] = h["text"].replace(item["tamper_find"], item["tamper_replace"])
            tampered_chunks.append(h["chunk_id"])
    if not target_seen or not tampered_chunks:
        return {"id": item["id"], "type": item["type"], "tamper_applied": False,
                "followed_tampered_text": None}

    answer = _answer(item["question"], hits, legal_question, trace)
    combined = _all_text(answer).lower()
    followed = (item["expect_in_answer"].lower() in combined
                and item["expect_not_in_answer"].lower() not in combined)
    return {"id": item["id"], "type": item["type"], "tamper_applied": True,
            "chunks_tampered": len(tampered_chunks), "followed_tampered_text": followed}


RUNNERS = {
    "in_scope": run_in_scope,
    "stale_law_check": run_in_scope,
    "out_of_scope": run_out_of_scope,
    "must_answer": run_must_answer,
    "adversarial": run_adversarial,
}


def run_item(item: dict) -> dict:
    trace = Trace(mode=MODE)
    t0 = time.perf_counter()
    try:
        result = RUNNERS[item["type"]](item, trace)
    except llm.LLMError as exc:
        # A free-tier quota can run out mid-run (tokens per day). Record it and
        # keep going rather than lose every result so far; errors are counted
        # and gated separately, so a quota-starved run can't pass as a good one.
        print(f"    LLM error: {exc}", flush=True)
        result = {"id": item["id"], "type": item["type"], "skipped": "llm_error",
                  "error": str(exc)[:300]}
    result.update(
        latency_s=round(time.perf_counter() - t0, 3),
        llm_calls=len(trace.llm_calls),
        tokens_in=trace.tokens_in,
        tokens_out=trace.tokens_out,
        tool_calls=len(trace.tool_calls),
    )
    return result


def _git_sha() -> str:
    """HEAD, marked "-dirty" when the run measured uncommitted code."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, cwd=HERE, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, cwd=HERE, timeout=5).stdout.strip()
        return f"{sha}-dirty" if sha and dirty else sha
    except Exception:  # noqa: BLE001
        return ""


def print_summary(s: dict, model_label: str) -> None:
    f = scoring.fmt
    print(f"\n=== RESULTS ({model_label}) ===")
    print(f"In-scope retrieval recall@{K}:   {f(s['recall'])}")
    if RETRIEVAL_ONLY:
        print(f"Rewrites needed (in-scope):      {s['rewrites_needed_in_scope']}")
        print(f"Out-of-scope stopped pre-generation: {f(s['oos_pre_generation'])}")
        print(f"Answerable not over-gated:       {f(s['must_answer'])}"
              "  (rewrite-dependent probes excluded)")
    else:
        print(f"In-scope citation correctness:  {f(s['citation'])}")
        print(f"In-scope answer correctness:    {f(s['answer'])}")
        print(f"Answers with a hallucinated authority: {f(s['hallucination'])}")
        print(f"Out-of-scope correct abstention: {f(s['oos_abstention'])}")
        print(f"Answerable not over-gated:       {f(s['must_answer'])}")
        print(f"Stale-law (s.21) handled correctly: {f(s['stale_law'])}")
        print(f"Adversarial grounding-override:  {f(s['adversarial'])}")
        print(f"Latency p50 / p95:               {s['latency_p50_s']}s / {s['latency_p95_s']}s")
        print(f"Tokens per answered question:    {s['tokens_in_avg']} in / "
              f"{s['tokens_out_avg']} out, {s['llm_calls_avg']} LLM calls"
              + (f", {s['tool_calls_avg']} tool calls" if MODE == "agent" else ""))
    if s["skipped"]:
        print(f"Skipped (evidence not in corpus / needs generation): {s['skipped']}")


def main_cli(argv: list[str] | None = None) -> int:
    global MODE, RETRIEVAL_ONLY
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--mode", choices=["rag", "agent"], default="rag")
    ap.add_argument("--retrieval-only", action="store_true",
                    help="no LLM calls; deterministic retrieval + gate metrics")
    ap.add_argument("--types", help="comma-separated item types to run")
    ap.add_argument("--ids", help="comma-separated item ids to run")
    ap.add_argument("--out", type=Path, help="results file (default: tagged by model and mode)")
    ap.add_argument("--check", type=Path, help="thresholds JSON; exit 1 on a regression")
    ap.add_argument("--baseline", type=Path,
                    help="earlier results file; flag questions whose recall regressed")
    ap.add_argument("--summary-md", type=Path, help="append a markdown summary (CI job summary)")
    args = ap.parse_args(argv)
    MODE, RETRIEVAL_ONLY = args.mode, args.retrieval_only

    gold = load_gold()
    if args.types:
        gold = [g for g in gold if g["type"] in set(args.types.split(","))]
    if args.ids:
        gold = [g for g in gold if g["id"] in set(args.ids.split(","))]

    cfg = llm.get_config()
    model_label = "retrieval-only" if RETRIEVAL_ONLY else f"{cfg.provider}/{cfg.model} [{MODE}]"
    print(f"Under test: {model_label}\n")

    results = []
    for item in gold:
        print(f"[{item['id']}] {item['type']}: {item['question'][:70]}", flush=True)
        results.append(run_item(item))

    summary = scoring.summarize(results)
    print_summary(summary, model_label)

    if args.out:
        out_path = args.out
    elif RETRIEVAL_ONLY:
        out_path = HERE / "eval_results_retrieval.json"
    else:
        # Tag by model and mode so runs don't overwrite each other - the gold set
        # is the fixed benchmark, the model is the variable (Docs.pdf §6).
        slug = cfg.model.replace(":", "-").replace("/", "-")
        out_path = HERE / f"eval_results_{slug}{'_agent' if MODE == 'agent' else ''}.json"
    payload = {
        "model": "retrieval-only" if RETRIEVAL_ONLY else cfg.model,
        "provider": cfg.provider, "mode": MODE, "retrieval_only": RETRIEVAL_ONLY,
        "git_sha": _git_sha(), "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "corpus_chunks": len(CHUNK_META), "summary": summary, "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")

    failures: list[str] = []
    if args.check:
        spec = json.loads(args.check.read_text(encoding="utf-8"))
        spec = spec.get("retrieval_only" if RETRIEVAL_ONLY else "full", spec)
        baseline = scoring.load(args.baseline)["results"] if args.baseline else None
        failures = scoring.check(summary, results, spec, baseline)
        print("\n=== CHECK ===")
        print("\n".join(f"FAIL  {f}" for f in failures) or "All gated metrics within bounds.")

    if args.summary_md:
        rows = [(k, scoring.fmt(v) if isinstance(v, dict) else v) for k, v in summary.items()]
        md = [f"### Eval: {model_label}", "", "| metric | value |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in rows if v not in (None, "n/a")]
        if args.check:
            md += ["", "**Regression check:** " + ("passed" if not failures else "FAILED"), ""]
            md += [f"- {f}" for f in failures]
        with args.summary_md.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(md) + "\n\n")

    return 1 if failures else 0


if __name__ == "__main__":
    # Line-buffered so CI logs show progress while a slow run is underway.
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main_cli())
