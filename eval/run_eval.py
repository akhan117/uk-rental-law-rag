"""Score the RAG system against eval/gold_questions.jsonl:
- in_scope / stale_law_check: retrieval recall@k + whether the generated
  answer actually cites the expected provision (citation correctness).
- out_of_scope: did the scope gate correctly abstain.
- adversarial: tamper a specific retrieved chunk's numbers and check whether
  generation follows the retrieved (tampered) text rather than real-world
  memory - the grounding-override check that's the centerpiece of this demo.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from generate import MODEL  # noqa: E402
from main import answer_for, retrieve as main_retrieve  # noqa: E402
from retrieval import Retriever  # noqa: E402
from scope import keyword_scope_check, retrieval_confidence_check  # noqa: E402
from verify import unsupported_citations  # noqa: E402


def retrieve(question: str, retriever: Retriever) -> tuple[list[dict], str]:
    """Delegate to backend/main.py's own retrieval rather than re-implementing it.

    An earlier copy here drifted from the real thing the moment the pipeline
    changed, which meant the eval was scoring a system that no longer shipped.
    `retriever` is accepted for signature stability; main owns the instance.
    """
    return main_retrieve(question)

GOLD_PATH = Path(__file__).resolve().parent / "gold_questions.jsonl"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
K = 6


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


def run_in_scope(item: dict, retriever: Retriever) -> dict:
    hits, legal_question = retrieve(item["question"], retriever)
    retrieved_ids = {h["chunk_id"] for h in hits}
    expected = set(item.get("expected_chunk_ids", []))
    recall_hit = any(
        _matches_expected(e, r) for e in expected for r in retrieved_ids
    )

    conf_reason = retrieval_confidence_check(hits)
    if item["type"] == "stale_law_check":
        # Scored on whether the answer says the provision is no longer in force -
        # not on whether the system refused. Refusing was only ever a proxy for
        # "didn't state repealed law as current", and it is now the worse of the
        # two behaviours: telling a user that section 21 has been repealed is more
        # useful than declining to engage. Abstention still counts as a pass.
        if conf_reason:
            return {"id": item["id"], "type": item["type"], "recall_hit": recall_hit,
                    "correctly_handled_stale_law": True, "how": "abstained"}
        answer, _ = answer_for(item["question"], None, hits, legal_question)
        text = _all_text(answer).lower()
        said = any(p in text for p in ("repeal", "no longer in force", "abolish",
                                       "no longer exists", "has been removed"))
        return {"id": item["id"], "type": item["type"], "recall_hit": recall_hit,
                "correctly_handled_stale_law": said,
                "how": "explained repeal" if said else "answered as if in force"}

    if conf_reason:
        return {"id": item["id"], "type": item["type"], "recall_hit": recall_hit,
                "citation_correct": False, "note": "abstained (low retrieval confidence)"}

    answer, _ = answer_for(item["question"], None, hits, legal_question)
    combined = " ".join([answer.get("applicable_law", ""), answer.get("short_answer", "")])
    citation_correct = any(_citation_matches(cid, combined) for cid in expected)

    result = {"id": item["id"], "type": item["type"], "recall_hit": recall_hit,
              "citation_correct": citation_correct}

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


def run_out_of_scope(item: dict, retriever: Retriever) -> dict:
    """Run the *whole* gate, both checks, exactly as backend/main.py does.

    Scoring only the keyword check made the metric look perfect while testing
    half the system: it can only ever catch the jurisdictions someone wrote a
    regex for, and says nothing about questions that are simply absent from the
    corpus ("how do I appeal a parking ticket?"), which the confidence check is
    what actually handles.
    """
    if keyword_scope_check(item["question"]):
        return {"id": item["id"], "type": item["type"],
                "correctly_abstained": True, "gate": "keyword"}

    hits, legal_question = retrieve(item["question"], retriever)
    if retrieval_confidence_check(hits):
        return {"id": item["id"], "type": item["type"],
                "correctly_abstained": True, "gate": "confidence"}

    # Abstention is two-stage by design: the gate rejects questions with no
    # evidence, the generator rejects questions whose evidence doesn't address
    # them. Scoring only the gate would mark the second kind as a failure even
    # when the system behaved correctly - and would push the gate's threshold up
    # to compensate, which is exactly what made it refuse answerable questions.
    declined = _declined(answer_for(item["question"], None, hits, legal_question)[0])
    return {"id": item["id"], "type": item["type"],
            "correctly_abstained": declined, "gate": "generation" if declined else None}


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


def run_must_answer(item: dict, retriever: Retriever) -> dict:
    """The inverse guard: questions that must NOT be gated away.

    The scope gate is the easiest thing in the system to over-tighten - every
    abstention looks responsible in isolation - so these pin the other side.
    They mention a place or read like an edge case but are ordinary answerable
    questions.
    """
    if keyword_scope_check(item["question"]):
        return {"id": item["id"], "type": item["type"], "correctly_answered": False,
                "blocked_by": "keyword"}
    hits, _ = retrieve(item["question"], retriever)
    if retrieval_confidence_check(hits):
        return {"id": item["id"], "type": item["type"], "correctly_answered": False,
                "blocked_by": "confidence"}
    return {"id": item["id"], "type": item["type"], "correctly_answered": True,
            "blocked_by": None}


def run_adversarial(item: dict, retriever: Retriever) -> dict:
    hits, legal_question = retrieve(item["question"], retriever)
    # Tamper EVERY retrieved chunk containing the target text, not just the named
    # one. Other provisions and judgments often restate the same figure, so
    # tampering a single chunk leaves the real value visible elsewhere in context
    # and the probe stops measuring grounding - it just measures which of two
    # conflicting retrieved sources the model happened to prefer.
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

    answer, _ = answer_for(item["question"], None, hits, legal_question)
    combined = _all_text(answer).lower()
    followed = (item["expect_in_answer"].lower() in combined
                and item["expect_not_in_answer"].lower() not in combined)
    return {"id": item["id"], "type": item["type"], "tamper_applied": True,
            "chunks_tampered": len(tampered_chunks), "followed_tampered_text": followed}


def main() -> None:
    gold = load_gold()
    retriever = Retriever()
    results = []
    print(f"Model under test: {MODEL}\n")

    for item in gold:
        print(f"[{item['id']}] {item['type']}: {item['question'][:70]}")
        if item["type"] in ("in_scope", "stale_law_check"):
            results.append(run_in_scope(item, retriever))
        elif item["type"] == "out_of_scope":
            results.append(run_out_of_scope(item, retriever))
        elif item["type"] == "must_answer":
            results.append(run_must_answer(item, retriever))
        elif item["type"] == "adversarial":
            results.append(run_adversarial(item, retriever))

    in_scope = [r for r in results if r["type"] == "in_scope"]
    stale = [r for r in results if r["type"] == "stale_law_check"]
    oos = [r for r in results if r["type"] == "out_of_scope"]
    must = [r for r in results if r["type"] == "must_answer"]
    adv = [r for r in results if r["type"] == "adversarial"]

    def pct(n, d):
        return f"{n}/{d} ({100*n/d:.0f}%)" if d else "n/a"

    print("\n=== RESULTS ===")
    print(f"In-scope retrieval recall@{K}:   {pct(sum(r['recall_hit'] for r in in_scope), len(in_scope))}")
    print(f"In-scope citation correctness:  {pct(sum(r['citation_correct'] for r in in_scope), len(in_scope))}")
    scored = [r for r in in_scope if 'answer_correct' in r]
    if scored:
        print(f"In-scope answer correctness:    {pct(sum(r['answer_correct'] for r in scored), len(scored))}"
              f"  ({len(in_scope) - len(scored)} of {len(in_scope)} have no answer key)")
    halluc = [r for r in in_scope if 'hallucinated_authorities' in r]
    if halluc:
        bad = [r for r in halluc if r['hallucinated_authorities']]
        print(f"Answers with a hallucinated authority: {pct(len(bad), len(halluc))}"
              + (f"  ({', '.join(r['id'] for r in bad)})" if bad else ""))
    print(f"Out-of-scope correct abstention: {pct(sum(r['correctly_abstained'] for r in oos), len(oos))}")
    if must:
        blocked = [r["id"] for r in must if not r["correctly_answered"]]
        print(f"Answerable not over-gated:       {pct(sum(r['correctly_answered'] for r in must), len(must))}"
              + (f"  (blocked: {', '.join(blocked)})" if blocked else ""))
    if stale:
        print(f"Stale-law (s.21) handled correctly: {pct(sum(r['correctly_handled_stale_law'] for r in stale), len(stale))}"
              + f"  ({', '.join(r['how'] for r in stale)})")
    adv_scored = [r for r in adv if r["tamper_applied"]]
    print(f"Adversarial grounding-override:  {pct(sum(r['followed_tampered_text'] for r in adv_scored), len(adv_scored))}"
          f"  ({len(adv) - len(adv_scored)} skipped - tamper target not retrieved)")

    # Tag by model so runs of different models don't overwrite each other - the
    # gold set is the fixed benchmark, the model is the variable (Docs.pdf §6).
    slug = MODEL.replace(":", "-").replace("/", "-")
    out_path = Path(__file__).resolve().parent / f"eval_results_{slug}.json"
    payload = {"model": MODEL, "results": results}
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (Path(__file__).resolve().parent / "eval_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
