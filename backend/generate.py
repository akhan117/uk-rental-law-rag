from __future__ import annotations

import llm

SYSTEM_PROMPT = """You are a legal information assistant for England residential landlord-tenant law. \
You are NOT a solicitor and this is NOT legal advice.

You will be given a user question and a set of retrieved excerpts from legislation and case law, each \
labelled with its citation. Answer ONLY using the retrieved excerpts below - do not use any legal \
knowledge from your own training. If the excerpts do not fully support an answer, say so explicitly \
in "uncertainties" rather than guessing. Never invent a statute, section number, or case name that is \
not present in the excerpts. Every substantive claim in "applicable_law" and "how_it_applies" must be \
traceable to one of the given citations.

Respond with ONLY a JSON object with exactly these keys:
{
  "short_answer": "one or two sentence direct answer",
  "applicable_law": "the governing provisions, with citations, drawn only from the excerpts",
  "how_it_applies": "how the rule generally applies to situations like this one",
  "uncertainties": "what's uncertain, or what information would be needed to answer more precisely; empty string if none"
}

Answer the question the user actually asked - never restate it as a different question, and never
answer a different question that the excerpts happen to address instead.

A case excerpt is evidence of a legal principle; it is not the subject of the answer. State the rule
it establishes and apply it to the user's situation. Never make the parties, dates or facts of a
judgment the subject of "short_answer" - the user is not asking what happened to the litigants.

If the question takes something for granted that the excerpts contradict, correct it before
answering the rest.

Default to answering. If the excerpts support even a partial answer, give it and put the
remaining gaps in "uncertainties". Only when nothing in the excerpts bears on the question at all
should you say so - and then put exactly one short sentence in "short_answer" ("The retrieved
sources don't cover this."), leave "applicable_law" and "how_it_applies" empty, and explain what
is missing in "uncertainties". Never write an essay about what the excerpts do or don't contain."""


def build_user_prompt(question: str, facts: dict | None, hits: list[dict],
                      legal_question: str | None = None,
                      repeal_note: str | None = None) -> str:
    parts = [f"QUESTION: {question}"]
    if repeal_note:
        # First, so it frames everything after it: the model is about to be shown
        # excerpts that say nothing about a provision the user believes exists.
        parts.append(repeal_note)
    if legal_question and legal_question.strip().lower() != question.strip().lower():
        # The user's own words stay above as the thing to answer; this is the
        # question stripped of narrative, which is what retrieval actually ran on.
        parts.append(f"THE LEGAL QUESTION BEHIND IT: {legal_question}")
    if facts:
        fact_lines = [f"- {k}: {v}" for k, v in facts.items() if v]
        if fact_lines:
            parts.append("KNOWN FACTS:\n" + "\n".join(fact_lines))

    excerpts = []
    citations = []
    for h in hits:
        citation = h.get("citation") or h.get("case_name", "")
        citations.append(citation)
        # A Schedule 2 ground's Part decides whether the court *must* or *may*
        # order possession - the difference between a tenant having a defence and
        # not. It isn't in the ground's own wording, so without this the model has
        # to guess, and it did.
        label = citation
        if h.get("part_title"):
            label = f"{citation} - {h['part_title']}"
        excerpts.append(f"[{label}]\n{_trim_excerpt(h['text'])}")
    parts.append("RETRIEVED EXCERPTS:\n\n" + "\n\n---\n\n".join(excerpts))

    allowed = sorted(set(citations))
    parts.append(
        "The ONLY citations you may use anywhere in your answer are exactly these "
        "(copy verbatim, do not alter section/paragraph numbers): " + "; ".join(allowed) +
        ". If the excerpts above don't actually state the fact needed to answer, say so "
        "in uncertainties rather than citing something outside this list or relying on "
        "outside knowledge."
    )
    return "\n\n".join(parts)


# Chunk sizes are very skewed: the median retrieved chunk is ~570 characters but
# the largest seen is 8,315 (a whole schedule), and 23 of 72 chunks in a sample
# held 82% of all the prompt text. Unbounded, a six-chunk prompt reached 26,000
# characters - roughly 6,500 tokens against a 4,096-token context window, so the
# runtime was silently dropping excerpts the model was being told to rely on.
# Capping bounds the prompt, keeps it inside the window, and on a CPU-only
# machine cuts prompt-processing time, which is the bulk of the wait there.
# The heading and opening words carry the citation and the operative rule, so the
# head of a chunk is the part worth keeping.
EXCERPT_CHAR_LIMIT = 1000


def _trim_excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHAR_LIMIT:
        return text
    cut = text[:EXCERPT_CHAR_LIMIT]
    # Prefer a sentence or line break so the excerpt doesn't stop mid-clause.
    boundary = max(cut.rfind(". "), cut.rfind("\n"))
    if boundary > EXCERPT_CHAR_LIMIT // 2:
        cut = cut[:boundary + 1]
    return cut.rstrip() + "\n[... excerpt truncated ...]"


ANSWER_FIELDS = ["short_answer", "applicable_law", "how_it_applies", "uncertainties"]


def answer_schema(allowed_citations: list[str]) -> dict:
    """Answer schema with the citation list constrained to what was retrieved.

    Docs.pdf §5 proposes fine-tuning to improve citation discipline. A JSON-schema
    enum gets part of the way there for free: constrained decoding cannot emit a
    token sequence outside the allowed set, so naming an authority that was never
    retrieved becomes structurally impossible rather than merely discouraged.
    Verified against a prompt explicitly instructing the model to cite a
    non-retrieved Act - the citations list stayed clean.

    It is a partial fix, and worth being precise about the limit: only this field
    is constrained. Free text can still mention an invented authority, which is
    why backend/verify.py still checks the prose.
    """
    props: dict = {f: {"type": "string"} for f in ANSWER_FIELDS}
    required = list(ANSWER_FIELDS)
    allowed = sorted({c for c in allowed_citations if c})
    if allowed:
        props["citations"] = {"type": "array", "items": {"type": "string", "enum": allowed}}
        required.append("citations")
    return {"type": "object", "properties": props, "required": required}

# Constrain decoding to the exact answer shape. Plain {"format": "json"} only
# guarantees *valid* JSON, and the model was observed inventing its own keys
# ("case_name", "paragraphs") and echoing source text back, which surfaced to
# the user as a blank answer card.
ANSWER_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in ANSWER_FIELDS},
    "required": ANSWER_FIELDS,
}


def generate_answer(question: str, facts: dict | None, hits: list[dict],
                    legal_question: str | None = None,
                    repeal_note: str | None = None, trace=None) -> dict:
    user_prompt = build_user_prompt(question, facts, hits, legal_question, repeal_note)
    allowed = [h.get("citation") or h.get("case_name", "") for h in hits]
    # max_tokens bounds a degenerate loop; repeat_penalty (Ollama only) makes one
    # less likely in the first place. Observed on the repealed-s.21 question,
    # where the model has little it is allowed to say and restated "the retrieved
    # excerpts do not cover..." until the request timed out after ten minutes.
    # Unbounded generation turns a weak answer into a hung app.
    #
    # The timeout (llm.py, RAG_TIMEOUT) is generous for Ollama because the floor
    # is set by the slowest plausible machine: on a laptop with no GPU a single
    # answer is most of a minute. A timeout that trips reads to the user as "the
    # app is broken" - far worse than waiting.
    result_llm = llm.complete_json(
        SYSTEM_PROMPT, user_prompt, answer_schema(allowed), name="legal_answer",
        temperature=0.1, max_tokens=900, repeat_penalty=1.15,
        trace=trace, stage="generation")
    parsed = result_llm.parsed or {}

    result = {k: _as_text(v) for k, v in parsed.items() if k in ANSWER_FIELDS}
    # The schema enum constrains this to retrieved citations on providers that
    # enforce json_schema. Not all do (llm.py degrades to plain JSON mode when a
    # provider rejects the schema), so enforce the same guarantee in code: an
    # authority that wasn't retrieved can't reach the citations list either way.
    allowed_set = {c for c in allowed if c}
    raw_cited = parsed.get("citations", [])
    cited = [c for c in (raw_cited if isinstance(raw_cited, list) else [])
             if isinstance(c, str) and c in allowed_set]
    if not result.get("short_answer"):
        # Schema-constrained decoding should prevent this; if it still happens,
        # say so rather than rendering an empty answer card.
        return {
            "short_answer": "The model did not return a usable answer for this question.",
            "applicable_law": "",
            "how_it_applies": "",
            "uncertainties": "Generation error - the model's output did not match the "
                             "required answer format. The retrieved sources are still shown below.",
            "citations": [],
        }
    return {**{f: result.get(f, "") for f in ANSWER_FIELDS}, "citations": cited}


def _as_text(value) -> str:
    """The model sometimes returns a list of strings instead of one string
    for a free-text field (observed on applicable_law); normalize either
    shape down to plain text rather than erroring the request."""
    if isinstance(value, list):
        return "\n".join(_as_text(v) for v in value)
    if isinstance(value, dict):
        return "\n".join(f"{k}: {_as_text(v)}" for k, v in value.items())
    return "" if value is None else str(value)
