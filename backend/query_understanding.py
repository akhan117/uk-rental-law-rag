"""Turn a user's question into retrieval queries written in statutory vocabulary.

Measured problem this solves: real questions arrive as narrative prose ("I've
lived here 3 years, always paid on time, and now out of nowhere my landlord says
I have to go..."). Embedded raw, that text sits closer to *judicial narrative*
than to terse statutory drafting, so retrieval returned 5 of 6 case-law chunks
and the model then answered by summarising the case it was handed - reporting
what happened to Ms Rouncefield's gas safety record instead of answering the
user. Hand-rewriting the same questions into statutory wording moved best-match
distance from 0.29-0.49 to 0.18-0.24 and flipped the retrieved set to
legislation, which showed the corpus had the answers all along and the query was
the defect.

The rewrite is done by the same local model, constrained to a small JSON schema.
It only reformulates: it is given no legal text and must not answer anything.
Any failure falls back to the original question, so this can degrade but not
break retrieval.
"""
from __future__ import annotations

import json
import os
import re

import httpx

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("RAG_MODEL", "qwen3:4b")
# Same reasoning as generate.py: a timeout here silently drops the rewrite and
# sends the raw question to retrieval, which is the behaviour this module exists
# to avoid. On a slow machine that would fire exactly when it is needed most.
OLLAMA_TIMEOUT = float(os.environ.get("RAG_TIMEOUT", "600"))

SYSTEM_PROMPT = """You rewrite questions about England residential landlord-tenant law into search \
queries for a database of legislation and case law. You do NOT answer questions and you do NOT state \
any law.

Given the user's question, return JSON:
{
  "legal_question": "the user's question restated as one clear, neutral legal question",
  "search_queries": ["1 to 3 short search queries in the vocabulary legislation actually uses"]
}

Rules for "search_queries":
- Use statutory wording, not everyday wording: "notice to quit", "possession", "repairing covenant",
  "tenancy deposit", "prohibited payment", "assured tenancy", "grounds for possession".
- Strip all narrative, emotion, names, dates and background. Keep only the legal subject matter.
- No question marks, no "how do I", no "can my landlord" - write them as topic phrases.
- If the user asks about several distinct things, give one query per thing.
- If the question is already precise legal wording, keep it essentially as-is.

Rules for "legal_question":
- Preserve what the user actually wants to know. Do not narrow it to something easier.
- If the question assumes something ("since section 21 still exists...", "why does my landlord only
  need 24 hours..."), phrase it so the assumption is tested rather than accepted: e.g. "whether
  section 21 notices still exist and how they may be challenged".

Examples:
Q: "landlord wont fix boiler been 3 weeks wot can i do"
{"legal_question": "What can a tenant do when the landlord has not repaired the heating?",
 "search_queries": ["landlord repairing covenant heating and hot water installations",
                    "fitness for human habitation implied term"]}

Q: "My deposit wasn't protected and there's black mould in the bathroom. What can I do about both?"
{"legal_question": "What remedies does a tenant have for an unprotected deposit and for disrepair?",
 "search_queries": ["tenancy deposit not protected in authorised scheme remedy",
                    "landlord repairing covenant fitness for human habitation damp"]}

Q: "Why does my landlord only need to give me 24 hours notice to evict me?"
{"legal_question": "Whether a landlord can evict a tenant on 24 hours notice, and what notice period
                    the law actually requires",
 "search_queries": ["minimum notice period to end an assured tenancy",
                    "protection from eviction notice to quit minimum period",
                    "possession order required before eviction"]}

Q: "Since section 21 still exists, how do I fight a section 21 notice?"
{"legal_question": "Whether section 21 no-fault possession notices still exist, and how possession is
                    obtained now",
 "search_queries": ["abolition of assured shorthold tenancies section 21",
                    "grounds for possession notice requirements"]}"""

SCHEMA = {
    "type": "object",
    "properties": {
        "legal_question": {"type": "string"},
        "search_queries": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["legal_question", "search_queries"],
}

MAX_QUERIES = 3
# A rewrite that collapses to a couple of words has lost the question rather than
# clarified it; fall back to the user's own words in that case.
MIN_QUERY_CHARS = 12


def understand(question: str) -> dict:
    """Return {"legal_question": str, "search_queries": [str]}.

    Always returns something usable - on any failure the original question is
    used as its own search query, which is exactly the previous behaviour.
    """
    fallback = {"legal_question": question, "search_queries": [question], "rewritten": False}
    try:
        with httpx.Client(timeout=OLLAMA_TIMEOUT) as client:
            resp = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": MODEL,
                    "system": SYSTEM_PROMPT,
                    "prompt": f"Q: {question}",
                    "stream": False,
                    "think": False,
                    "format": SCHEMA,
                    "options": {"temperature": 0.0},
                },
            )
            resp.raise_for_status()
        parsed = json.loads(resp.json()["response"])
    except (httpx.HTTPError, json.JSONDecodeError, KeyError):
        return fallback

    queries = [
        _clean(q) for q in parsed.get("search_queries", [])
        if isinstance(q, str) and len(_clean(q)) >= MIN_QUERY_CHARS
    ]
    if not queries:
        return fallback

    legal_question = parsed.get("legal_question")
    if not isinstance(legal_question, str) or not legal_question.strip():
        legal_question = question

    # Keep the user's own wording as a final query: the rewrite can drop a term
    # that only exact lexical match would have found (BM25 relies on those), and
    # the rewriter has no view of what is actually in the corpus.
    queries = queries[:MAX_QUERIES]
    if question not in queries:
        queries.append(question)

    return {"legal_question": legal_question.strip(), "search_queries": queries, "rewritten": True}


def _clean(q: str) -> str:
    return re.sub(r"\s+", " ", q).strip().strip("?").strip()
