"""Scope gate: decide whether to answer at all before generation runs.

Two independent checks, either of which can trigger abstention:
1. Keyword check for questions this demo's corpus structurally cannot answer -
   other jurisdictions (Scotland/Wales/NI), commercial lets, and questions
   premised on the law varying by location *within* England.
2. Retrieval-confidence check: if the corpus doesn't contain a cluster of
   closely-matching chunks, it probably doesn't answer the question, so we
   abstain rather than let the model improvise from weak evidence.
"""
from __future__ import annotations

import re

OUT_OF_SCOPE_PATTERNS = [
    (r"\bscotland\b|\bscottish\b", "Scotland has separate housing law (not covered by this England-only demo)."),
    (r"\bwales\b|\bwelsh\b", "Wales has separate housing law, including the Renting Homes (Wales) Act 2016 (not covered by this England-only demo)."),
    (r"\bnorthern ireland\b", "Northern Ireland has separate housing law (not covered by this England-only demo)."),
    (r"\bcommercial (lease|tenancy|property|premises)\b|\bbusiness tenancy\b", "Commercial/business tenancies are governed by different law (this demo covers residential tenancies only)."),
]

# Questions premised on residential tenancy law differing between places inside
# England. Retrieval can't catch these: the question is squarely on-topic, so it
# pulls back six perfectly relevant tenancy chunks that simply don't address the
# premise - and the model then narrates one of them back as if it were an answer
# (observed: "main leasing law differences between London and Manchester"
# returned a restatement of Tenant Fees Act 2019 s.1(10)). The honest response
# is to correct the premise, which is a statement about how the corpus and the
# statute book are territorially organised rather than a retrieved legal claim.
_ENGLISH_PLACES = (
    r"london|manchester|birmingham|leeds|liverpool|bristol|sheffield|newcastle|"
    r"nottingham|brighton|leicester|coventry|bradford|hull|southampton|portsmouth|"
    r"oxford|cambridge|york|norwich|plymouth|cornwall|yorkshire|"
    r"where (i|you|they) live|which (city|town|area|council)|my (city|town|area|council)|"
    r"postcode|part of the country|different (cities|towns|areas|regions|councils)"
)
_COMPARISON_CUES = (
    r"differ|different|difference|vary|varies|variation|compare|compared|"
    r"versus|\bvs\b|same in|same across|depend on where|depends on where"
)

GEOGRAPHY_NOTE = (
    "Residential tenancy law is the same everywhere in England - the Housing Act 1988, "
    "Protection from Eviction Act 1977, Housing Act 2004, Tenant Fees Act 2019 and "
    "Renters' Rights Act 2025 all apply nationally, so the rules in London and in "
    "Manchester are the same rules. What does vary locally is council-level regulation "
    "such as selective and additional HMO licensing schemes, which individual local "
    "authorities designate for their own areas - that material is not in this demo's "
    "corpus. Ask the underlying question without the location and this system can "
    "answer it from the legislation."
)

# A question is about intra-England geography only if it names a place (or asks
# about location generally) AND frames it as a comparison. "My landlord in London
# hasn't protected my deposit" names a place but asks an ordinary question, and
# must still be answered normally.
_GEOGRAPHY_RE = re.compile(_ENGLISH_PLACES)
_COMPARISON_RE = re.compile(_COMPARISON_CUES)

# This gate deliberately only catches questions with *no foothold at all* in the
# corpus, and is not asked to judge topical coverage.
#
# Measured, after query rewriting was added: answerable questions return 3-6
# chunks within 0.30 cosine distance (best 0.134-0.217), and clearly off-corpus
# ones ("what is the best mortgage rate right now?", "can I get planning
# permission for a loft conversion?") return 0-1. But questions that are merely
# *adjacent* to the corpus overlap the answerable range completely: "what are the
# rules on dog ownership in flats?" scores 3 chunks at 0.236 - better than "my
# deposit wasn't protected and there's black mould" at 3 chunks and 0.195 - because
# generic tenancy-agreement provisions really are the nearest neighbours to a
# question about tenancy terms. No threshold separates those two.
#
# So the split is by layer, not by number: this gate rejects questions with no
# evidence, and the generator rejects questions whose evidence doesn't address
# them (it has the excerpts and can say what's missing, which a distance cannot).
# Pushing the threshold up to catch the adjacent cases is what previously made the
# system refuse "landlord wont fix boiler been 3 weeks wot can i do".
CLOSE_DISTANCE = 0.30
MIN_CLOSE_HITS = 2


GENERIC_ABSTENTION = (
    "This question is outside the scope of this demo "
    "(England residential landlord-tenant law only)."
)


def scope_short_answer(reason: str) -> str:
    """Headline to show above a keyword-gate abstention. The intra-England
    geography case isn't really 'out of scope' - the question has a correct
    answer and the note *is* that answer - so it gets its own headline."""
    if reason == GEOGRAPHY_NOTE:
        return ("Residential tenancy law does not differ between places within England "
                "- the same Acts apply across the whole country.")
    return GENERIC_ABSTENTION


def keyword_scope_check(question: str) -> str | None:
    lowered = question.lower()
    for pattern, reason in OUT_OF_SCOPE_PATTERNS:
        if re.search(pattern, lowered):
            return reason
    if _GEOGRAPHY_RE.search(lowered) and _COMPARISON_RE.search(lowered):
        return GEOGRAPHY_NOTE
    return None


def retrieval_confidence_check(hits: list[dict]) -> str | None:
    """Abstain unless the corpus returned a *cluster* of close matches.

    Counting close hits rather than thresholding the single best one is what
    makes this robust: an irrelevant question still has a nearest neighbour, but
    it doesn't have several. Lexical-only hits carry a sentinel distance of 1.0
    (no dense score exists for them) and so never count toward the cluster.
    """
    close = sum(1 for h in hits if h["distance"] <= CLOSE_DISTANCE)
    if close < MIN_CLOSE_HITS:
        return ("No sufficiently relevant England residential landlord-tenant provision "
                "was found in the corpus for this question.")
    return None
