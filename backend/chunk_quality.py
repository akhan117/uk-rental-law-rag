"""Exclude chunks that carry a heading but no operative text.

Chunks are stored as "HEADING\n\nBODY". A handful end up with an empty body:
provisions repealed by the Renters' Rights Act 2025 (Housing Act 1988 ss.19A,
20, 21), cross-headings whose grounds were removed, and one-line judicial
concurrences ("[47] I agree.").

These are actively harmful in retrieval, not merely useless. A bare heading like
"Recovery of possession on expiry or termination of assured shorthold tenancy"
is a pure topical label with no qualifying content, so it embeds close to *any*
question on that topic and wins a top slot - the repealed s.21 stub was coming
back as the single best match (0.182) for a question about deposit deductions -
and then contributes nothing to the answer, because there is no text to cite.

Dropping them from the index does not hide the repeals: Finding 1 in the README
rests on the repeal, and `data/processed/*.jsonl` still records these chunks.
This only keeps them out of the retrievable set.
"""
from __future__ import annotations

MIN_OPERATIVE_CHARS = 30


def operative_text(text: str) -> str:
    """The chunk's body - everything after the heading line."""
    parts = text.split("\n\n", 1)
    return parts[1].strip() if len(parts) == 2 else ""


def has_operative_text(text: str) -> bool:
    return len(operative_text(text)) >= MIN_OPERATIVE_CHARS
