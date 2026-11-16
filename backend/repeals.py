"""Detect when a question asks about a provision that is no longer in force.

Docs.pdf §3 asks that amendment/version metadata be retained. This is the part
of that which changes answers rather than merely recording history.

The failure it fixes: asked "what notice period does a section 21 no-fault
eviction need?", the system answered "two months" - flatly, with no hedge and no
citation warning. Section 21 was repealed by the Renters' Rights Act 2025. The
retrieved excerpts contained nothing about s.21 (its text is gone), so nothing
contradicted the model and it answered from training memory. That is exactly the
hallucination this project exists to prevent, on the single most recognisable
provision in English rental law.

Retrieval cannot fix it: there is no text left to retrieve. The signal has to
come from the absence itself. legislation.gov.uk's *current revised* text keeps
a repealed section's heading and drops its body, so a provision that parses to a
heading with no operative text is one that has been repealed or emptied - a
direct statement from the source, not an inference from amendment wording.
(Parsing the amending Act instead does not work here: the Renters' Rights Act
repeals s.21 via "omit Chapter 2 of Part 1", which never names section 21.)

The manifest is built by ingestion/build_index.py, which already identifies these
chunks in order to keep them out of the index.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

MANIFEST_PATH = (Path(os.environ.get("RAG_DATA_DIR") or Path(__file__).resolve().parent.parent / "data")
                 / "processed" / "repealed.json")

# "section 21", "s.21", "s 21A", "Ground 3" - the forms a user actually types.
_SECTION_RE = re.compile(r"\b(?:s\.?|sections?)\s*(\d+[A-Z]*)\b", re.IGNORECASE)
_GROUND_RE = re.compile(r"\bground\s*(\d+[A-Z]*)\b", re.IGNORECASE)


def load_manifest(path: Path = MANIFEST_PATH) -> list[dict]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


_MANIFEST = load_manifest()


def _refs(text: str) -> set[str]:
    refs = {f"s{m.group(1).upper()}" for m in _SECTION_RE.finditer(text)}
    refs |= {f"ground{m.group(1).upper()}" for m in _GROUND_RE.finditer(text)}
    return refs


def repealed_in(question: str, manifest: list[dict] | None = None) -> list[dict]:
    """Entries from the manifest that the question refers to.

    Matching is on the bare provision number, deliberately. A user asking about
    "section 21" rarely names the Act, and the cost of being slightly broad is a
    caveat the answer did not strictly need - against the cost of being narrow,
    which is stating repealed law as current.
    """
    entries = _MANIFEST if manifest is None else manifest
    asked = _refs(question)
    return [e for e in entries if e["ref"] in asked]


def prompt_note(entries: list[dict]) -> str:
    """Instruction injected into the generation prompt."""
    lines = [
        f"- {e['citation']} ({e['heading']}) is NO LONGER IN FORCE: it was repealed or "
        f"removed, and the current revised text retains only its heading."
        for e in entries
    ]
    return (
        "IMPORTANT - the question refers to provisions that have been repealed:\n"
        + "\n".join(lines)
        + "\nSay so plainly before anything else, and do not state what the repealed "
        "provision used to require. If the excerpts show what replaced it, explain that "
        "instead; if they do not, say what is missing in \"uncertainties\"."
    )


def user_note(entries: list[dict]) -> str:
    """Plain-language note shown in the answer, independent of what the model says."""
    which = "; ".join(f"{e['citation']} ({e['heading']})" for e in entries)
    return (
        f"Note: {which} is no longer in force in the current text of the legislation. "
        "This corpus was built from the current revised text, in which the provision's "
        "wording has been removed."
    )
