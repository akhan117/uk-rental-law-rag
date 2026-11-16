"""Post-generation citation verification.

The prompt tells the model to cite only the retrieved excerpts, but a 4B model
doesn't always comply - the observed failure (see README Finding 1) was it
citing "Housing Act 1988, s.21(1) and (2)" when no s.21 text was retrieved at
all. Prompting alone can't guarantee this, so every statutory citation the
answer makes is checked against what was actually retrieved, and unsupported
ones are reported rather than passed off as grounded.

Citations are checked *Act-qualified*. An earlier version compared bare section
numbers, which let a whole class of error through: asked about section 21, the
model answered "...as per section 215 of the Housing Act 1988". Housing Act 2004
s.215 had been retrieved, so the bare number "s215" looked supported and no
warning fired - even though Housing Act 1988 s.215 does not exist. Attributing a
real provision to the wrong Act is a serious citation error, so the Act name and
the section number are now verified together.
"""
from __future__ import annotations

import re

# "Housing Act 1988, s.13(2)" / "s.213(3)" / "section 11" / "Sch.2 Ground 8"
SECTION_PATTERN = re.compile(r"\b(?:s\.?|section)\s*(\d+[A-Z]*)", re.IGNORECASE)
GROUND_PATTERN = re.compile(r"\bground\s*(\d+[A-Z]*)", re.IGNORECASE)
# "Housing Act 1988", "Landlord and Tenant Act 1985", "Homes (Fitness for Human
# Habitation) Act 2018", "Renters' Rights Act 2025"
ACT_PATTERN = re.compile(r"\b((?:[A-Z][\w'()]*\s+(?:of\s+|and\s+|for\s+)?){1,6}Act\s+\d{4})")

# How far back from a section reference to look for the Act it belongs to. Long
# enough to span "section 215 of the Housing Act 1988", short enough that an Act
# named in an unrelated earlier sentence doesn't get attached.
ACT_LOOKBACK_CHARS = 60


# The Act regex has to start at a capital, so a capitalised word running into the
# title gets swallowed: "Under Landlord and Tenant Act 1985" captured "Under" and
# no longer matched the retrieved Act. Strip these leading connectives.
_LEADING_NOISE = re.compile(
    r"^(?:under|the|of|in|per|as|see|by|to|and|a|an|from|with|for|is|was|this|that)\s+",
    re.IGNORECASE)


def _normalize_text(text: str) -> str:
    """Fold typographic apostrophes to ASCII. The model writes "Renters’ Rights
    Act 2025" while the corpus citation uses "Renters' Rights Act 2025"; without
    this the Act name fails to match its own source and a correct citation is
    reported as unsupported."""
    return text.replace("’", "'").replace("‘", "'")


def _normalize_act(name: str) -> str:
    name = re.sub(r"\s+", " ", name).strip().lower()
    while True:
        stripped = _LEADING_NOISE.sub("", name)
        if stripped == name:
            return name
        name = stripped


def _canonical_act(name: str, known: set[str]) -> str:
    """Resolve an Act mention against the Acts actually retrieved.

    Matching on the title's tail means a stray leading word can't turn a correct
    citation into a false warning, while a genuinely different Act ("Housing Act
    1988" vs "Housing Act 2004") still fails to resolve and gets flagged.
    """
    for k in known:
        if name == k or name.endswith(" " + k):
            return k
    return name


def _act_positions(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), _normalize_act(m.group(1))) for m in ACT_PATTERN.finditer(text)]


# "section 215 of the Housing Act 2004" - the Act directly after the number owns
# it, and must win over any Act named earlier in the sentence. Without this, a
# chunk headed "Housing Act 1988, s.8(5B)" whose body cross-refers to "section 215
# of the Housing Act 2004" registered support for *Housing Act 1988* s.215, which
# does not exist - and then silently vouched for the model making that same error.
_GENITIVE = re.compile(r"^[\s,]*(?:of\s+)?(?:the\s+)?$")


def _nearest_act(acts: list[tuple[int, int, str]], pos: int, end_pos: int,
                 text: str, default: str | None) -> str | None:
    """The Act this reference belongs to."""
    for start, _end, name in acts:
        if start >= end_pos and _GENITIVE.match(text[end_pos:start]):
            return name

    best = None
    for start, end, name in acts:
        if start <= pos and pos - end <= ACT_LOOKBACK_CHARS:
            best = name
        elif pos < start <= pos + ACT_LOOKBACK_CHARS and best is None:
            best = name
            break
    return best or default


def _cited_refs(text: str, default_act: str | None = None,
                known_acts: set[str] | None = None) -> set[str]:
    """Refs as "act|ref" when the Act is identifiable, else bare "ref"."""
    text = _normalize_text(text)
    acts = _act_positions(text)
    if known_acts:
        acts = [(s, e, _canonical_act(n, known_acts)) for s, e, n in acts]
    refs: set[str] = set()
    for pattern, prefix in ((SECTION_PATTERN, "s"), (GROUND_PATTERN, "ground")):
        for m in pattern.finditer(text):
            ref = f"{prefix}{m.group(1).upper()}"
            act = _nearest_act(acts, m.start(), m.end(), text, default_act)
            refs.add(f"{act}|{ref}" if act else ref)
    return refs


def _bare(ref: str) -> str:
    return ref.split("|", 1)[1] if "|" in ref else ref


def _bare_act(ref: str) -> str:
    return ref.split("|", 1)[0]


def _supported_refs(hits: list[dict]) -> set[str]:
    """Refs the retrieved evidence can actually support: anything appearing in a
    chunk's own citation line, plus anything the chunk text itself quotes.

    A chunk's text refers to its own Act implicitly ("subsection (3)", "section
    5"), so unqualified references inside a chunk are attributed to the Act named
    in that chunk's citation.
    """
    supported: set[str] = set()
    for h in hits:
        citation = h.get("citation") or ""
        own_acts = _act_positions(citation)
        own_act = own_acts[0][2] if own_acts else None
        supported |= _cited_refs(citation, default_act=own_act)
        supported |= _cited_refs(h.get("text", ""), default_act=own_act)
    return supported


def unsupported_citations(answer_fields: list[str], hits: list[dict]) -> list[str]:
    """Return citation refs the answer makes that the retrieved evidence doesn't back.

    An Act-qualified claim must be supported for that exact Act. A claim with no
    Act anywhere near it ("under section 11...") can't be attributed, so it is
    accepted if the number is supported under any Act - unattributable isn't the
    same as wrong, and warning on it would be noise.
    """
    supported = _supported_refs(hits)
    known_acts = {_bare_act(r) for r in supported if "|" in r}
    # Each field is parsed on its own: concatenating them put the end of
    # "short_answer" within lookback range of an Act named at the start of
    # "applicable_law", so a bare "section 21" in one field was attributed to an
    # unrelated Act in the next and wrongly reported.
    claimed: set[str] = set()
    for field in answer_fields:
        claimed |= _cited_refs(field, known_acts=known_acts)
    supported_bare = {_bare(r) for r in supported}

    unsupported = []
    for ref in claimed:
        if ref in supported:
            continue
        if "|" not in ref and ref in supported_bare:
            continue
        unsupported.append(ref)
    return sorted(unsupported)


def format_warning(unsupported: list[str]) -> str:
    pretty = ", ".join(_pretty(r) for r in unsupported)
    return (
        f"Automated check: this answer referred to {pretty}, which does not appear in "
        "the retrieved source material. Treat those references as unverified."
    )


def _pretty(ref: str) -> str:
    act, _, bare = ref.rpartition("|")
    label = (bare.replace("ground", "Ground ") if bare.startswith("ground")
             else bare.replace("s", "s.", 1))
    return f"{act.title()} {label}" if act else label
