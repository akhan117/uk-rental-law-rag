"""Corpus tools for the agent (and the MCP server).

Three tools, each answering something single-shot retrieval structurally can't:

- ``search_law``     - the existing hybrid search, but the *model* chooses the
                       query wording and the filters, and can search again.
- ``get_provision``  - exact lookup by Act + section/Ground. Retrieval is
                       similarity-ranked, so "s.8(4AA)" can lose its slot to a
                       topically similar neighbour; a lookup cannot. It also
                       knows which provisions were repealed (the manifest from
                       build_index.py), which is the one fact README Finding 1
                       showed the model is least reliable about.
- ``list_sources``   - what the corpus actually holds, so the agent can say
                       "not covered" from knowledge rather than from a miss.

Every tool returns (text for the model, hits). The hits are what the final
answer is generated from, through the same citation-constrained, verified path
as normal RAG mode - the agent gathers evidence, it never writes the answer.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from repeals import load_manifest

TOOL_EXCERPT_CHARS = 600
LOOKUP_MAX_CHUNKS = 6
SEARCH_DEFAULT_K = 5
SEARCH_MAX_K = 8

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_law",
            "description": (
                "Hybrid semantic + keyword search over the England residential tenancy "
                "corpus (legislation and case law). Use statutory vocabulary ('notice to "
                "quit', 'possession', 'tenancy deposit', 'repairing covenant'), one topic "
                "per search. Returns excerpts with their citations."),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query, statutory wording."},
                    "source_type": {"type": "string", "enum": ["legislation", "case"],
                                    "description": "Optional: restrict to legislation or case law."},
                    "act": {"type": "string",
                            "description": "Optional: restrict to one Act, e.g. 'Housing Act 2004'."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_provision",
            "description": (
                "Fetch the exact text of a provision by Act and reference, e.g. "
                "act='Housing Act 1988', provision='s.8(4AA)' or 'Ground 8' or 'section 21'. "
                "Reports if the provision has been repealed. Use it when the question names "
                "a provision, or to follow a cross-reference found in another excerpt."),
            "parameters": {
                "type": "object",
                "properties": {
                    "act": {"type": "string", "description": "Act title, e.g. 'Housing Act 1988'."},
                    "provision": {"type": "string",
                                  "description": "e.g. 's.21', 'section 8(4AA)', 'Ground 10', 'Schedule 10'."},
                },
                "required": ["act", "provision"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sources",
            "description": "List the Acts and cases in the corpus. Use when unsure whether a topic is covered.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class ToolResult:
    content: str                    # what the model reads
    hits: list[dict] = field(default_factory=list)   # evidence for the answer
    summary: str = ""               # one line for the UI / logs
    exact: bool = False             # hits came from an exact lookup, not similarity
    repealed: list[dict] = field(default_factory=list)


def _trim(text: str, limit: int = TOOL_EXCERPT_CHARS) -> str:
    text = re.sub(r"\s+\n", "\n", text).strip()
    return text if len(text) <= limit else text[:limit].rstrip() + " [...]"


def _norm(s: str) -> str:
    s = s.lower().replace("&", " and ").replace("’", "'")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


# Words ignored when comparing an Act name by overlap ("housing act" ~ "Housing Act 1988").
_ACRONYM_STOP = {"of", "and", "for", "the", "from", "act", "order", "regulations"}
# Conventional abbreviations keep "Act": HA, LTA, PEA, TFA, RRA.
_INITIALS_SKIP = {"of", "and", "for", "the", "from"}


def _acronym(title: str) -> str:
    words = [w for w in _norm(title).split() if not w.isdigit() and w not in _INITIALS_SKIP]
    return "".join(w[0] for w in words)


def _provision_label(chunk: dict) -> str:
    """Short human label for the provision a chunk belongs to ("s.13", "Ground 8")."""
    tail = chunk["chunk_id"].split(":", 1)[-1]
    if m := re.search(r"ground-(\w+)$", tail):
        return f"Ground {m.group(1).upper()}"
    parent = chunk.get("parent_section") or tail
    if m := re.match(r"section-(\w+)", parent):
        return f"s.{m.group(1)}"
    if m := re.match(r"schedule-(\w+)", parent):
        return f"Sch.{m.group(1)}"
    return parent


class Corpus:
    """Index over the retriever's chunks for exact lookups."""

    def __init__(self, chunks: list[dict], manifest: list[dict] | None = None) -> None:
        self.by_id = {c["chunk_id"]: c for c in chunks}
        self.acts: dict[str, str] = {}          # slug -> title
        self.by_doc: dict[str, list[dict]] = defaultdict(list)
        self.cases: dict[str, dict] = {}        # slug -> {case_name, citation, court, date}
        for c in chunks:
            slug = c["chunk_id"].split(":", 1)[0]
            self.by_doc[slug].append(c)
            if c.get("source_type") == "legislation" and c.get("act_title"):
                self.acts.setdefault(slug, c["act_title"])
            elif c.get("source_type") == "case":
                self.cases.setdefault(slug, {k: c.get(k, "") for k in
                                             ("case_name", "citation", "court", "date")})
        self.manifest = load_manifest() if manifest is None else manifest

    # ---- Act resolution

    def resolve_act(self, name: str) -> str | None:
        """Slug for an Act named loosely: full title, slug, or abbreviation with
        year ("HA 1988", "LTA 1985", "Renters Rights Act")."""
        if not name:
            return None
        n = _norm(name)
        year = next(iter(re.findall(r"\b(1[89]\d\d|20\d\d)\b", n)), None)
        best, best_score = None, 0.0
        for slug, title in self.acts.items():
            t = _norm(title)
            if n in (t, _norm(slug)):
                return slug
            if year and year not in t:
                continue
            letters = re.sub(r"\b\d{4}\b", "", n).replace(" ", "")
            if letters and letters == _acronym(title):
                score = 0.95
            else:
                words = set(n.split()) - _ACRONYM_STOP - ({year} if year else set())
                twords = set(t.split()) - _ACRONYM_STOP
                score = len(words & twords) / max(len(words), 1) if words else 0.0
                if year and score:
                    score += 0.1
            if score > best_score:
                best, best_score = slug, score
        return best if best_score >= 0.6 else None

    # ---- provision lookup

    _SECTION = re.compile(r"\b(?:s|ss|sec|section|sections)\.?\s*(\d+[a-z]*)\s*(?:\(\s*([0-9a-z]+)\s*\))?",
                          re.IGNORECASE)
    _GROUND = re.compile(r"\bground\s*(\d+[a-z]*)", re.IGNORECASE)
    _SCHEDULE = re.compile(r"\b(?:sch|schedule)\.?\s*(\d+[a-z]*)", re.IGNORECASE)

    def lookup(self, act: str, provision: str) -> ToolResult:
        slug = self.resolve_act(act)
        if slug is None:
            available = "; ".join(sorted(self.acts.values()))
            return ToolResult(
                content=f"No Act matching '{act}' is in the corpus. Available Acts: {available}.",
                summary=f"unknown Act '{act}'")
        title = self.acts[slug]
        chunks = self.by_doc[slug]

        g = self._GROUND.search(provision)
        s = self._SECTION.search(provision)
        sch = self._SCHEDULE.search(provision)
        if g:
            ref = f"ground{g.group(1).upper()}"
            want = f"ground-{g.group(1).lower()}"
            matched = [c for c in chunks if c["chunk_id"].lower().endswith(want)]
            label = f"Ground {g.group(1).upper()}"
        elif s:
            ref = f"s{s.group(1).upper()}"
            sec, sub = s.group(1).lower(), (s.group(2) or "").lower()
            prefix = f"section-{sec}"
            matched = [c for c in chunks
                       if (tail := c["chunk_id"].split(":", 1)[1].lower()) == prefix
                       or tail.startswith(prefix + "-")]
            if sub:
                exact_sub = [c for c in matched
                             if c["chunk_id"].split(":", 1)[1].lower() == f"{prefix}-{sub}"]
                matched = exact_sub or matched
            label = f"s.{s.group(1)}" + (f"({s.group(2)})" if s.group(2) else "")
        elif sch:
            ref = None
            prefix = f"schedule-{sch.group(1).lower()}"
            matched = [c for c in chunks
                       if (tail := c["chunk_id"].split(":", 1)[1].lower()) == prefix
                       or tail.startswith(prefix + "-")]
            label = f"Schedule {sch.group(1)}"
        else:
            return ToolResult(
                content=(f"Could not parse provision '{provision}'. Use a form like 's.8', "
                         f"'section 13(2)', 'Ground 8' or 'Schedule 2'."),
                summary=f"unparsed provision '{provision}'")

        if not matched:
            repealed = [e for e in self.manifest
                        if ref and e["ref"] == ref and e["chunk_id"].startswith(slug + ":")]
            if repealed:
                e = repealed[0]
                return ToolResult(
                    content=(f"{e['citation']} ({e['heading']}) is NO LONGER IN FORCE: it has "
                             "been repealed or removed, and the current revised text keeps only "
                             "its heading. Do not state what it used to require."),
                    summary=f"{e['citation']} - repealed", repealed=repealed)
            nearby = list(dict.fromkeys(_provision_label(c) for c in chunks))[:30]
            return ToolResult(
                content=(f"{label} of {title} is not in this corpus (the corpus holds selected "
                         f"provisions only). Provisions held for this Act include: "
                         f"{', '.join(nearby)}."),
                summary=f"{title} {label} - not in corpus")

        shown = matched[:LOOKUP_MAX_CHUNKS]
        more = len(matched) - len(shown)
        body = "\n\n".join(f"[{c.get('citation', c['chunk_id'])}]\n{_trim(c['text'])}"
                           for c in shown)
        if more:
            rest = ", ".join(c.get("citation", c["chunk_id"]) for c in matched[len(shown):][:20])
            body += (f"\n\n({more} more subsections not shown: {rest}. "
                     f"Request one specifically, e.g. '{label}(2)'.)")
        return ToolResult(content=body, hits=[dict(c, distance=0.0) for c in shown],
                          summary=f"{title} {label} - {len(shown)} chunk(s)", exact=True)

    def list_sources(self) -> ToolResult:
        acts = "\n".join(f"- {t} ({len(self.by_doc[s])} provisions)"
                         for s, t in sorted(self.acts.items(), key=lambda kv: kv[1]))
        cases = "\n".join(f"- {c['case_name']} {c['citation']}"
                          for c in sorted(self.cases.values(), key=lambda c: c["case_name"]))
        content = ("Corpus: England residential landlord-tenant law (selected provisions).\n"
                   f"Acts and instruments:\n{acts}\n"
                   + (f"Cases:\n{cases}" if cases else "Cases: none in this deployment."))
        return ToolResult(content=content,
                          summary=f"{len(self.acts)} Acts, {len(self.cases)} cases")


def _label(h: dict) -> str:
    """Cases carry a neutral citation in "citation"; name the case too."""
    if h.get("source_type") == "case" and h.get("case_name"):
        return f"{h['case_name']} {h.get('citation', '')}".strip()
    return h.get("citation") or h.get("case_name", "")


def search(retriever, corpus: Corpus, query: str, source_type: str | None = None,
           act: str | None = None, k: int = SEARCH_DEFAULT_K) -> ToolResult:
    if not query or not query.strip():
        return ToolResult(content="Empty query.", summary="empty query")
    where: dict = {}
    if source_type in ("legislation", "case"):
        where["source_type"] = source_type
    note = ""
    if act:
        slug = corpus.resolve_act(act)
        if slug:
            where["act_title"] = corpus.acts[slug]
        else:
            note = f"(No Act matching '{act}' - searched the whole corpus instead.)\n\n"
    k = max(1, min(int(k or SEARCH_DEFAULT_K), SEARCH_MAX_K))
    hits = retriever.search(query, k=k, where=where or None)
    if not hits:
        return ToolResult(content=note + "No results.", summary=f"'{query}' - 0 results")
    body = "\n\n".join(
        f"[{_label(h)}] (relevance distance {h.get('distance', 1.0):.2f})\n{_trim(h['text'])}"
        for h in hits)
    return ToolResult(content=note + body, hits=hits,
                      summary=f"'{query}' - {len(hits)} results")


def execute(name: str, arguments: dict, retriever, corpus: Corpus) -> ToolResult:
    """Dispatch a tool call. Never raises: a bad call becomes a message the model
    can read and correct, rather than an aborted request."""
    try:
        if name == "search_law":
            return search(retriever, corpus, str(arguments.get("query", "")),
                          arguments.get("source_type"), arguments.get("act"))
        if name == "get_provision":
            return corpus.lookup(str(arguments.get("act", "")), str(arguments.get("provision", "")))
        if name == "list_sources":
            return corpus.list_sources()
    except Exception as exc:  # noqa: BLE001 - reported to the model, logged by the caller
        return ToolResult(content=f"Tool error: {exc}", summary=f"{name} error: {exc}")
    return ToolResult(content=f"Unknown tool '{name}'. Available: search_law, get_provision, "
                              "list_sources.", summary=f"unknown tool {name}")
