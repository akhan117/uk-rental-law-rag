"""MCP server: the corpus tools, plus the full grounded pipeline, for any MCP client.

The agent's tools (tools.py) are exactly the kind of thing MCP exists to share,
so they are exposed here unchanged: an assistant like Claude Desktop or Claude
Code can search the corpus, fetch a provision by number and see what's repealed,
with the same exact-lookup and repeal behaviour the in-app agent gets.
``answer_question`` additionally runs the whole app pipeline (scope gate,
retrieval, citation-constrained generation, verification) with the configured
LLM backend, for clients that want an answer rather than raw evidence.

Run (stdio):   python backend/mcp_server.py
Requires:      pip install -r requirements-mcp.txt
Client config: see README, "MCP server".

stdout is the protocol channel on stdio, so nothing here prints; logging goes
to stderr (logging's default stream).

No ``from __future__ import annotations`` here: FastMCP builds each tool's input
schema by inspecting the real annotation objects, and string annotations break it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# mcp 1.x: 2.x renames FastMCP and requires a newer pydantic/starlette than the
# FastAPI version this app pins, so the 1.x line is used deliberately.
from mcp.server.fastmcp import FastMCP  # noqa: E402

import main  # noqa: E402
import tools  # noqa: E402
from schemas import QueryRequest  # noqa: E402

server = FastMCP(
    name="uk-rental-law",
    instructions=(
        "Tools over a curated corpus of England residential landlord-tenant law "
        "(Housing Acts 1988/2004, Landlord and Tenant Act 1985, Protection from Eviction Act "
        "1977, Tenant Fees Act 2019, Renters' Rights Act 2025, and others). Legal information, "
        "not legal advice. Prefer get_provision when a provision is named; it also reports "
        "repealed provisions, which search cannot."),
)


@server.tool()
def search_law(query: str, source_type: str | None = None, act: str | None = None) -> str:
    """Hybrid semantic + keyword search over the corpus. Use statutory wording
    ("possession", "tenancy deposit", "repairing covenant"). Optionally restrict to
    source_type "legislation" or "case", or to one Act by name."""
    return tools.search(main.get_retriever(), main.get_corpus(), query, source_type, act).content


@server.tool()
def get_provision(act: str, provision: str) -> str:
    """Exact text of a provision, e.g. act="Housing Act 1988", provision="s.8(4AA)",
    "Ground 8" or "section 21". Says so if the provision has been repealed."""
    return main.get_corpus().lookup(act, provision).content


@server.tool()
def list_sources() -> str:
    """The Acts, instruments and cases this corpus holds."""
    return main.get_corpus().list_sources().content


@server.tool()
def answer_question(question: str, mode: str = "rag") -> str:
    """Full grounded answer with citations, via the app's own pipeline and the
    configured LLM (LLM_PROVIDER). mode: "rag" or "agent". Abstains when the
    corpus doesn't support an answer."""
    req = QueryRequest(question=question, mode="agent" if mode == "agent" else "rag")
    answer = None
    for kind, payload in main.run_query(req):
        if kind == "answer":
            answer = payload
    if answer is None:
        return "No answer was produced."
    if answer.abstained:
        return f"Not answered from the corpus: {answer.short_answer}\n{answer.scope_note or ''}"
    parts = [f"Short answer: {answer.short_answer}",
             f"Applicable law: {answer.applicable_law}",
             f"How it applies: {answer.how_it_applies}"]
    if answer.uncertainties:
        parts.append(f"Uncertainties: {answer.uncertainties}")
    parts.append("Sources: " + "; ".join(f"{s.citation} <{s.url}>" for s in answer.sources))
    return "\n\n".join(parts)


if __name__ == "__main__":
    # Load the embedding model and index on the main thread before serving.
    # Left lazy, the first tool call imported torch on a worker thread inside the
    # server's event loop and hung (observed on Windows); loading up front also
    # means the first call is fast rather than ~10s.
    main.get_retriever()
    main.get_corpus()
    server.run()
