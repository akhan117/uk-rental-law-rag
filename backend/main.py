from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from pathlib import Path

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from generate import MODEL, OLLAMA_URL, generate_answer
from query_understanding import understand
from repeals import prompt_note, repealed_in, user_note
from retrieval import Retriever
from schemas import QueryRequest, QueryResponse, Source
from scope import keyword_scope_check, retrieval_confidence_check, scope_short_answer
from verify import format_warning, unsupported_citations

app = FastAPI(title="UK Residential Rental Law RAG (demo)")
retriever = Retriever()


@app.on_event("startup")
def warm_model() -> None:
    """Load the model into Ollama in the background as the server starts.

    Otherwise the first question pays model load on top of an already slow
    first answer, which is exactly the moment a new user decides whether this
    thing is working. Failures are ignored: warming is an optimisation, and the
    app must still start if Ollama isn't up yet.
    """
    def _warm() -> None:
        try:
            httpx.post(f"{OLLAMA_URL}/api/generate", timeout=180.0, json={
                "model": MODEL, "prompt": "ok", "stream": False,
                "think": False, "options": {"num_predict": 1},
            })
            # Once loaded, Ollama reports how much of the model sits in VRAM.
            # Zero means it fell back to CPU, which is a ~10x slowdown (measured
            # on this project: 5s -> 58s per answer). Ollama does that silently,
            # and it can happen for transient reasons - a failed CUDA probe when
            # two instances race at startup - so a machine that worked yesterday
            # can be crawling today with nothing to show why. Record it so the UI
            # can say so instead of just seeming broken.
            ps = httpx.get(f"{OLLAMA_URL}/api/ps", timeout=10.0).json()
            loaded = [m for m in ps.get("models", []) if m.get("name") == MODEL]
            RUNTIME["gpu"] = bool(loaded and loaded[0].get("size_vram", 0) > 0)
            RUNTIME["ready"] = True
        except Exception:
            pass

    threading.Thread(target=_warm, daemon=True).start()


@app.get("/health")
def health() -> dict:
    return {"model": MODEL, **RUNTIME}

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
K = 6

# Filled in by the startup warm-up; "gpu" stays None until it has run.
RUNTIME: dict = {"gpu": None, "ready": False}


def retrieve(question: str, where: dict | None = None) -> tuple[list[dict], str]:
    """Retrieve for a question, rewriting it only if its own wording retrieves badly.

    Rewriting unconditionally made well-formed questions *worse*. With k=6 split
    across three sub-queries each gets two slots, so a good question whose own
    wording would have found the provision at rank 3 loses it - measured: q09,
    q16 and q27 all retrieve correctly on the raw question alone and all missed
    once their slots were shared with paraphrases.

    A question's own phrasing is the best query whenever it is already precise,
    and the confidence check is exactly the signal for "it wasn't": answerable
    well-formed questions return 3-6 chunks within 0.30, while the narrative and
    misspelled ones that need rewriting return 0-2. So retrieve plainly first and
    only pay for a rewrite when that comes back thin - which also means most
    questions skip the extra model call entirely.
    """
    hits = retriever.search(question, k=K, where=where)
    if retrieval_confidence_check(hits) is None:
        return hits, question

    understood = understand(question)
    rewritten = retriever.multi_search(understood["search_queries"], k=K, where=where)
    # Keep the rewrite only if it actually did better; a rewrite of a question the
    # corpus simply can't answer shouldn't replace the original evidence.
    if retrieval_confidence_check(rewritten) is None:
        return rewritten, understood["legal_question"]
    best = min(h["distance"] for h in hits)
    return ((rewritten, understood["legal_question"])
            if min(h["distance"] for h in rewritten) < best else (hits, question))


def answer_for(question: str, facts: dict | None, hits: list[dict],
               legal_question: str) -> tuple[dict, list[dict]]:
    """Generate an answer, warning the model about any repealed provision the
    question names. Shared with the eval so both exercise the same path."""
    repealed = repealed_in(question)
    generated = generate_answer(question, facts, hits, legal_question,
                                prompt_note(repealed) if repealed else None)
    return generated, repealed


def _sources(hits: list[dict]) -> list[Source]:
    return [
        Source(
            chunk_id=h["chunk_id"],
            citation=h.get("citation") or h.get("case_name", ""),
            url=h.get("url", ""),
            source_type=h.get("source_type", ""),
            text=h["text"],
        )
        for h in hits
    ]


def run_query(req: QueryRequest) -> Iterator[tuple[str, QueryResponse]]:
    """The query pipeline, as a sequence of events.

    Generation is by far the slowest step (on a machine without a GPU it is most
    of a minute), but the sources are known a second or two in. Emitting them as
    a separate event lets the UI show real evidence while the answer is still
    being written, instead of a spinner that is indistinguishable from a hang.

    Both endpoints consume this, so the plain JSON API and the streaming one
    cannot drift apart.
    """
    oos_reason = keyword_scope_check(req.question)
    if oos_reason:
        yield "answer", QueryResponse(
            abstained=True,
            short_answer=scope_short_answer(oos_reason),
            applicable_law="",
            how_it_applies="",
            uncertainties="",
            scope_note=oos_reason,
        )
        return

    hits, legal_question = retrieve(req.question,
                                    req.filters.as_where() if req.filters else None)
    conf_reason = retrieval_confidence_check(hits)
    if conf_reason:
        yield "answer", QueryResponse(
            abstained=True,
            short_answer="The corpus doesn't contain sufficiently relevant material to answer this reliably.",
            applicable_law="",
            how_it_applies="",
            uncertainties="",
            scope_note=conf_reason,
            sources=_sources(hits[:3]),
        )
        return

    yield "sources", QueryResponse(
        abstained=False, short_answer="", applicable_law="", how_it_applies="",
        uncertainties="", sources=_sources(hits),
    )

    facts = req.facts.model_dump(exclude_none=True) if req.facts else None
    generated, repealed = answer_for(req.question, facts, hits, legal_question)

    uncertainties = generated.get("uncertainties", "")
    if repealed:
        # Stated independently of the model: this is the one fact we know for
        # certain and the model is least reliable about, so it must not depend on
        # the model having followed the instruction.
        note = user_note(repealed)
        uncertainties = f"{note}\n\n{uncertainties}".strip() if uncertainties else note
    unsupported = unsupported_citations(
        [generated.get("applicable_law", ""), generated.get("short_answer", ""),
         generated.get("how_it_applies", "")],
        hits,
    )
    if unsupported:
        warning = format_warning(unsupported)
        uncertainties = f"{uncertainties}\n\n{warning}".strip() if uncertainties else warning

    yield "answer", QueryResponse(
        abstained=False,
        short_answer=generated.get("short_answer", ""),
        applicable_law=generated.get("applicable_law", ""),
        how_it_applies=generated.get("how_it_applies", ""),
        uncertainties=uncertainties,
        citations=generated.get("citations", []),
        unsupported_citations=unsupported,
        sources=_sources(hits),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    answer = None
    for kind, payload in run_query(req):
        if kind == "answer":
            answer = payload
    return answer


@app.post("/query/stream")
def query_stream(req: QueryRequest) -> StreamingResponse:
    def events():
        try:
            for kind, payload in run_query(req):
                yield f"event: {kind}\ndata: {payload.model_dump_json()}\n\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class NoCacheStaticFiles(StaticFiles):
    """Serve the UI with revalidation forced.

    Starlette sends ETag and Last-Modified but no Cache-Control, so browsers
    apply heuristic caching and can hold a stale app.js for a long time. That
    bites hardest on exactly the handover path this project cares about: the
    launcher skips starting the server if one is already listening, so after an
    update a user can end up on a cached frontend talking to a new backend with
    no indication anything is wrong. "no-cache" still permits a 304 via the
    ETag, so this costs a conditional request, not a re-download.
    """

    def file_response(self, *args, **kwargs):
        resp = super().file_response(*args, **kwargs)
        resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        return resp


app.mount("/", NoCacheStaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
