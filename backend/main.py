from __future__ import annotations

import json
import logging
import os
import queue
import threading
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import agent as agent_mod
import guard
import llm
import tools
from generate import generate_answer
from query_understanding import understand
from repeals import prompt_note, repealed_in, user_note
from schemas import AgentStep, QueryRequest, QueryResponse, ResponseMeta, Source
from scope import keyword_scope_check, retrieval_confidence_check, scope_short_answer
from telemetry import Metrics, Trace
from verify import format_warning, unsupported_citations

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(message)s")
log = logging.getLogger("rag.app")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
K = 6

# Filled in by the startup warm-up; "gpu" stays None until it has run (and for
# hosted providers, where there is no GPU of ours to report on).
RUNTIME: dict = {"gpu": None, "ready": False}

_CFG = llm.get_config()
METRICS = Metrics(price_in_per_m=_CFG.price_in_per_m, price_out_per_m=_CFG.price_out_per_m)
RATE_LIMIT, DAILY_CAP, CACHE = guard.from_env()

# --------------------------------------------------------------------------- retriever

_retriever = None
_corpus: tools.Corpus | None = None
_retriever_lock = threading.Lock()


def get_retriever():
    """Loaded on first use, not at import: the embedding model and BM25 index take
    seconds to build, and tests and tooling import this module without needing
    them. The server pre-loads it in the startup warm-up."""
    global _retriever
    with _retriever_lock:
        if _retriever is None:
            from retrieval import Retriever
            _retriever = Retriever()
        return _retriever


def get_corpus() -> tools.Corpus:
    global _corpus
    r = get_retriever()
    with _retriever_lock:
        if _corpus is None:
            _corpus = tools.Corpus(r.chunks)
        return _corpus


def set_retriever(retriever, corpus: tools.Corpus | None = None) -> None:
    """Swap in a retriever (tests)."""
    global _retriever, _corpus
    with _retriever_lock:
        _retriever, _corpus = retriever, corpus


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the retriever and the model in the background as the server starts.

    Otherwise the first question pays model load on top of an already slow
    first answer, which is exactly the moment a new user decides whether this
    thing is working. Failures are logged, never fatal: the app must still
    start if Ollama isn't up yet.
    """
    def _warm() -> None:
        try:
            get_retriever()
        except Exception:  # noqa: BLE001
            log.exception("retriever failed to load")
        RUNTIME.update(llm.warm_up())

    threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="UK Residential Rental Law RAG (demo)", lifespan=lifespan)


# --------------------------------------------------------------------------- pipeline

def retrieve(question: str, where: dict | None = None, trace: Trace | None = None,
             allow_rewrite: bool = True) -> tuple[list[dict], str]:
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

    ``allow_rewrite=False`` is the LLM-free path the CI retrieval eval uses.
    """
    trace = trace or Trace()
    retriever = get_retriever()
    with trace.stage("retrieval"):
        hits = retriever.search(question, k=K, where=where)
    if retrieval_confidence_check(hits) is None:
        return hits, question
    if not allow_rewrite:
        trace.outcome["rewrite_needed"] = True
        return hits, question

    with trace.stage("rewrite"):
        understood = understand(question, trace=trace)
    trace.outcome["rewritten"] = bool(understood.get("rewritten"))
    with trace.stage("retrieval"):
        rewritten = retriever.multi_search(understood["search_queries"], k=K, where=where)
    # Keep the rewrite only if it actually did better; a rewrite of a question the
    # corpus simply can't answer shouldn't replace the original evidence.
    if retrieval_confidence_check(rewritten) is None:
        return rewritten, understood["legal_question"]
    best = min((h["distance"] for h in hits), default=1.0)
    return ((rewritten, understood["legal_question"])
            if min((h["distance"] for h in rewritten), default=1.0) < best else (hits, question))


def answer_for(question: str, facts: dict | None, hits: list[dict],
               legal_question: str, trace: Trace | None = None,
               extra_repealed: list[dict] | None = None) -> tuple[dict, list[dict]]:
    """Generate an answer, warning the model about any repealed provision the
    question names (or the agent looked up). Shared with the eval so both
    exercise the same path."""
    repealed = repealed_in(question)
    for e in extra_repealed or []:
        if all(e["chunk_id"] != r["chunk_id"] for r in repealed):
            repealed.append(e)
    trace = trace or Trace()
    with trace.stage("generation"):
        generated = generate_answer(question, facts, hits, legal_question,
                                    prompt_note(repealed) if repealed else None, trace=trace)
    return generated, repealed


def _agent_events(question: str, facts: dict | None, trace: Trace):
    """Run the agent on a worker thread, yielding each tool step as it happens.

    The agent loop is synchronous and reports steps through a callback; a queue
    turns that into events the stream can forward live, so the UI shows "looked
    up Housing Act 1988 s.21" while the agent is still working. Returns the
    AgentResult as the generator's return value.
    """
    q: queue.Queue = queue.Queue()
    holder: dict = {}
    retriever, corpus = get_retriever(), get_corpus()

    def work() -> None:
        try:
            holder["result"] = agent_mod.run_agent(
                question, retriever, corpus, facts=facts, trace=trace,
                on_step=lambda s: q.put(s))
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller's thread
            holder["error"] = exc
        finally:
            q.put(None)

    threading.Thread(target=work, daemon=True).start()
    while (step := q.get()) is not None:
        yield "step", AgentStep(**vars(step))
    if "error" in holder:
        raise holder["error"]
    return holder["result"]


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


def _abstain(short: str, note: str, sources: list[Source] | None = None) -> QueryResponse:
    return QueryResponse(abstained=True, short_answer=short, applicable_law="",
                         how_it_applies="", uncertainties="", scope_note=note,
                         sources=sources or [])


def run_query(req: QueryRequest, trace: Trace | None = None) -> Iterator[tuple[str, object]]:
    """The query pipeline, as a sequence of events.

    Generation is by far the slowest step (on a machine without a GPU it is most
    of a minute), but the sources are known a second or two in. Emitting them as
    a separate event lets the UI show real evidence while the answer is still
    being written, instead of a spinner that is indistinguishable from a hang.
    Agent mode additionally emits a "step" event per tool call.

    Both endpoints consume this, so the plain JSON API and the streaming one
    cannot drift apart.
    """
    trace = trace or Trace(mode=req.mode)
    with trace.stage("scope_gate"):
        oos_reason = keyword_scope_check(req.question)
    if oos_reason:
        trace.outcome.update(abstained=True, gate="keyword")
        yield "answer", _abstain(scope_short_answer(oos_reason), oos_reason)
        return

    facts = req.facts.model_dump(exclude_none=True) if req.facts else None
    where = req.filters.as_where() if req.filters else None
    steps: list[AgentStep] = []
    extra_repealed: list[dict] = []
    exact_hits = 0

    if req.mode == "agent":
        result = None
        gen = _agent_events(req.question, facts, trace)
        while True:
            try:
                kind, step = next(gen)
            except StopIteration as stop:
                result = stop.value
                break
            steps.append(step)
            yield kind, step
        hits, legal_question = result.evidence, req.question
        extra_repealed, exact_hits = result.repealed, result.exact_hits
        trace.outcome["agent_stop"] = result.stop_reason
        if not hits:
            # The agent made no tool calls (a small local model sometimes
            # won't) or found nothing: fall back to ordinary retrieval, so agent
            # mode is never worse than having no agent at all.
            trace.outcome["agent_fallback"] = True
            hits, legal_question = retrieve(req.question, where, trace)
    else:
        hits, legal_question = retrieve(req.question, where, trace)

    # An exact lookup is the agent naming a provision outright; similarity
    # distance is meaningless for it, so it satisfies the gate on its own.
    conf_reason = None if exact_hits else retrieval_confidence_check(hits)
    if conf_reason:
        trace.outcome.update(abstained=True, gate="confidence")
        resp = _abstain("The corpus doesn't contain sufficiently relevant material to answer "
                        "this reliably.", conf_reason, _sources(hits[:3]))
        resp.agent_steps = steps
        yield "answer", resp
        return

    yield "sources", QueryResponse(
        abstained=False, short_answer="", applicable_law="", how_it_applies="",
        uncertainties="", sources=_sources(hits), agent_steps=steps,
    )

    generated, repealed = answer_for(req.question, facts, hits, legal_question, trace,
                                     extra_repealed)

    uncertainties = generated.get("uncertainties", "")
    if repealed:
        # Stated independently of the model: this is the one fact we know for
        # certain and the model is least reliable about, so it must not depend on
        # the model having followed the instruction.
        note = user_note(repealed)
        uncertainties = f"{note}\n\n{uncertainties}".strip() if uncertainties else note
    with trace.stage("verify"):
        unsupported = unsupported_citations(
            [generated.get("applicable_law", ""), generated.get("short_answer", ""),
             generated.get("how_it_applies", "")],
            hits,
        )
    if unsupported:
        warning = format_warning(unsupported)
        uncertainties = f"{uncertainties}\n\n{warning}".strip() if uncertainties else warning

    trace.outcome.update(abstained=False, unsupported_citations=len(unsupported),
                         citations=len(generated.get("citations", [])),
                         generation_failed=generated.get("uncertainties", "")
                         .startswith("Generation error"))
    yield "answer", QueryResponse(
        abstained=False,
        short_answer=generated.get("short_answer", ""),
        applicable_law=generated.get("applicable_law", ""),
        how_it_applies=generated.get("how_it_applies", ""),
        uncertainties=uncertainties,
        citations=generated.get("citations", []),
        unsupported_citations=unsupported,
        sources=_sources(hits),
        agent_steps=steps,
    )


# --------------------------------------------------------------------------- serving

def _client_id(request: Request) -> str:
    # Behind a reverse proxy the socket peer is the proxy; the visitor is the
    # first X-Forwarded-For entry.
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def _cache_key(req: QueryRequest) -> str:
    cfg = llm.get_config()
    return CACHE.key(req.mode, CACHE.normalize_question(req.question),
                     req.facts.model_dump() if req.facts else None,
                     req.filters.model_dump() if req.filters else None,
                     cfg.provider, cfg.model)


def _admit(req: QueryRequest, request: Request) -> QueryResponse | None:
    """Rate-limit, then serve from cache if possible, then charge the daily cap.
    Raises 429 when refused; returns a cached response on a hit."""
    if not RATE_LIMIT.allow(_client_id(request)):
        raise HTTPException(429, "Too many questions in the last minute - please wait a "
                                 "moment and try again.")
    cached = CACHE.get(_cache_key(req))
    if cached is not None:
        return cached
    if not DAILY_CAP.take():
        raise HTTPException(429, "This free demo has reached its daily question limit. "
                                 "Please try again tomorrow, or run it locally (see README).")
    return None


def _with_meta(resp: QueryResponse, trace: Trace, cached: bool = False) -> QueryResponse:
    cfg = llm.get_config()
    out = resp.model_copy(deep=True)
    out.meta = ResponseMeta(**trace.meta(cfg.provider, cfg.model, cached=cached))
    return out


def _cacheable(resp: QueryResponse) -> bool:
    return not resp.uncertainties.startswith("Generation error")


def _events(req: QueryRequest, cached: QueryResponse | None) -> Iterator[tuple[str, object]]:
    """run_query plus caching, response metadata and the per-request trace."""
    trace = Trace(mode=req.mode)
    if cached is not None:
        if cached.sources and not cached.abstained:
            yield "sources", cached
        trace.finish(METRICS, cached=True, abstained=cached.abstained)
        yield "answer", _with_meta(cached, trace, cached=True)
        return
    try:
        for kind, payload in run_query(req, trace):
            if kind == "answer":
                if _cacheable(payload):
                    CACHE.put(_cache_key(req), payload)
                payload = _with_meta(payload, trace)
                trace.finish(METRICS)
            yield kind, payload
    except Exception as exc:
        trace.finish(METRICS, error=type(exc).__name__, error_detail=str(exc)[:300])
        raise


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest, request: Request) -> QueryResponse:
    cached = _admit(req, request)
    answer = None
    try:
        for kind, payload in _events(req, cached):
            if kind == "answer":
                answer = payload
    except llm.LLMError as exc:
        raise HTTPException(502, f"The language model provider failed: {exc}") from exc
    return answer


@app.post("/query/stream")
def query_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    cached = _admit(req, request)

    def events():
        try:
            for kind, payload in _events(req, cached):
                yield f"event: {kind}\ndata: {payload.model_dump_json()}\n\n"
        except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
            yield f"event: error\ndata: {json.dumps({'detail': str(exc)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.get("/health")
def health() -> dict:
    corpus = None
    if _retriever is not None:
        c = get_corpus()
        corpus = {"acts": len(c.acts), "cases": len(c.cases), "chunks": len(c.by_id)}
    return {
        **llm.get_config().public(),
        **RUNTIME,
        "corpus": corpus,
        "modes": ["rag", "agent"],
        "version": os.environ.get("APP_VERSION", "dev"),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> PlainTextResponse:
    """Prometheus exposition format."""
    return PlainTextResponse(METRICS.prometheus(), media_type="text/plain; version=0.0.4")


@app.get("/metrics/summary")
def metrics_summary() -> dict:
    """Human-readable rolling metrics: p50/p95 latency, tokens and est. cost per answer."""
    return {
        **llm.get_config().public(),
        **METRICS.summary(),
        "cache": {"hits": CACHE.hits, "misses": CACHE.misses},
        "daily_cap": {"limit": DAILY_CAP.limit, "used_today": DAILY_CAP.used},
    }


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
