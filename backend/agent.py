"""Agent mode: the model plans its own retrieval with tools, then the normal
grounded generator writes the answer from what it found.

Why split it that way. Single-shot RAG fails in two measured places (README):
multi-hop questions, where no single chunk states the combined rule ("how does
eviction work now that s.21 is gone?" needs "shortholds abolished" *and*
"possession only on Schedule 2 grounds"), and questions naming a provision that
similarity ranking then fails to surface. Both are retrieval-planning problems,
so that's the part handed to the agent: it can search more than once, look a
provision up exactly, and follow a cross-reference it reads in an excerpt.

Writing the answer is *not* handed over. Evidence goes through the same
citation-enum-constrained, schema-decoded, post-verified generate_answer as RAG
mode, so every grounding guarantee still holds, and the eval comparison between
the modes isolates a single variable: how the evidence was chosen. The evidence
cap equals RAG mode's k for the same reason.

Hand-rolled rather than LangGraph: the control flow is one bounded loop, and
writing it out keeps every step visible in the trace and debuggable against a
4B local model, which is where tool calling is least reliable.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import llm
import tools

MAX_ROUNDS = 4
MAX_CALLS_PER_ROUND = 3
EVIDENCE_K = 6   # same as RAG mode's k, so the two modes compare like for like

# Token budget for each planning turn. Measured: Qwen3-4B on Ollama 0.40 writes
# ~2,000 tokens of reasoning before every tool call when tools are attached -
# whether think is false, true, or "/no_think" is in the prompt - which is ~60s
# per round on an RTX 3070 Laptop GPU. A tight budget therefore means "no tool
# call" locally, and the pipeline falls back to single-shot retrieval (never
# worse than RAG mode). Hosted models (gpt-oss-20b at reasoning_effort=low) plan
# in a few hundred tokens. Raise this to run the agent locally anyway.
PLAN_MAX_TOKENS = int(os.environ.get("AGENT_PLAN_MAX_TOKENS", "").strip() or 600)

SYSTEM_PROMPT = """You are the research step of a legal information assistant for England \
residential landlord-tenant law. Your job is to FIND the provisions and case law that answer the \
user's question, using the tools. You do not write the answer - a separate step writes it from what \
you find, and it can only use what your tool calls returned.

Strategy:
1. If the question names a provision ("section 21", "Ground 8", "s.13"), fetch it with get_provision.
2. Otherwise use search_law with statutory wording, not the user's everyday words: "possession" not
   "kick out", "tenancy" not "lease", "notice to quit" not "move out". One topic per search; a
   question about two things needs two searches.
3. If an excerpt depends on another provision ("subject to section 21A", "Ground 1A in Schedule 2"),
   and that provision is needed to answer, fetch it with get_provision.
4. Stop as soon as you have the provisions that answer the question - usually 1 to 3 tool calls.
   Then reply with a single line summarising what you found, without calling a tool.
5. If the question is not about England residential tenancy law, do not search; reply that it is
   out of scope.

Never answer from memory. Never invent a provision to look up - if unsure what exists, search."""


@dataclass
class AgentStep:
    round: int
    tool: str
    arguments: dict
    summary: str
    latency_ms: int


@dataclass
class AgentResult:
    evidence: list[dict] = field(default_factory=list)
    steps: list[AgentStep] = field(default_factory=list)
    exact_hits: int = 0
    repealed: list[dict] = field(default_factory=list)
    stop_reason: str = ""
    notes: str = ""


def _rank_evidence(found: list[tuple[bool, dict]], k: int) -> list[dict]:
    """Exact lookups first (the agent asked for that provision by name), then
    search hits in the order they arrived; deduplicated, capped at k."""
    ordered = [h for exact, h in found if exact] + [h for exact, h in found if not exact]
    out: list[dict] = []
    seen: set[str] = set()
    for h in ordered:
        if h["chunk_id"] in seen:
            continue
        seen.add(h["chunk_id"])
        out.append(h)
        if len(out) >= k:
            break
    return out


def run_agent(question: str, retriever, corpus: tools.Corpus, *, facts: dict | None = None,
              trace=None, on_step: Callable[[AgentStep], None] | None = None,
              max_rounds: int = MAX_ROUNDS, evidence_k: int = EVIDENCE_K) -> AgentResult:
    user = f"Question: {question}"
    if facts:
        user += "\nKnown facts: " + "; ".join(f"{k}: {v}" for k, v in facts.items() if v)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user}]
    result = AgentResult()
    found: list[tuple[bool, dict]] = []

    for rnd in range(1, max_rounds + 1):
        reply = llm.chat(messages, tools.TOOL_SCHEMAS, temperature=0.0,
                         max_tokens=PLAN_MAX_TOKENS, trace=trace, stage="agent_plan")
        calls = reply.tool_calls[:MAX_CALLS_PER_ROUND]
        if not calls:
            result.stop_reason = "done"
            result.notes = reply.text.strip()
            break

        messages.append({
            "role": "assistant", "content": reply.text or "",
            "tool_calls": [{"id": c.id, "type": "function",
                            "function": {"name": c.name,
                                         "arguments": _json(c.arguments)}} for c in calls],
        })
        for call in calls:
            t0 = time.perf_counter()
            out = tools.execute(call.name, call.arguments, retriever, corpus)
            step = AgentStep(round=rnd, tool=call.name, arguments=call.arguments,
                             summary=out.summary,
                             latency_ms=round((time.perf_counter() - t0) * 1000))
            result.steps.append(step)
            if trace is not None:
                trace.tool_calls.append(call.name)
            if on_step is not None:
                on_step(step)
            found.extend((out.exact, h) for h in out.hits)
            result.exact_hits += len(out.hits) if out.exact else 0
            result.repealed.extend(e for e in out.repealed if e not in result.repealed)
            messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name,
                             "content": out.content})
    else:
        result.stop_reason = "max_rounds"

    result.evidence = _rank_evidence(found, evidence_k)
    return result


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)
