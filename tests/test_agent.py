"""The agent loop: tool execution, bounds, evidence ranking, repeal propagation."""
from __future__ import annotations

import agent
import tools
from conftest import CHUNKS, MANIFEST, FakeRetriever, chat_reply
from telemetry import Trace


def _run(question="How much notice for a rent increase?", **kw):
    retriever = FakeRetriever()
    return agent.run_agent(question, retriever, tools.Corpus(CHUNKS, manifest=MANIFEST), **kw)


def test_lookup_then_stop(provider):
    fake = provider(
        chat_reply(tool_calls=[("get_provision", {"act": "Housing Act 1988", "provision": "s.13(2)"})]),
        chat_reply("Found s.13(2)."))
    trace = Trace(mode="agent")
    steps = []
    result = _run(trace=trace, on_step=steps.append)
    assert result.stop_reason == "done"
    assert [h["chunk_id"] for h in result.evidence] == ["housing-act-1988:section-13-2"]
    assert result.exact_hits == 1
    assert [s.tool for s in steps] == ["get_provision"] and steps[0].round == 1
    assert trace.tool_calls == ["get_provision"] and len(trace.llm_calls) == 2
    # The tool result went back to the model, tied to its call id.
    tool_msg = fake.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_0"
    assert "two months" in tool_msg["content"]


def test_exact_hits_rank_ahead_of_search_hits(provider):
    provider(
        chat_reply(tool_calls=[("search_law", {"query": "tenancy deposit"})]),
        chat_reply(tool_calls=[("get_provision", {"act": "LTA 1985", "provision": "s.11(1)"})]),
        chat_reply("done"))
    result = _run()
    assert result.evidence[0]["chunk_id"] == "landlord-and-tenant-act-1985:section-11-1"
    assert len({h["chunk_id"] for h in result.evidence}) == len(result.evidence)  # deduped
    assert len(result.evidence) <= agent.EVIDENCE_K


def test_round_limit_bounds_a_model_that_never_stops(provider):
    provider(chat_reply(tool_calls=[("search_law", {"query": "rent"})]))
    result = _run(max_rounds=3)
    assert result.stop_reason == "max_rounds" and len(result.steps) == 3


def test_calls_per_round_are_capped(provider):
    many = [("search_law", {"query": f"q{i}"}) for i in range(6)]
    provider(chat_reply(tool_calls=many), chat_reply("done"))
    assert len(_run().steps) == agent.MAX_CALLS_PER_ROUND


def test_repealed_lookup_is_propagated(provider):
    provider(chat_reply(tool_calls=[("get_provision", {"act": "Housing Act 1988",
                                                       "provision": "section 21"})]),
             chat_reply("s.21 is repealed."))
    result = _run("What notice does a section 21 notice need?")
    assert result.repealed == MANIFEST and result.evidence == []


def test_no_tool_calls_yields_no_evidence(provider):
    provider(chat_reply("This is out of scope."))
    result = _run("What is the best mortgage rate?")
    assert result.evidence == [] and result.steps == [] and result.stop_reason == "done"
