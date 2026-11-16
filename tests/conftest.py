"""Shared fixtures. No test here needs a model, an index or the network.

The LLM is replaced at the HTTP layer (httpx.MockTransport), not by patching
our own functions, so the real provider adapters - payload shape, JSON-mode
ladder, retries, usage accounting - are what's under test.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "backend", ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import llm  # noqa: E402

CHUNKS = [
    {"chunk_id": "housing-act-1988:section-13-1", "source_type": "legislation",
     "act_title": "Housing Act 1988", "citation": "Housing Act 1988, s.13(1)",
     "parent_section": "section-13", "url": "https://example.test/ha1988/13/1",
     "text": "Increases of rent. (s.13(1))\n\nThis section applies to a statutory periodic "
             "tenancy and any other periodic assured tenancy."},
    {"chunk_id": "housing-act-1988:section-13-2", "source_type": "legislation",
     "act_title": "Housing Act 1988", "citation": "Housing Act 1988, s.13(2)",
     "parent_section": "section-13", "url": "https://example.test/ha1988/13/2",
     "text": "Increases of rent. (s.13(2))\n\nThe landlord may propose a new rent by notice "
             "giving a minimum period of two months before the increase in rent takes effect."},
    {"chunk_id": "housing-act-1988:schedule-2-part-I-crossheading-ground-8",
     "source_type": "legislation", "act_title": "Housing Act 1988",
     "citation": "Housing Act 1988, Sch.2 Ground 8",
     "part_title": "Grounds on which Court must order possession",
     "url": "https://example.test/ha1988/sch2/ground8",
     "text": "Housing Act 1988, Schedule 2, Ground 8\n\nBoth at the date of service of the notice "
             "and at the date of the hearing at least three months rent is unpaid."},
    {"chunk_id": "housing-act-2004:section-213-3", "source_type": "legislation",
     "act_title": "Housing Act 2004", "citation": "Housing Act 2004, s.213(3)",
     "parent_section": "section-213", "url": "https://example.test/ha2004/213/3",
     "text": "Requirements relating to tenancy deposits. (s.213(3))\n\nThe initial requirements "
             "of an authorised scheme must be complied with by the landlord within the period "
             "of 30 days beginning with the date on which the deposit is received."},
    {"chunk_id": "landlord-and-tenant-act-1985:section-11-1", "source_type": "legislation",
     "act_title": "Landlord and Tenant Act 1985",
     "citation": "Landlord and Tenant Act 1985, s.11(1)", "parent_section": "section-11",
     "url": "https://example.test/lta1985/11/1",
     "text": "Repairing obligations in short leases. (s.11(1))\n\nThere is implied a covenant by "
             "the lessor to keep in repair the structure and exterior of the dwelling-house."},
]

MANIFEST = [{"ref": "s21", "chunk_id": "housing-act-1988:section-21",
             "citation": "Housing Act 1988, s.21",
             "heading": "Recovery of possession on expiry or termination of assured shorthold "
                        "tenancy.", "url": ""}]


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", s.lower()))


class FakeRetriever:
    """Ranks by word overlap and reports one fixed distance for every hit, so a
    test decides directly whether the scope gate's confidence check passes
    (distance <= 0.30 on >= 2 hits) or fails."""

    def __init__(self, chunks=None, distance: float = 0.2) -> None:
        self.chunks = [dict(c) for c in (chunks or CHUNKS)]
        self.distance = distance
        self.queries: list[tuple[str, dict | None]] = []

    def search(self, query: str, k: int = 6, where: dict | None = None) -> list[dict]:
        self.queries.append((query, where))
        pool = [c for c in self.chunks
                if not where or all(c.get(key) == v for key, v in where.items())]
        q = _words(query)
        ranked = sorted(pool, key=lambda c: -len(q & _words(c["text"])))
        return [dict(c, distance=self.distance) for c in ranked[:k]]

    def multi_search(self, queries: list[str], k: int = 6, where: dict | None = None):
        return self.search(" ".join(queries), k=k, where=where)


# --------------------------------------------------------------------------- fake LLM

def chat_reply(content: str = "", tool_calls: list[tuple[str, dict]] | None = None,
               usage: tuple[int, int] = (100, 20)) -> dict:
    msg: dict = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": name, "arguments": json.dumps(args)}}
            for i, (name, args) in enumerate(tool_calls)]
    return {"choices": [{"message": msg}], "model": "test-model",
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]}}


def json_reply(obj: dict, usage: tuple[int, int] = (400, 80)) -> dict:
    return chat_reply(json.dumps(obj), usage=usage)


ANSWER = {
    "short_answer": "The landlord must give at least two months' notice of a rent increase.",
    "applicable_law": "Housing Act 1988, s.13(2) requires a minimum period of two months.",
    "how_it_applies": "A proposed increase cannot take effect before that period ends.",
    "uncertainties": "",
    "citations": ["Housing Act 1988, s.13(2)"],
}


class FakeProvider:
    """A scripted OpenAI-compatible (or Ollama) server. Replies are consumed in
    order; the last one repeats. A reply may be a dict (200 JSON), an
    httpx.Response, or a callable(body) -> either."""

    def __init__(self, replies) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.paths: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        self.paths.append(request.url.path)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if callable(reply):
            reply = reply(body)
        if isinstance(reply, httpx.Response):
            return reply
        return httpx.Response(200, json=reply)


def _no_network(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"unexpected LLM call in a test: {request.url}")


@pytest.fixture(autouse=True)
def _isolate_llm(monkeypatch):
    """Every test starts with an LLM that fails loudly if called."""
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    llm.configure(llm.LLMConfig(provider="groq", model="test-model",
                                base_url="https://llm.test/v1", api_key="k", max_retries=2),
                  transport=httpx.MockTransport(_no_network))
    yield
    llm.configure(llm.LLMConfig(provider="groq", model="test-model",
                                base_url="https://llm.test/v1", api_key="k"),
                  transport=httpx.MockTransport(_no_network))


@pytest.fixture
def provider():
    """provider(*replies, **config_overrides) -> FakeProvider wired into llm."""
    def make(*replies, **overrides):
        fake = FakeProvider(replies)
        cfg = dict(provider="groq", model="test-model", base_url="https://llm.test/v1",
                   api_key="k", max_retries=2)
        cfg.update(overrides)
        llm.configure(llm.LLMConfig(**cfg), transport=httpx.MockTransport(fake.handler))
        return fake
    return make


@pytest.fixture
def retriever():
    return FakeRetriever()
