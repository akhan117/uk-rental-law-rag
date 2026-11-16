from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Facts(BaseModel):
    tenancy_type: str | None = None
    location: str | None = None
    tenancy_start_date: str | None = None
    notice_details: str | None = None


class RetrievalFilters(BaseModel):
    """Restrict retrieval by chunk metadata (Docs.pdf §3).

    Jurisdiction isn't offered because this corpus is England-only by
    construction; out-of-jurisdiction questions are handled by the scope gate
    rather than by filtering a corpus that contains nothing else.
    """
    source_type: str | None = None   # "legislation" | "case"
    court: str | None = None         # e.g. "UKSC", "EWCA-Civil"
    act_title: str | None = None     # e.g. "Housing Act 2004"
    date: str | None = None          # judgment date, ISO

    def as_where(self) -> dict | None:
        where = self.model_dump(exclude_none=True)
        return where or None


class QueryRequest(BaseModel):
    # Bounded: on a public demo an unbounded question is an unbounded prompt,
    # paid for out of a free-tier token quota.
    question: str = Field(min_length=1, max_length=2000)
    facts: Facts | None = None
    filters: RetrievalFilters | None = None
    # "rag": single-shot hybrid retrieval. "agent": the model plans retrieval
    # with tools (backend/agent.py); the answer step is identical.
    mode: Literal["rag", "agent"] = "rag"


class AgentStep(BaseModel):
    round: int
    tool: str
    arguments: dict
    summary: str
    latency_ms: int


class ResponseMeta(BaseModel):
    """What serving this answer cost. Also logged per request (telemetry.py)."""
    request_id: str
    mode: str
    provider: str
    model: str
    latency_ms: int
    stages_ms: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    cached: bool = False


class Source(BaseModel):
    chunk_id: str
    citation: str
    url: str
    source_type: str  # "legislation" | "case"
    text: str


class QueryResponse(BaseModel):
    abstained: bool
    short_answer: str
    applicable_law: str
    how_it_applies: str
    uncertainties: str
    sources: list[Source] = Field(default_factory=list)
    scope_note: str | None = None
    # Authorities the model cited, constrained by JSON-schema enum to citations
    # that were actually retrieved (see generate.answer_schema)
    citations: list[str] = Field(default_factory=list)
    # Citation refs the answer made that aren't backed by the retrieved evidence
    unsupported_citations: list[str] = Field(default_factory=list)
    # Agent mode only: the tool calls that gathered the evidence
    agent_steps: list[AgentStep] = Field(default_factory=list)
    meta: ResponseMeta | None = None
