from __future__ import annotations

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
    question: str
    facts: Facts | None = None
    filters: RetrievalFilters | None = None


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
