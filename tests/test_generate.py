"""Answer generation: schema, prompt bounds, and guarantees that hold whatever
the provider does with the schema."""
from __future__ import annotations

from conftest import ANSWER, CHUNKS, json_reply

import generate


def _hits():
    return [dict(c, distance=0.2) for c in CHUNKS[:3]]


def test_answer_schema_constrains_citations_to_retrieved():
    schema = generate.answer_schema(["Housing Act 1988, s.13(2)", "", "Housing Act 1988, s.13(2)"])
    assert schema["properties"]["citations"]["items"] == {
        "type": "string", "enum": ["Housing Act 1988, s.13(2)"]}
    assert "citations" in schema["required"]
    assert "citations" not in generate.answer_schema([])["properties"]


def test_prompt_labels_part_title_and_lists_allowed_citations():
    prompt = generate.build_user_prompt("q?", None, _hits())
    assert "[Housing Act 1988, Sch.2 Ground 8 - Grounds on which Court must order possession]" in prompt
    assert "The ONLY citations you may use" in prompt


def test_long_excerpts_are_trimmed():
    hit = dict(CHUNKS[0], text="Heading\n\n" + "word. " * 1000, distance=0.1)
    prompt = generate.build_user_prompt("q?", None, [hit])
    assert "[... excerpt truncated ...]" in prompt
    assert len(prompt) < 3000


def test_unretrieved_citation_is_dropped_even_without_enum_enforcement(provider):
    # Simulates a provider that accepted the schema but didn't enforce the enum.
    provider(json_reply(dict(ANSWER, citations=["Housing Act 1988, s.13(2)",
                                                "Invented Act 2099, s.1"])))
    out = generate.generate_answer("q?", None, _hits())
    assert out["citations"] == ["Housing Act 1988, s.13(2)"]


def test_blank_answer_becomes_honest_fallback(provider):
    provider(json_reply({"case_name": "Superstrike", "paragraphs": "..."}))
    out = generate.generate_answer("q?", None, _hits())
    assert "did not return a usable answer" in out["short_answer"]
    assert out["citations"] == []


def test_list_valued_field_is_flattened(provider):
    provider(json_reply(dict(ANSWER, applicable_law=["Housing Act 1988, s.13(2)", "two months"])))
    out = generate.generate_answer("q?", None, _hits())
    assert out["applicable_law"] == "Housing Act 1988, s.13(2)\ntwo months"


def test_generation_records_usage(provider):
    from telemetry import Trace
    provider(json_reply(ANSWER, usage=(900, 120)))
    trace = Trace()
    generate.generate_answer("q?", None, _hits(), trace=trace)
    assert (trace.tokens_in, trace.tokens_out) == (900, 120)
