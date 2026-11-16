"""Scope gate: both checks, and the over-gating guard."""
from __future__ import annotations

import pytest

from scope import GEOGRAPHY_NOTE, keyword_scope_check, retrieval_confidence_check


@pytest.mark.parametrize("question", [
    "How much notice does my landlord need to give in Scotland?",
    "What are my rights under the Renting Homes (Wales) Act?",
    "What's the process for ending a commercial lease early?",
    "I run a business tenancy from a shop unit, what notice do I need?",
])
def test_other_jurisdictions_and_commercial_are_out_of_scope(question):
    assert keyword_scope_check(question)


def test_intra_england_comparison_gets_geography_note():
    assert keyword_scope_check("main leasing law differences between London and Manchester") \
        == GEOGRAPHY_NOTE


@pytest.mark.parametrize("question", [
    "My landlord in London hasn't protected my deposit, what can I do?",
    "What's the difference between Ground 8 and Ground 10 possession?",
    "landlord wont fix boiler been 3 weeks wot can i do",
])
def test_ordinary_questions_are_not_keyword_gated(question):
    assert keyword_scope_check(question) is None


def test_confidence_needs_a_cluster_not_one_close_hit():
    assert retrieval_confidence_check([{"distance": 0.1}, {"distance": 0.5}])
    assert retrieval_confidence_check([{"distance": 0.1}, {"distance": 0.29}]) is None
    # Lexical-only hits carry the 1.0 sentinel and never count.
    assert retrieval_confidence_check([{"distance": 1.0}] * 6)
