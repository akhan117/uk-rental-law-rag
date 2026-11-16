"""Agent tools: Act resolution, exact provision lookup, repeal awareness, search."""
from __future__ import annotations

import pytest

import tools
from conftest import CHUNKS, MANIFEST, FakeRetriever


@pytest.fixture
def corpus():
    return tools.Corpus(CHUNKS, manifest=MANIFEST)


@pytest.mark.parametrize("name,slug", [
    ("Housing Act 1988", "housing-act-1988"),
    ("housing act 2004", "housing-act-2004"),
    ("HA 1988", "housing-act-1988"),
    ("LTA 1985", "landlord-and-tenant-act-1985"),
    ("Landlord & Tenant Act 1985", "landlord-and-tenant-act-1985"),
    ("the Landlord and Tenant Act", "landlord-and-tenant-act-1985"),
])
def test_resolve_act(corpus, name, slug):
    assert corpus.resolve_act(name) == slug


@pytest.mark.parametrize("name", ["Housing Act 1996", "Tenant Fees Act 2019", "", "banana"])
def test_resolve_act_refuses_what_is_not_there(corpus, name):
    assert corpus.resolve_act(name) is None


def test_lookup_exact_subsection(corpus):
    r = corpus.lookup("Housing Act 1988", "s.13(2)")
    assert r.exact and [h["chunk_id"] for h in r.hits] == ["housing-act-1988:section-13-2"]
    assert r.hits[0]["distance"] == 0.0
    assert "two months" in r.content


def test_lookup_whole_section_returns_all_subsections(corpus):
    r = corpus.lookup("HA 1988", "section 13")
    assert {h["chunk_id"] for h in r.hits} == {"housing-act-1988:section-13-1",
                                               "housing-act-1988:section-13-2"}


def test_lookup_ground(corpus):
    r = corpus.lookup("Housing Act 1988", "Schedule 2 Ground 8")
    assert r.hits[0]["chunk_id"].endswith("ground-8")


def test_lookup_repealed_provision_says_so(corpus):
    r = corpus.lookup("Housing Act 1988", "section 21")
    assert not r.hits and r.repealed == MANIFEST
    assert "NO LONGER IN FORCE" in r.content


def test_lookup_missing_provision_lists_what_exists(corpus):
    r = corpus.lookup("Housing Act 1988", "s.999")
    assert not r.hits and "not in this corpus" in r.content
    assert "s.13" in r.content and "Ground 8" in r.content


def test_lookup_unknown_act_lists_acts(corpus):
    r = corpus.lookup("Rent Act 1977", "s.1")
    assert "Available Acts" in r.content and "Housing Act 2004" in r.content


def test_lookup_unparseable_provision(corpus):
    assert "Could not parse" in corpus.lookup("Housing Act 1988", "the rent bit").content


def test_search_applies_filters(corpus):
    retriever = FakeRetriever()
    r = tools.search(retriever, corpus, "deposit", source_type="legislation", act="HA 2004")
    assert retriever.queries[-1][1] == {"source_type": "legislation",
                                        "act_title": "Housing Act 2004"}
    assert [h["chunk_id"] for h in r.hits] == ["housing-act-2004:section-213-3"]


def test_search_with_unknown_act_falls_back_to_whole_corpus(corpus):
    retriever = FakeRetriever()
    r = tools.search(retriever, corpus, "rent increase", act="Made Up Act 2001")
    assert retriever.queries[-1][1] is None and "searched the whole corpus" in r.content


def test_list_sources(corpus):
    r = corpus.list_sources()
    assert "Housing Act 1988" in r.content and "Cases: none" in r.content


def test_execute_never_raises(corpus):
    class Broken:
        def search(self, *a, **k):
            raise RuntimeError("index offline")
    assert "Tool error" in tools.execute("search_law", {"query": "x"}, Broken(), corpus).content
    assert "Unknown tool" in tools.execute("delete_db", {}, Broken(), corpus).content


def test_tool_schemas_are_well_formed():
    names = {t["function"]["name"] for t in tools.TOOL_SCHEMAS}
    assert names == {"search_law", "get_provision", "list_sources"}
    for t in tools.TOOL_SCHEMAS:
        assert t["function"]["parameters"]["type"] == "object"
