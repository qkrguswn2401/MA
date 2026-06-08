"""Deterministic wiki access + LLM-output parsing — no network."""

from __future__ import annotations

import pytest

from apps.agent.graph.nodes import parse_action
from apps.agent.io import lookup, open_page, trace_links

# --- parse_action: salvage one JSON object from messy model output ---------------------


@pytest.mark.parametrize("raw, expected", [
    ('{"verdict": "ok"}', {"verdict": "ok"}),
    ('```json\n{"a": 1}\n```', {"a": 1}),
    ('생각: 이렇게 하자 {"a": 1, "b": 2} 끝', {"a": 1, "b": 2}),
    ('{"a": {"b": 2}}', {"a": {"b": 2}}),          # nested braces → outermost object
    ('no json here', None),
    ('', None),
    ('{bad json}', None),
])
def test_parse_action(raw, expected):
    assert parse_action(raw) == expected


# --- trace_links: BFS over the sheet-level formula DAG (the provenance hop) ------------

# A → B → {C, D}, with a B↔D cycle; only A and C have pages.
SYNTH = {
    "sheet_dag": {
        "A": {"feeds_into": ["B"], "depends_on": []},
        "B": {"feeds_into": ["C", "D"], "depends_on": ["A", "D"]},
        "C": {"feeds_into": [], "depends_on": ["B"]},
        "D": {"feeds_into": ["B"], "depends_on": ["B"]},
    },
    "pages": {"A": {}, "C": {}},
}


def test_trace_down_collects_chain_and_flags_pages():
    chain = trace_links(SYNTH, "A", "down")
    sheets = [c["sheet"] for c in chain]
    assert sheets[0] == "B" and set(sheets) == {"B", "C", "D"}        # reachable downstream
    has_page = {c["sheet"]: c["has_page"] for c in chain}
    assert has_page == {"B": False, "C": True, "D": False}


def test_trace_up_follows_depends_on():
    sheets = {c["sheet"] for c in trace_links(SYNTH, "C", "up")}
    assert sheets == {"B", "A", "D"}                                  # upstream of C


def test_trace_is_cycle_safe():
    # B↔D must not loop forever; each sheet appears at most once.
    sheets = [c["sheet"] for c in trace_links(SYNTH, "A", "down")]
    assert len(sheets) == len(set(sheets))


def test_trace_respects_depth_and_cap():
    assert [c["sheet"] for c in trace_links(SYNTH, "A", "down", max_depth=1)] == ["B"]
    assert len(trace_links(SYNTH, "A", "down", cap=1)) == 1


def test_trace_unknown_start_is_empty():
    assert trace_links(SYNTH, "ZZZ", "down") == []


# --- against the real built index (skips if not generated) ----------------------------


def test_valuation_chain_reaches_ev(index):
    """성과보수 flows through the engine sheets to the DCF EV exhibit — the project thesis."""
    chain = {c["sheet"]: c for c in trace_links(index, "성과보수, 배당금", "down")}
    assert "Operating Revenue" in chain          # engine intermediary (no page)
    assert chain["DCF 장표 #1_MGT"]["has_page"]   # reachable EV exhibit, openable
    assert not chain["Operating Revenue"]["has_page"]


# --- lookup / open_page ---------------------------------------------------------------


def test_lookup_resolves_known_term(index):
    out = lookup(index, "성과보수")
    assert "성과보수, 배당금" in out and "hit" in out


def test_lookup_miss_is_graceful(index):
    assert "no matching pages" in lookup(index, "절대없는용어xyz")


def test_open_existing_page_has_cells():
    out = open_page("DCF 장표 #1_MGT")
    assert out.startswith("OPEN") and "## " in out


def test_open_missing_page_guides_back():
    assert "no such page" in open_page("Nonexistent Sheet 999")


def test_open_page_trims_aliases_frontmatter():
    # the aliases: line is dropped to save context, sheet/section kept
    out = open_page("성과보수, 배당금")
    assert "aliases:" not in out and "sheet:" in out
