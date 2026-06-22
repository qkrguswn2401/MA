"""Deterministic wiki access + LLM-output parsing — no network."""

from __future__ import annotations

import pytest

from apps.agent.backends.wiki.nodes import parse_action
from apps.agent.retrieval import (
    cross_ref_partners,
    extract_page_items,
    lookup,
    open_page,
    route_lookup,
    trace_links,
)


def test_cross_ref_partners_directional():
    idx = {"pages": {
        "FDD1": {"source": "PDF", "derives_from": [
            {"page": "E1", "via": "x"}, {"page": "E2", "via": "y"}, {"page": "E3", "via": "z"}]},
        "E1": {"cited_by": ["FDD1", "FDD9"]},
    }}
    assert cross_ref_partners(idx, "FDD1", cap=2) == ["E1", "E2"]   # PDF → derives_from (capped)
    assert cross_ref_partners(idx, "E1") == ["FDD1", "FDD9"]        # Excel → cited_by
    assert cross_ref_partners(idx, "missing") == []

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


def test_trace_real_index_is_raw_only(index):
    """The provenance DAG is built from `_raw`, so a real trace stays within `_raw` sheets.

    (Tracing the carry → engine → DCF EV valuation chain is the *graph* paradigm's job — it
    needs the full workbook the wiki deliberately excludes.)
    """
    chain = {c["sheet"]: c for c in trace_links(index, "제2호_거래내역", "down")}
    assert chain["제2호_비용"]["has_page"]        # ledger → cost: both `_raw` pages, openable
    assert "성과보수, 배당금" not in chain         # carry sheet is out of the `_raw` wiki


# --- extract_page_items: deterministic "value [cell]" table parse (retriever bypass) ---

_EXCEL_MD = """---
sheet: 제2호_비용
---
# Operating Expenses

## 운영비용

| Item | KO | role | cell | 2017 | 2018 |
|---|---|---|---|---|---|
| GP관리보수 | GP관리보수 | expense | `C6` | 46328767 [D6] | 890000000 [E6] |
| 합계 | 합계 | expense | `C13` | 51235594 [D13] | 1085403492 [E13] |
"""

_FDD_MD = """# FDD8 — WACC

## Key figures

| 항목 | 기간 | value |
|---|---|---|
| Risk Free Rate |  | 3.70% [FDD8] |
| Cost of Equity |  | 14.6% [FDD8] |
"""


def test_extract_excel_table_values_periods_cells():
    rows = extract_page_items(_EXCEL_MD)
    gp = [r for r in rows if r["term"] == "GP관리보수"]
    assert {"term": "GP관리보수", "period": "2017", "value": "46328767", "cell": "D6"} in gp
    assert {"term": "GP관리보수", "period": "2018", "value": "890000000", "cell": "E6"} in gp
    # the `cell` column (`C6`) is not a value cell → not emitted; only [ref]-tagged values are
    assert all(r["cell"] in ("D6", "E6", "D13", "E13") for r in rows)


def test_extract_fdd_value_table():
    rows = extract_page_items(_FDD_MD)
    assert {"term": "Risk Free Rate", "period": "", "value": "3.70%", "cell": "FDD8"} in rows
    assert {"term": "Cost of Equity", "period": "", "value": "14.6%", "cell": "FDD8"} in rows


def test_extract_hint_filter_keeps_only_matching_labels():
    rows = extract_page_items(_EXCEL_MD, hint_terms=["GP관리보수"])
    assert rows and all(r["term"] == "GP관리보수" for r in rows)


def test_extract_prose_only_page_is_empty():
    assert extract_page_items("# Title\n\n본문 텍스트, 표 없음.\n") == []


def test_deterministic_retrieve_flag_defaults_off():
    from src.stella_kb.config import agent_deterministic_retrieve
    assert agent_deterministic_retrieve() is False


# --- route_lookup: curated routes.yaml → pages (deterministic, no LLM) -----------------
# The routes table is resolved by config.agent_routes_yaml; point it at a tmp file via the
# MNA_AGENT_ROUTES env override so these stay decoupled from the data/<version>/ layout.

_IDX = {"pages": {"FDD8 — [CAESAR] WACC": {}, "회사 조직도": {}, "엉뚱페이지": {}}}


def _routes_env(tmp_path, monkeypatch, body):
    f = tmp_path / "routes.yaml"
    f.write_text(body, encoding="utf-8")
    monkeypatch.setenv("MNA_AGENT_ROUTES", str(f))


def test_route_lookup_hits_and_validates(tmp_path, monkeypatch):
    _routes_env(tmp_path, monkeypatch,
        "WACC: FDD8 — [CAESAR] WACC\n조직도: 회사 조직도\nSTALE: 존재하지않는페이지\n")
    # normalized key match (whitespace/case-insensitive), order-preserving, deduped
    assert route_lookup(["wacc"], _IDX) == ["FDD8 — [CAESAR] WACC"]
    assert route_lookup([" 조직도 "], _IDX) == ["회사 조직도"]
    # a curated target absent from the index is dropped → empty → caller uses the LLM router
    assert route_lookup(["STALE"], _IDX) == []


def test_route_lookup_no_file_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("MNA_AGENT_ROUTES", str(tmp_path / "nope.yaml"))
    assert route_lookup(["WACC"], _IDX) == []


def test_route_lookup_dedups_across_terms(tmp_path, monkeypatch):
    _routes_env(tmp_path, monkeypatch, "a: 회사 조직도\nb: 회사 조직도\n")
    assert route_lookup(["a", "b"], _IDX) == ["회사 조직도"]  # one page, not two


# --- every committed routes.yaml target must exist in its index (catches typos / the YAML
#     '#'-comment footgun) — skips a dataset whose wiki isn't built in this checkout ----------

@pytest.mark.parametrize("version", ["v0.1", "v0.2"])
def test_committed_routes_targets_exist(version):
    import json

    from src.stella_kb import ROOT

    routes_path = ROOT / "data" / version / "routes.yaml"
    index_path = ROOT / "data" / version / "wiki" / "index.json"
    if not routes_path.exists() or not index_path.exists():
        pytest.skip(f"{version}: routes or built index absent")
    import yaml

    raw = yaml.safe_load(routes_path.read_text(encoding="utf-8")) or {}
    pages = json.loads(index_path.read_text(encoding="utf-8"))["pages"]
    dangling = [(k, p) for k, v in raw.items()
                for p in (v if isinstance(v, list) else [v]) if p not in pages]
    assert not dangling, f"{version}: routes point at non-existent pages: {dangling}"


# --- lookup / open_page ---------------------------------------------------------------


def test_lookup_resolves_known_term(index):
    # assert on an EXACT-match alias (관리보수율) of the page, not a substring term against the
    # whole corpus — the bare "관리보수" ranks 제2호_관리보수 past the 12-row window once FDD pages
    # join the alias index, which tests display truncation, not resolution.
    out = lookup(index, "관리보수율")
    assert "제2호_관리보수" in out and "hit" in out


def test_lookup_miss_is_graceful(index):
    assert "no matching pages" in lookup(index, "절대없는용어xyz")


def test_open_existing_page_has_cells():
    out = open_page("DCF 장표 #1_MGT")
    assert out.startswith("OPEN") and "## " in out


def test_open_missing_page_guides_back():
    assert "no such page" in open_page("Nonexistent Sheet 999")


def test_open_page_trims_aliases_frontmatter():
    # the aliases: line is dropped to save context, sheet/section kept
    out = open_page("제2호_관리보수")
    assert "aliases:" not in out and "sheet:" in out
