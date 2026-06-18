"""The graph plumbing that makes fan-out safe: reducer channels, trace renumbering, build."""

from __future__ import annotations

import operator
import typing

from apps.agent.core import _renumber
from apps.agent.graph import build_app
from apps.agent.graph.state import AgentState


# --- reducer channels: the parallel branches must MERGE, not overwrite -----------------


def test_accumulator_channels_use_add_reducer():
    hints = typing.get_type_hints(AgentState, include_extras=True)
    for ch in ("evidence", "paths", "trace", "steps"):
        meta = getattr(hints[ch], "__metadata__", None)
        assert meta and meta[0] is operator.add, f"{ch} must carry operator.add"


def test_branch_private_fields_are_plain():
    # sub/sub_idx travel in the Send payload per branch — must NOT be shared reducers
    hints = typing.get_type_hints(AgentState, include_extras=True)
    for ch in ("sub", "sub_idx", "plan", "answer"):
        assert not hasattr(hints[ch], "__metadata__"), f"{ch} should be a plain channel"


# --- _renumber: order the merged trace and assign a sequential global step -------------


def test_renumber_orders_by_branch_then_intra_branch():
    # planner(sub=-1) → branch0(sub=0) → branch1(sub=1) → synthesizer(sub=1e9),
    # deliberately shuffled and with branch-local steps
    merged = [
        {"sub": 1, "step": 1, "agent": "retriever"},
        {"sub": 0, "step": 0, "agent": "router"},
        {"sub": -1, "step": 0, "agent": "planner"},
        {"sub": 1, "step": 0, "agent": "router"},
        {"sub": 10**9, "step": 0, "agent": "synthesizer"},
        {"sub": 0, "step": 1, "agent": "retriever"},
    ]
    out = _renumber(merged)
    assert [e["step"] for e in out] == [0, 1, 2, 3, 4, 5]
    assert [e["agent"] for e in out] == [
        "planner", "router", "retriever", "router", "retriever", "synthesizer"]


def test_renumber_empty():
    assert _renumber([]) == []


# --- the graph compiles from the real index -------------------------------------------


def test_build_app_compiles(index):
    app = build_app(index)
    assert app is not None
    assert "solve" in app.get_graph().nodes  # fan-out node is wired in


# --- curated first-layer deck override (decks.yaml) -----------------------------------
# build_document's upper layer is curated > LLM > default. When BOTH title and description
# are pinned, the LLM call is skipped — so these run offline/deterministically.

_ENTRIES = {
    "FDD1": {"title": "Company Snapshot", "desc": "회사 개요"},
    "FDD2": {"title": "Valuation", "desc": "가치 평가"},
}


def test_curated_overrides_win_and_skip_llm(monkeypatch):
    from src.stella_kb.wiki import pdf_pages

    # if the LLM is reached, fail loudly — a fully-pinned deck must not call it
    monkeypatch.setattr(pdf_pages, "cached_chat",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    node = pdf_pages.build_document(
        "STELLA", _ENTRIES,
        curated={"title": "Curated Title", "description": "Curated description."})
    assert node["title"] == "Curated Title"
    assert node["description"] == "Curated description."
    # lower layer (ToC) is always derived from the pages, regardless of curation
    assert [t["page"] for t in node["toc"]] == ["FDD1", "FDD2"]
    assert node["n_pages"] == 2


def test_partial_curation_still_calls_llm_for_the_missing_field(monkeypatch):
    from src.stella_kb.wiki import pdf_pages

    monkeypatch.setattr(pdf_pages, "cached_chat",
                        lambda *a, **k: '{"title": "LLM Title", "description": "LLM desc"}')
    node = pdf_pages.build_document(
        "STELLA", _ENTRIES, curated={"title": "Pinned Title"})  # description omitted
    assert node["title"] == "Pinned Title"          # curated wins
    assert node["description"] == "LLM desc"          # LLM fills the gap


def test_no_curation_is_unchanged_pure_llm(monkeypatch):
    from src.stella_kb.wiki import pdf_pages

    monkeypatch.setattr(pdf_pages, "cached_chat",
                        lambda *a, **k: '{"title": "LLM Title", "description": "LLM desc"}')
    node = pdf_pages.build_document("STELLA", _ENTRIES)  # curated=None
    assert node["title"] == "LLM Title"
    assert node["description"] == "LLM desc"


def test_load_decks_absent_file_is_empty(monkeypatch, tmp_path):
    from src.stella_kb import config
    from src.stella_kb.wiki import pdf_pages

    monkeypatch.setattr(config, "wiki_decks_yaml", lambda: tmp_path / "nope.yaml")
    assert pdf_pages._load_decks() == {}


def test_load_decks_reads_yaml(monkeypatch, tmp_path):
    from src.stella_kb import config
    from src.stella_kb.wiki import pdf_pages

    f = tmp_path / "decks.yaml"
    f.write_text("CAESAR:\n  title: T\n  description: D\nBOGUS: not-a-dict\n", encoding="utf-8")
    monkeypatch.setattr(config, "wiki_decks_yaml", lambda: f)
    decks = pdf_pages._load_decks()
    assert decks == {"CAESAR": {"title": "T", "description": "D"}}  # non-dict entry dropped


# --- routes.yaml short-circuit: a curated hit skips the router LLM (the latency win) ---

_ROUTE_IDX = {"pages": {"WACC 페이지": {}}, "alias_index": {}}


def test_route_curated_hit_skips_router_llm(monkeypatch):
    from apps.agent.graph import nodes

    monkeypatch.setattr(nodes, "route_lookup", lambda hints, idx, wd: ["WACC 페이지"])
    monkeypatch.setattr(nodes, "_ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("router LLM called")))
    sub = {"ask": "WACC?", "hint_terms": ["WACC"], "mode": "lookup"}
    picks, path, thought = nodes._route(sub, [], _ROUTE_IDX, "INDEX", wiki_dir=None)
    assert picks == ["WACC 페이지"] and "routes.yaml" in thought


def test_route_retry_bypasses_shortcut_and_calls_llm(monkeypatch):
    from apps.agent.graph import nodes

    # even if the table would hit, a retry (tried non-empty) must diverge via the LLM router
    monkeypatch.setattr(nodes, "route_lookup", lambda *a, **k: ["WACC 페이지"])
    monkeypatch.setattr(nodes, "_ask", lambda *a, **k: ({"pages": ["WACC 페이지"]}, "raw"))
    sub = {"ask": "WACC?", "hint_terms": ["WACC"], "mode": "lookup"}
    picks, _, _ = nodes._route(sub, ["WACC 페이지"], _ROUTE_IDX, "INDEX", wiki_dir=None)
    assert picks == ["WACC 페이지"]  # came from the LLM path, not the shortcut


def test_route_miss_falls_back_to_llm(monkeypatch):
    from apps.agent.graph import nodes

    monkeypatch.setattr(nodes, "route_lookup", lambda *a, **k: [])  # no curated hit
    monkeypatch.setattr(nodes, "_ask", lambda *a, **k: ({"pages": ["WACC 페이지"]}, "raw"))
    sub = {"ask": "WACC?", "hint_terms": ["WACC"], "mode": "lookup"}
    picks, _, _ = nodes._route(sub, [], _ROUTE_IDX, "INDEX", wiki_dir=None)
    assert picks == ["WACC 페이지"]


# --- multi-page routing: the router opens up to top_k pages in one round, capped --------

_MULTI_IDX = {"pages": {f"P{i}": {} for i in range(6)}, "alias_index": {}}


def test_router_opens_up_to_top_k_pages(monkeypatch):
    from apps.agent.graph import nodes
    from src.stella_kb import config

    monkeypatch.setattr(config, "agent_router_top_k", lambda: 4)
    monkeypatch.setattr(nodes, "route_lookup", lambda *a, **k: [])
    # LLM returns 6 valid pages — must be capped to top_k=4, order preserved
    monkeypatch.setattr(nodes, "_ask",
                        lambda *a, **k: ({"pages": [f"P{i}" for i in range(6)]}, "raw"))
    sub = {"ask": "여러 페이지에 흩어진 값", "hint_terms": [], "mode": "lookup"}
    picks, _, _ = nodes._route(sub, [], _MULTI_IDX, "INDEX", wiki_dir=None)
    assert picks == ["P0", "P1", "P2", "P3"]


def test_router_cap_also_bounds_curated_routes(monkeypatch):
    from apps.agent.graph import nodes
    from src.stella_kb import config

    monkeypatch.setattr(config, "agent_router_top_k", lambda: 2)
    monkeypatch.setattr(nodes, "route_lookup", lambda *a, **k: ["P0", "P1", "P2"])
    monkeypatch.setattr(nodes, "_ask",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("LLM called")))
    sub = {"ask": "x", "hint_terms": ["t"], "mode": "lookup"}
    picks, _, _ = nodes._route(sub, [], _MULTI_IDX, "INDEX", wiki_dir=None)
    assert picks == ["P0", "P1"]  # curated list capped too


# --- committed curation: version-token → curation/<version>/{decks,routes}.yaml --------

def test_version_token_handles_data_and_wiki_dirs():
    from src.stella_kb.config import _version_token

    assert _version_token("data/v0.2") == "v0.2"
    assert _version_token("data/v0.2/wiki") == "v0.2"        # leaf 'wiki' → parent


def test_curation_paths_default_into_committed_tree(monkeypatch):
    from src.stella_kb import config

    monkeypatch.delenv("MNA_WIKI_DECKS", raising=False)
    monkeypatch.delenv("MNA_AGENT_ROUTES", raising=False)
    monkeypatch.setattr(config, "curation_dir", lambda: __import__("pathlib").Path("curation"))
    monkeypatch.setattr(config, "wiki_data_dir", lambda: __import__("pathlib").Path("data/v0.2"))
    assert config.wiki_decks_yaml().as_posix() == "curation/v0.2/decks.yaml"
    assert config.agent_routes_yaml("data/v0.2/wiki").as_posix() == "curation/v0.2/routes.yaml"
