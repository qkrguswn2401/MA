"""Graph query layer — the multi-hop fan-out, tested deterministically.

The LLM is monkeypatched (these run offline): we assert the whitelist guard + parsing in
``resolve_metrics`` and that ``ask`` gathers evidence for *every* resolved metric. The live
end-to-end synthesis is exercised by the ``--run-llm`` smoke, not here.
"""

from __future__ import annotations

import networkx as nx

from src.stella_kb import llm
from src.stella_kb.graph import query


# --- resolve_metrics: parse a JSON array, guard each id against the whitelist -----------


def test_resolve_metrics_guards_dedups_and_caps(monkeypatch):
    # off-whitelist ids dropped, duplicates collapsed, order preserved, cap honoured
    monkeypatch.setattr(llm, "chat",
                        lambda *a, **k: '["equity_value", "equity_value", "not_real", "wacc"]')
    assert llm.resolve_metrics("q") == ["equity_value", "wacc"]
    assert llm.resolve_metrics("q", max_metrics=1) == ["equity_value"]


def test_resolve_metrics_tolerates_fences_and_dict_items(monkeypatch):
    monkeypatch.setattr(llm, "chat",
                        lambda *a, **k: '```json\n[{"id": "ebitda"}, {"id": "fcff"}]\n```')
    assert llm.resolve_metrics("q") == ["ebitda", "fcff"]


def test_resolve_metrics_bad_output_is_empty(monkeypatch):
    monkeypatch.setattr(llm, "chat", lambda *a, **k: "no array here")
    assert llm.resolve_metrics("q") == []


# --- ask: fan out evidence over every resolved metric ----------------------------------


def _toy_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    g.add_node("Metric:alpha", type="Metric", label="Alpha", category="revenue", value=10)
    g.add_node("Metric:beta", type="Metric", label="Beta", category="expense", value=20)
    return g


def test_ask_gathers_evidence_for_all_focal_metrics(monkeypatch):
    monkeypatch.setattr(llm, "resolve_metrics", lambda q, m=4: ["alpha", "beta"])
    ev = query.ask("compare alpha and beta", synthesize=False, g=_toy_graph())
    assert "Alpha" in ev and "Beta" in ev          # both metrics' evidence present
    assert "Value: 10" in ev and "Value: 20" in ev


def test_ask_falls_back_to_single_resolver(monkeypatch):
    # fan-out finds nothing -> single-metric resolver still answers (no regression)
    monkeypatch.setattr(llm, "resolve_metrics", lambda q, m=4: [])
    monkeypatch.setattr(query, "resolve", lambda q: "Metric:alpha")
    ev = query.ask("just alpha", synthesize=False, g=_toy_graph())
    assert "Alpha" in ev and "Beta" not in ev


def test_ask_unresolved_is_graceful(monkeypatch):
    monkeypatch.setattr(llm, "resolve_metrics", lambda q, m=4: [])
    monkeypatch.setattr(query, "resolve", lambda q: None)
    assert "Could not resolve" in query.ask("???", g=_toy_graph())
