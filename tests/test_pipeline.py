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
