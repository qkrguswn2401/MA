"""Planner node — break the question into a minimal list of sub-questions, each of which fans
out to its own ``solve`` branch (LangGraph ``Send``)."""

from __future__ import annotations

from . import engine
from .engine import PLANNER, _rec
from .state import AgentState


def planner_node(state: AgentState) -> AgentState:
    """Break the question into a minimal list of sub-questions (each fans out to a branch)."""
    user = f"INDEX:\n{state['index_md']}\n\nQuestion: {state['question']}\n\nReturn the plan JSON."
    act, _ = engine._ask(PLANNER, user, 400)
    plan = [p for p in ((act or {}).get("plan") or []) if isinstance(p, dict) and p.get("ask")]
    if not plan:  # parse miss / empty → fall back to a single pass-through sub-question
        plan = [{"ask": state["question"], "hint_terms": []}]
    for p in plan:  # normalize the routing controls so the solve branch can rely on them
        p["mode"] = "trace" if p.get("mode") == "trace" else "lookup"
        p["direction"] = "up" if p.get("direction") == "up" else "down"
    if state.get("verbose"):
        print(f"[planner] {len(plan)} sub-question(s) → fan out")
    return {
        "plan": plan,
        "trace": [_rec(-1, 0, "planner", "plan", f"{len(plan)} sub-Q", (act or {}).get("thought", ""))],
    }
