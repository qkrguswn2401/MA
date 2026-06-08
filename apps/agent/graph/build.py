"""Wire the fan-out pipeline into a compiled LangGraph ``StateGraph``.

    START → planner ─┬─Send→ solve ─┐
                     ├─Send→ solve ─┼─→ synthesizer → END
                     └─Send→ solve ─┘

The planner emits N sub-questions; ``_fanout`` dispatches one ``solve`` branch per
sub-question with the ``Send`` API, and they run concurrently. ``synthesizer`` is downstream
of ``solve``, so LangGraph runs it once, after every branch has merged its evidence/paths/
trace into the shared ``operator.add`` channels. ``index`` is closed over by ``solve`` (it
needs the alias index, page whitelist, and provenance DAG).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .nodes import planner_node, solve_node, synthesizer_node
from .state import AgentState


def _fanout(state: AgentState):
    """One ``solve`` branch per sub-question; the payload carries the branch's private work."""
    return [
        Send("solve", {"sub": p, "sub_idx": i, "index_md": state["index_md"],
                       "max_steps": state.get("max_steps", 8), "verbose": state.get("verbose")})
        for i, p in enumerate(state["plan"])
    ]


def build_app(index: dict):
    """Compile the fan-out graph; ``index`` is closed over by the solve branches."""
    g = StateGraph(AgentState)
    g.add_node("planner", planner_node)
    g.add_node("solve", lambda s: solve_node(s, index))
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", _fanout, ["solve"])
    g.add_edge("solve", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()
