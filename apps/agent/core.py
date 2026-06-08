"""Public API of the wiki query agent: ``run`` / ``ask`` / ``stream_run``.

Each seeds the multi-agent pipeline (``apps.agent.graph``: planner → router → retriever →
verifier → synthesizer) with the wiki's ``INDEX.md`` table of contents and the question,
then drives it to a cited Korean answer. The router is handed the ToC and must navigate to
the right page on its own, using only the deterministic ``apps.agent.io`` reads.

  - ``run``         → ``{answer, trace, steps}`` (trace = the per-agent routing record)
  - ``ask``         → just the answer string
  - ``stream_run``  → generator of routing events, for live (SSE) display
"""

from __future__ import annotations

from typing import Any

from .graph import AgentState, build_app
from .io import INDEX_MD, load_index


def _seed(question: str, max_steps: int, verbose: bool = False) -> AgentState:
    """Initial graph state: INDEX ToC + the question. Each node builds its own prompt."""
    return {
        "question": question,
        "index_md": INDEX_MD.read_text(encoding="utf-8"),
        "plan": [], "evidence": [], "paths": [], "trace": [], "steps": 0,
        "max_steps": max_steps, "verbose": verbose,
    }


def _renumber(trace: list) -> list:
    """Order the merged trace (planner → branches by sub-question → synthesizer) and assign a
    sequential global ``step``. Parallel branches finish in nondeterministic order, so sort by
    (branch index, intra-branch order) for a stable, readable trace."""
    ordered = sorted(trace, key=lambda e: (e.get("sub", 0), e.get("step", 0)))
    for i, e in enumerate(ordered):
        e["step"] = i
    return ordered


def _limit() -> dict:
    # planner → solve (fan-out, one superstep) → synthesizer = 3 supersteps; give headroom
    return {"recursion_limit": 25}


def run(question: str, max_steps: int = 8, verbose: bool = False,
        index: dict | None = None) -> dict[str, Any]:
    """Navigate the wiki to answer ``question``; return ``{answer, trace, steps}``.

    ``trace`` is the per-turn routing record (which page it opened, why) — the whole point
    of testing the index as a lookup table. Pass ``verbose=True`` to also print it.
    """
    app = build_app(index if index is not None else load_index())
    final: dict[str, Any] = app.invoke(_seed(question, max_steps, verbose), config=_limit())
    return {"answer": (final.get("answer") or "(no answer)").strip(),
            "trace": _renumber(final.get("trace", [])),
            "steps": final.get("steps", 0)}


def ask(question: str, max_steps: int = 8, verbose: bool = False,
        index: dict | None = None) -> str:
    """Convenience wrapper around :func:`run` that returns just the answer string."""
    return run(question, max_steps=max_steps, verbose=verbose, index=index)["answer"]


def stream_run(question: str, max_steps: int = 8, index: dict | None = None):
    """Generator yielding routing events as the agent navigates, for live (SSE) display.

    Uses LangGraph's native ``app.stream(stream_mode="values")``: after every node the
    full state is emitted, so new ``trace`` entries surface as the agent makes each
    decision. Event dicts carry a ``type``:

      {"type": "step",   "step": int, "action": str, "arg": str, "thought": str}
      {"type": "answer", "answer": str, "steps": int}
    """
    app = build_app(index if index is not None else load_index())
    emitted = 0
    final: dict[str, Any] = {}
    for state in app.stream(_seed(question, max_steps), config=_limit(),
                            stream_mode="values"):
        final = state
        trace = state.get("trace", [])
        while emitted < len(trace):                       # surface each new decision
            e = dict(trace[emitted])
            e["step"] = emitted                           # running global step (branches merge)
            yield {"type": "step", **e}
            emitted += 1
    if final.get("answer"):
        yield {"type": "answer", "answer": final["answer"], "steps": final.get("steps", 0)}
