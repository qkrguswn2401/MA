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


def route(question: str) -> str:
    """Classify a question to a backend: ``"dart"`` (public listed company via DART) or
    ``"wiki"`` (internal Centroid valuation KB). LLM call via the guest vLLM (no tools
    needed); defaults to ``"wiki"`` on any parse/endpoint failure."""
    from src.stella_kb.llm import chat

    from .graph.nodes import parse_action
    from .prompts import load as load_prompt

    try:
        raw = chat(
            [{"role": "system", "content": load_prompt("route")},
             {"role": "user", "content": f"Question: {question}\nJSON:"}],
            max_tokens=120, timeout=30.0,
        )
        act = parse_action(raw) or {}
        return "dart" if act.get("source") == "dart" else "wiki"
    except Exception:  # noqa: BLE001 — routing must never hard-fail; fall back to wiki
        return "wiki"


def answer(question: str, source: str = "auto", max_steps: int = 3,
           verbose: bool = False, index: dict | None = None, store: Any = None) -> dict[str, Any]:
    """Unified entry point: route (or honor an explicit ``source``) and dispatch.

    ``source`` is ``"auto"`` (route), ``"wiki"`` (Centroid KB), or ``"dart"`` (public co.).
    ``store`` selects the wiki dataset (ignored by the DART backend, which has no wiki).
    Returns ``{source, answer, trace, steps}`` — same shape for both backends."""
    src = route(question) if source == "auto" else source
    if src == "dart":
        from .dart_agent import run_dart
        return {"source": "dart", **run_dart(question)}
    return {"source": "wiki",
            **run(question, max_steps=max_steps, verbose=verbose, index=index, store=store)}


def _seed(question: str, max_steps: int, verbose: bool = False, store: Any = None) -> AgentState:
    """Initial graph state: INDEX ToC + the question. Each node builds its own prompt.

    ``store`` (a :class:`apps.agent.datasets.WikiStore`) selects the dataset for this run — its
    INDEX.md seeds the planner and its dir threads to the page/ledger reads. ``None`` falls back
    to the process-default wiki (``INDEX_MD`` global / ``tools`` globals)."""
    return {
        "question": question,
        "index_md": store.index_md if store is not None else INDEX_MD.read_text(encoding="utf-8"),
        "wiki_dir": str(store.wiki_dir) if store is not None else None,
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
    # planner → solve (fan-out) → auditor → synthesizer = 4 supersteps; give headroom
    return {"recursion_limit": 25}


def run(question: str, max_steps: int = 3, verbose: bool = False,
        index: dict | None = None, app: Any = None, store: Any = None) -> dict[str, Any]:
    """Navigate the wiki to answer ``question``; return ``{answer, trace, steps, evidence}``.

    ``trace`` is the per-turn routing record (which page it opened, why) — the whole point
    of testing the index as a lookup table. Pass ``verbose=True`` to also print it.
    ``evidence`` is the cell-anchored facts the agent actually retrieved
    (``[{page, cell, term, value, ask}]``) — the *retrieved context*, exposed for RAGAS-style
    evaluation; callers that don't need it can ignore the key.

    ``store`` (a :class:`apps.agent.datasets.WikiStore`) selects the dataset (its index + dir);
    it takes precedence over ``index``. ``app`` lets a caller pass a graph compiled once and
    reused across many questions (the eval builds it per-question otherwise); when given,
    ``index`` is ignored — but pass a matching ``store`` so the seeded dir lines up with it.
    """
    if app is None:
        idx = store.index if store is not None else (index if index is not None else load_index())
        app = build_app(idx)
    final: dict[str, Any] = app.invoke(_seed(question, max_steps, verbose, store), config=_limit())
    return {"answer": (final.get("answer") or "(no answer)").strip(),
            "trace": _renumber(final.get("trace", [])),
            "steps": final.get("steps", 0),
            "evidence": final.get("evidence", [])}


def ask(question: str, max_steps: int = 3, verbose: bool = False,
        index: dict | None = None) -> str:
    """Convenience wrapper around :func:`run` that returns just the answer string."""
    return run(question, max_steps=max_steps, verbose=verbose, index=index)["answer"]


def stream_run(question: str, max_steps: int = 3, index: dict | None = None, store: Any = None):
    """Generator yielding routing events as the agent navigates, for live (SSE) display.

    Uses LangGraph's native ``app.stream(stream_mode="values")``: after every node the
    full state is emitted, so new ``trace`` entries surface as the agent makes each
    decision. Event dicts carry a ``type``:

      {"type": "step",   "step": int, "action": str, "arg": str, "thought": str}
      {"type": "answer", "answer": str, "steps": int}
    """
    idx = store.index if store is not None else (index if index is not None else load_index())
    app = build_app(idx)
    emitted = 0
    final: dict[str, Any] = {}
    for state in app.stream(_seed(question, max_steps, False, store), config=_limit(),
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
