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

import asyncio
from typing import Any

from .graph import AgentState, build_app
from .graph.nodes import synthesize, synthesize_stream
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
           verbose: bool = False, index: dict | None = None, store: Any = None,
           save: bool = False) -> dict[str, Any]:
    """Unified entry point: route (or honor an explicit ``source``) and dispatch.

    ``source`` is ``"auto"`` (route), ``"wiki"`` (Centroid KB), or ``"dart"`` (public co.).
    ``store`` selects the wiki dataset (ignored by the DART backend, which has no wiki).
    ``save=True`` compounds a grounded wiki answer back onto its page (no-op for DART).
    Returns ``{source, answer, trace, steps}`` — same shape for both backends."""
    if source == "auto":
        from .supervisor import run_supervised  # handoff-tool supervisor (wiki + DART as tools)
        return run_supervised(question, store=store)
    if source == "dart":
        from .dart_agent import run_dart
        return {"source": "dart", **run_dart(question)}
    return {"source": "wiki",
            **run(question, max_steps=max_steps, verbose=verbose, index=index, store=store,
                  save=save)}


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
    # planner → solve (fan-out) → auditor = 3 supersteps (synthesis runs after the graph); headroom
    return {"recursion_limit": 25}


def _resolve_index(store: Any, index: dict | None) -> dict:
    """Resolve the wiki index with store > explicit index > process default precedence."""
    if store is not None:
        return store.index
    if index is not None:
        return index
    return load_index()


def _build_result(final: dict[str, Any], answer: str, synth_trace: dict) -> dict[str, Any]:
    """Assemble the four standard result keys: the post-graph ``answer`` and its trace record
    merged into the graph's accumulated trace, plus the merged steps/evidence."""
    return {
        "answer": (answer or "(no answer)").strip(),
        "trace": _renumber(list(final.get("trace", [])) + [synth_trace]),
        "steps": final.get("steps", 0),
        "evidence": final.get("evidence", []),
    }


async def _aiter_in_thread(make_gen):
    """Drive a blocking generator (``make_gen()``) in a worker thread, yielding its items on the
    event loop — so the SSE handler can stream the synchronous ``synthesize_stream`` (urllib) token
    by token without pinning the loop. Exceptions from the generator are re-raised on the loop."""
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()
    done = object()

    def produce():
        try:
            for item in make_gen():
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except Exception as exc:  # noqa: BLE001 — forward to the consumer on the loop
            loop.call_soon_threadsafe(queue.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, done)

    loop.run_in_executor(None, produce)
    while True:
        item = await queue.get()
        if item is done:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def run(question: str, max_steps: int = 3, verbose: bool = False,
        index: dict | None = None, app: Any = None, store: Any = None,
        save: bool = False) -> dict[str, Any]:
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

    ``save=True`` *compounds* the answer back onto its most-cited wiki page (the query→page
    step) when it's grounded — adds a ``saved`` key with the persist result. Off by default:
    only a deliberately-saved query becomes permanent.
    """
    if app is None:
        app = build_app(_resolve_index(store, index))
    final: dict[str, Any] = app.invoke(_seed(question, max_steps, verbose, store), config=_limit())
    answer, synth_trace = synthesize(final)  # graph ends at auditor; write the answer here
    result = _build_result(final, answer, synth_trace)
    if save:
        from .io import persist_answer
        result["saved"] = persist_answer(
            question, result["answer"], result["evidence"],
            wiki_dir=(str(store.wiki_dir) if store is not None else None))
    return result


def ask(question: str, max_steps: int = 3, verbose: bool = False,
        index: dict | None = None) -> str:
    """Convenience wrapper around :func:`run` that returns just the answer string."""
    return run(question, max_steps=max_steps, verbose=verbose, index=index)["answer"]


async def arun(question: str, max_steps: int = 3, verbose: bool = False,
               index: dict | None = None, app: Any = None, store: Any = None) -> dict[str, Any]:
    """Async twin of :func:`run` — drives the graph with ``ainvoke`` so the event loop is never
    blocked (the sync node functions run in LangGraph's executor). Same return shape as ``run``.
    Used by the async API; the sync ``run`` stays for the eval/CLI callers."""
    if app is None:
        app = build_app(_resolve_index(store, index))
    final: dict[str, Any] = await app.ainvoke(_seed(question, max_steps, verbose, store), config=_limit())
    answer, synth_trace = await asyncio.to_thread(synthesize, final)  # blocking call off the loop
    return _build_result(final, answer, synth_trace)


async def aanswer(question: str, source: str = "auto", max_steps: int = 3,
                  verbose: bool = False, index: dict | None = None,
                  store: Any = None) -> dict[str, Any]:
    """Async twin of :func:`answer`: dispatch by ``source``. ``"auto"`` runs the handoff-tool
    supervisor (wiki + DART as tools); explicit ``"wiki"``/``"dart"`` go straight to that
    backend (the sync DART agent runs via ``to_thread`` so it doesn't block the loop)."""
    if source == "auto":
        from .supervisor import arun_supervised  # handoff-tool supervisor (wiki + DART as tools)
        return await arun_supervised(question, store=store)
    if source == "dart":
        from .dart_agent import run_dart
        return {"source": "dart", **(await asyncio.to_thread(run_dart, question))}
    return {"source": "wiki",
            **(await arun(question, max_steps=max_steps, verbose=verbose, index=index, store=store))}


def stream_run(question: str, max_steps: int = 3, index: dict | None = None, store: Any = None):
    """Generator yielding routing events as the agent navigates, for live (SSE) display.

    Uses LangGraph's native ``app.stream(stream_mode="values")`` for the routing steps; the graph
    ends at the auditor, then the answer is streamed token by token via ``synthesize_stream``.
    Event dicts carry a ``type``:

      {"type": "step",   "step": int, "action": str, "arg": str, "thought": str}
      {"type": "token",  "text": str}                 # one per answer fragment, in order
      {"type": "answer", "answer": str, "steps": int} # the joined final answer (last)
    """
    app = build_app(_resolve_index(store, index))
    emitted = 0
    final: dict[str, Any] = {}
    for state in app.stream(_seed(question, max_steps, False, store), config=_limit(),
                            stream_mode="values"):
        final = state
        trace = state.get("trace", [])
        while emitted < len(trace):                       # surface each routing decision
            e = dict(trace[emitted])
            e["step"] = emitted                           # running global step (branches merge)
            yield {"type": "step", **e}
            emitted += 1
    parts: list[str] = []                                 # graph done → stream the answer tokens
    for delta in synthesize_stream(final):
        parts.append(delta)
        yield {"type": "token", "text": delta}
    yield {"type": "answer", "answer": "".join(parts).strip() or "(답변 없음)",
           "steps": final.get("steps", 0)}


async def astream_run(question: str, max_steps: int = 3, index: dict | None = None,
                      store: Any = None, source: str = "wiki"):
    """Async twin of :func:`stream_run` — an async generator over ``app.astream`` so the SSE
    endpoint can stream without pinning a threadpool thread for the connection's lifetime. Emits
    the same ``step``/``token``/``answer`` event dicts; LangGraph runs the sync nodes in its
    executor, and the blocking token stream runs in a worker thread via ``_aiter_in_thread``.

    ``source="auto"`` delegates to the handoff-tool supervisor (``supervisor.astream_supervised``),
    which yields the same event shape; ``"wiki"`` (the default) streams the wiki graph directly."""
    if source == "auto":
        from .supervisor import astream_supervised
        async for ev in astream_supervised(question, store=store):
            yield ev
        return
    app = build_app(_resolve_index(store, index))
    emitted = 0
    final: dict[str, Any] = {}
    async for state in app.astream(_seed(question, max_steps, False, store), config=_limit(),
                                   stream_mode="values"):
        final = state
        trace = state.get("trace", [])
        while emitted < len(trace):
            e = dict(trace[emitted])
            e["step"] = emitted
            yield {"type": "step", **e}
            emitted += 1
    parts: list[str] = []                                 # graph done → stream the answer tokens
    async for delta in _aiter_in_thread(lambda: synthesize_stream(final)):
        parts.append(delta)
        yield {"type": "token", "text": delta}
    yield {"type": "answer", "answer": "".join(parts).strip() or "(답변 없음)",
           "steps": final.get("steps", 0)}
