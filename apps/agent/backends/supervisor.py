"""Supervisor as a real LangGraph ``StateGraph`` — a supervisor node routing to wiki/dart
worker nodes via ``Command(goto=…)``, the workers handing control back.

Replaces the earlier ``create_agent``-as-tools, two-phase implementation. The graph is::

    START → supervisor ─Command(goto)→ wiki ─┐
                  ↑                          │ Command(goto="supervisor")
                  ├──────────────────────────┘
                  └─Command(goto)→ dart ─────┘
                  │
                  └─Command(goto="compose") → END

Why a graph (not ``create_agent`` with handoff tools):
  - **No two-phase hack.** Each node is explicit, so the final answer is just the ``compose``
    node's output — no need to separate "final" tokens from "tool-announcing" tokens in a
    message stream, and no per-delta channel-token surgery.
  - **One answer for both paths.** Buffered (``ainvoke``) and streaming (``astream``) end at
    the same ``compose`` node, so they can't diverge.
  - **Provenance preserved.** When only one worker ran, ``compose`` returns that worker's
    cited answer **verbatim** (passthrough) instead of re-prosing it (which dropped cell
    refs). Only a genuine ≥2-source merge gets an LLM compose pass.
  - **Re-routing is first-class.** The supervisor node sees a worker's result and can call
    the other worker (composite question) or finish — real orchestration, not one shot.

The supervisor *decides* with a plain JSON completion (stdlib :func:`chat` + ``parse_action``,
same machinery as the old ``core.route``), not tool-calling — more robust on the gemma-4 we
serve, and routing needs no tools. Only the rare composite-merge ``compose`` calls the LLM
again. Per-request :class:`apps.agent.datasets.WikiStore` (``store``) is closed over by the
wiki node, so concurrency safety is preserved. On any failure the whole thing degrades to
``core.route`` + direct dispatch, so a flaky round never hard-fails a request.

Config (env), shared with ``dart``:
    STELLA_TOOL_LLM_URL / STELLA_TOOL_LLM_MODEL   (unused here — kept for the dart worker)
"""

from __future__ import annotations

import asyncio
import operator
from typing import Annotated, Any, TypedDict

from src.stella_kb.llm import chat

from .dart import _clean
from .wiki.nodes import parse_action
from ..prompts import load as load_prompt

_MAX_TURNS = 4   # supervisor visits before we force a finish (wiki + dart + slack)


# --- state ------------------------------------------------------------------------------


def _merge(a: dict, b: dict) -> dict:
    """Reducer for the ``answers`` channel: later worker writes win on a key collision (none
    expected — each worker writes its own key)."""
    return {**(a or {}), **(b or {})}


class SupervisorState(TypedDict, total=False):
    """State threaded through the supervisor graph. The ``operator.add`` / ``_merge`` channels
    accumulate across the supervisor↔worker loop; ``turns``/``next_query`` are last-write."""
    question: str                              # the user question
    next_query: str                            # sub-question the supervisor sent to the routed worker
    answers: Annotated[dict, _merge]           # {source: answer_text} gathered from workers
    called: Annotated[list, operator.add]      # worker names already run (for the decision + guard)
    evidence: Annotated[list, operator.add]    # wiki worker's cell-anchored facts (RAGAS contexts)
    trace: Annotated[list, operator.add]       # step records [{agent, action, arg, thought}]
    turns: int                                 # supervisor invocation count (loop guard)
    answer: str                                # final answer (compose node)
    source: str                                # final source label: wiki | dart | dart+wiki | none


# --- trace helpers ----------------------------------------------------------------------


def _tag(trace: list[dict], source: str) -> list[dict]:
    """Namespace a worker's trace entries (``planner`` → ``"wiki:planner"`` etc.), order kept."""
    out = []
    for e in trace:
        e = dict(e)
        e["agent"] = f"{source}:{e.get('agent', '')}".rstrip(":")
        out.append(e)
    return out


def _renumber(trace: list[dict]) -> list[dict]:
    """Assign a sequential global ``step`` over the merged (execution-ordered) trace."""
    out = []
    for i, e in enumerate(trace):
        e = dict(e)
        e["step"] = i
        out.append(e)
    return out


def _count_calls(trace: list[dict]) -> int:
    """How many worker dispatches the supervisor made (the 'work' metric)."""
    return sum(1 for e in trace if e.get("agent") == "supervisor" and e.get("action") == "call")


def _source(answers: dict) -> str:
    """Which backend(s) actually answered: ``"wiki"`` | ``"dart"`` | ``"dart+wiki"``."""
    return "+".join(sorted(answers)) if answers else "supervisor"


# --- the decision (supervisor node's brain) ---------------------------------------------


_DECISION_DIRECTIVE = (
    "이제 다음 행동 하나를 골라 JSON 객체 하나로만 출력하세요.\n"
    '- 자료가 더 필요하면 호출할 도구를 고르세요: next는 "wiki" 또는 "dart".\n'
    '- 더 호출할 도구가 없거나 충분히 모았으면: next는 "FINISH".\n'
    "- query에는 그 도구에 보낼 구체적 한국어 질의를 적으세요(FINISH면 빈 문자열).\n"
    "- 이미 호출한 도구는 다시 부르지 마세요.\n"
    '형식: {"next": "wiki"|"dart"|"FINISH", "query": "<질의>", "thought": "<짧은 이유>"}'
)


def _decide(question: str, called: list[str], answers: dict) -> dict:
    """One supervisor decision: which worker to call next (with a tailored sub-query), or
    FINISH. Plain JSON completion — no tool-calling. Defaults to FINISH on a parse failure."""
    avail = [w for w in ("wiki", "dart") if w not in called]
    system = load_prompt("supervisor") + "\n\n" + _DECISION_DIRECTIVE
    user = (f"질문: {question}\n"
            f"아직 호출하지 않은 도구: {avail or '없음'}\n"
            f"이미 수집한 자료: {list(answers) or '없음'}\n"
            "다음 행동을 JSON으로 출력하세요.")
    try:
        raw = chat([{"role": "system", "content": system},
                    {"role": "user", "content": user}], max_tokens=200, timeout=60.0)
        act = parse_action(raw) or {}
    except Exception:  # noqa: BLE001 — a decision must never hard-fail; finish with what we have
        act = {}
    nxt = (act.get("next") or "FINISH").strip()
    return {"next": nxt, "query": (act.get("query") or question).strip(),
            "thought": act.get("thought", "")}


# --- nodes ------------------------------------------------------------------------------


async def _supervisor_node(state: SupervisorState):
    """Decide the next hop. Routes to a worker, or to ``compose`` when done / capped. If it
    would finish having gathered nothing, it grounds via the wiki instead of finishing empty."""
    from langgraph.types import Command

    turns = state.get("turns", 0) + 1
    called = state.get("called", [])
    answers = state.get("answers", {})
    d = await asyncio.to_thread(_decide, state["question"], called, answers)
    nxt, query, thought = d["next"], d["query"], d["thought"]

    capped = turns > _MAX_TURNS
    valid = nxt in ("wiki", "dart") and nxt not in called
    if not capped and valid:
        rec = {"agent": "supervisor", "action": "call",
               "arg": f"{nxt}({query[:80]})", "thought": thought}
        return Command(goto=nxt, update={"next_query": query, "turns": turns, "trace": [rec]})

    # finishing (FINISH, capped, or a repeat/invalid pick) — but never finish empty-handed
    if not answers and "wiki" not in called:
        rec = {"agent": "supervisor", "action": "call", "arg": "wiki(grounding)", "thought": thought}
        return Command(goto="wiki",
                       update={"next_query": state["question"], "turns": turns, "trace": [rec]})
    rec = {"agent": "supervisor", "action": "route", "arg": "compose", "thought": thought}
    return Command(goto="compose", update={"turns": turns, "trace": [rec]})


def _make_wiki_node(store: Any):
    """Wiki worker node, closing over the per-request ``store`` (concurrency-safe)."""
    async def _wiki_node(state: SupervisorState):
        from langgraph.types import Command

        from ..core import arun  # deferred: avoid a core <-> supervisor import cycle
        q = state.get("next_query") or state["question"]
        out = await arun(q, store=store)
        ans = out.get("answer", "")
        trace = _tag(out.get("trace", []), "wiki")
        trace.append({"agent": "supervisor", "action": "result",
                      "arg": f"wiki: {ans[:120]}", "thought": ""})
        return Command(goto="supervisor",
                       update={"answers": {"wiki": ans}, "called": ["wiki"], "trace": trace,
                               "evidence": out.get("evidence", [])})
    return _wiki_node


async def _dart_node(state: SupervisorState):
    """DART worker node — the public-company tool-calling agent."""
    from langgraph.types import Command

    from .dart import _arun as dart_arun
    q = state.get("next_query") or state["question"]
    out = await dart_arun(q)
    ans = out.get("answer", "")
    trace = _tag(out.get("trace", []), "dart")
    trace.append({"agent": "supervisor", "action": "result",
                  "arg": f"dart: {ans[:120]}", "thought": ""})
    return Command(goto="supervisor",
                   update={"answers": {"dart": ans}, "called": ["dart"], "trace": trace})


def _compose_msgs(question: str, answers: dict) -> list[dict]:
    """Compose prompt for the composite (≥2-source) merge: write the final Korean answer over
    the gathered worker outputs only (no fresh retrieval)."""
    blocks = []
    if answers.get("wiki"):
        blocks.append(f"[센트로이드 위키 자료]\n{answers['wiki']}")
    if answers.get("dart"):
        blocks.append(f"[DART 자료]\n{answers['dart']}")
    gathered = "\n\n".join(blocks) or "(수집된 자료 없음)"
    system = (load_prompt("supervisor")
              + "\n\n이제 아래 수집 자료만 근거로 한국어 최종 답변을 작성하세요. 자료에 없는 내용은 지어내지 마세요.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"질문: {question}\n\n{gathered}"}]


def _compose_text(question: str, answers: dict) -> str:
    """LLM merge of multiple sources (buffered, stdlib chat)."""
    raw = chat(_compose_msgs(question, answers), max_tokens=900, timeout=120.0)
    return _clean(raw)


async def _compose_node(state: SupervisorState) -> dict:
    """Terminal node: produce the final answer. **Single source → passthrough** the worker's
    answer verbatim (keeps its cell citations). **≥2 sources → LLM merge.** No worker ran →
    empty/``none`` (the caller grounds via the wiki)."""
    answers = state.get("answers", {})
    if not answers:
        return {"answer": "", "source": "none"}
    if len(answers) == 1:
        src, ans = next(iter(answers.items()))
        return {"answer": ans, "source": src,
                "trace": [{"agent": "supervisor", "action": "passthrough", "arg": src, "thought": ""}]}
    final = await asyncio.to_thread(_compose_text, state["question"], answers)
    return {"answer": final, "source": _source(answers),
            "trace": [{"agent": "supervisor", "action": "answer", "arg": "", "thought": ""}]}


def _build_supervisor(store: Any):
    """Compile the supervisor graph; the wiki node closes over the per-request ``store``.
    Factored out so tests can drive a real graph with stubbed nodes/LLM offline."""
    from langgraph.graph import END, START, StateGraph

    g = StateGraph(SupervisorState)
    g.add_node("supervisor", _supervisor_node)
    g.add_node("wiki", _make_wiki_node(store))
    g.add_node("dart", _dart_node)
    g.add_node("compose", _compose_node)
    g.add_edge(START, "supervisor")   # supervisor/worker nodes route dynamically via Command(goto)
    g.add_edge("compose", END)
    return g.compile()


# --- public API (unchanged signatures: core.py / api dispatch here) ----------------------


def _seed(question: str) -> SupervisorState:
    return {"question": question, "answers": {}, "called": [], "trace": [], "turns": 0}


_LIMIT = {"recursion_limit": 25}


async def _fallback(question: str, store: Any) -> dict:
    """Degrade to the cheap classifier + direct dispatch when the supervisor graph fails."""
    from ..core import arun, route

    try:
        src = await asyncio.to_thread(route, question)
    except Exception:  # noqa: BLE001 — routing must never hard-fail
        src = "wiki"
    if src == "dart":
        from .dart import _arun as dart_arun
        return {"source": "dart", **(await dart_arun(question))}
    out = await arun(question, store=store)
    return {"source": "wiki", "answer": out["answer"], "trace": out["trace"],
            "steps": out["steps"], "evidence": out.get("evidence", [])}


async def arun_supervised(question: str, store: Any = None) -> dict:
    """Answer via the supervisor graph. Returns ``{source, answer, trace, steps}``.

    Drives the graph to its ``compose`` node; on any failure (or an empty/ungrounded result)
    degrades to ``route`` + direct dispatch so the request still gets a grounded answer."""
    try:
        final: dict = await _build_supervisor(store).ainvoke(_seed(question), config=_LIMIT)
    except Exception:  # noqa: BLE001 — graph/endpoint failure → degrade gracefully
        return await _fallback(question, store)

    answer = (final.get("answer") or "").strip()
    if not answer or final.get("source") == "none":  # nothing gathered → ground via the wiki
        return await _fallback(question, store)
    trace = _renumber(final.get("trace", []))
    return {"source": final.get("source") or _source(final.get("answers", {})),
            "answer": answer, "trace": trace, "steps": _count_calls(trace),
            "evidence": final.get("evidence", [])}


def run_supervised(question: str, store: Any = None) -> dict:
    """Sync wrapper around :func:`arun_supervised` (CLI / sync ``core.answer``)."""
    return asyncio.run(arun_supervised(question, store=store))


def _chunk(text: str, size: int = 24) -> list[str]:
    """Split a finished answer into replay fragments for token-style SSE delivery."""
    return [text[i:i + size] for i in range(0, len(text), size)] or [text]


async def astream_supervised(question: str, store: Any = None):
    """Async generator of ``step``/``token``/``answer`` events for the SSE endpoint.

    **Fast path — the common single-domain wiki question:** stream the wiki worker's *real*
    ``synthesize_stream`` tokens (true time-to-first-token) instead of routing through the
    supervisor graph, which buffers the whole answer and replays it as fake chunks. The
    ``route()=="wiki"`` gate is reliable: ``route`` flags ``dart`` only when a *listed-company*
    name is present, and a composite (wiki+dart) question always names one — so ``wiki`` implies
    wiki-only, with nothing for DART to add. This also skips the supervisor's two serial decision
    calls (decide→call, decide→FINISH) for ~all Centroid traffic.

    **Buffered path — dart / composite:** drive the graph with ``stream_mode="values"``, flushing
    each newly-recorded trace entry as a ``step`` event, then replay the final ``compose`` answer
    as ``token`` events (it's complete — workers buffer internally, and a ≥2-source merge can't be
    streamed before all sources are in). On a graph failure, or an ungrounded result, degrades to
    the direct wiki token stream."""
    from ..core import astream_run, route

    try:
        routed = await asyncio.to_thread(route, question)
    except Exception:  # noqa: BLE001 — routing must never hard-fail; default to the wiki stream
        routed = "wiki"
    if routed == "wiki":
        async for ev in astream_run(question, store=store, source="wiki"):
            yield ev
        return

    emitted = 0
    final: dict = {}
    try:
        async for state in _build_supervisor(store).astream(_seed(question), config=_LIMIT,
                                                             stream_mode="values"):
            final = state
            trace = state.get("trace", [])
            while emitted < len(trace):
                e = trace[emitted]   # read-only: the yielded event is a fresh dict, no copy needed
                yield {"type": "step", "step": emitted, "agent": e.get("agent", ""),
                       "action": e.get("action", ""), "arg": e.get("arg", ""),
                       "thought": e.get("thought", "")}
                emitted += 1
    except Exception:  # noqa: BLE001 — degrade to a direct wiki stream
        async for ev in astream_run(question, store=store, source="wiki"):
            yield ev
        return

    answer = (final.get("answer") or "").strip()
    if not answer or final.get("source") == "none":  # ungrounded → wiki stream instead of guessing
        async for ev in astream_run(question, store=store, source="wiki"):
            yield ev
        return

    for piece in _chunk(answer):
        yield {"type": "token", "text": piece}
    yield {"type": "answer", "answer": answer, "steps": _count_calls(final.get("trace", []))}


if __name__ == "__main__":
    import sys

    from src.stella_kb import config

    q = " ".join(sys.argv[1:]) or "센트로이드 기업가치는 얼마인가요?"
    print(f"tool LLM: {config.tool_llm_url()} ({config.tool_llm_model()})\n")
    out = run_supervised(q)
    for e in out["trace"]:
        print(f"  [{e['agent']}] {e['action']}: {e['arg']}")
    print(f"\nsource: {out['source']}  steps: {out['steps']}\n")
    print(out["answer"])
