"""Handoff-tool supervisor — routes a question across the wiki and DART backends as TOOLS.

Replaces the upfront wiki-vs-dart classifier (``core.route``) for ``source="auto"``. A
tool-calling gemma-4 (the same model/endpoint ``dart_agent`` uses — served WITH
``--tool-call-parser gemma4`` on :8001) is handed two **handoff tools**:

  - ``consult_centroid_wiki`` — wraps the wiki LangGraph pipeline (``core.arun``)
  - ``consult_dart``          — wraps the DART tool agent (``dart_agent._arun``)

It decides which tool(s) to call (BOTH for a composite cross-source question), then composes
the final Korean answer itself. Two-phase, because ``create_agent``'s message stream can't
cleanly separate the supervisor's final-answer tokens from its intermediate tool-announcing
tokens (gemma-4 also leaks ``<|channel>…<channel|>`` control tokens):

  Phase A (dispatch) — run the ``create_agent`` tool loop; tools record their worker trace +
                       answer into shared lists as they execute.
  Phase B (compose)  — the same model writes the final Korean answer over the gathered tool
                       outputs. Buffered path uses the agent's own terminal message; the
                       streaming path re-composes via ``ChatOpenAI.astream`` so tokens stream
                       (mirrors ``graph.nodes.synthesize_stream``).

Worker backends are reused unchanged — both already return ``{answer, trace, steps}``. The
per-request :class:`apps.agent.datasets.WikiStore` (``store``) is threaded into the wiki tool
via a closure, so concurrency safety is preserved (no process global). On any failure the
supervisor degrades to ``core.route`` + direct dispatch, so a flaky tool-calling round never
hard-fails a request.

Config (env), shared with ``dart_agent``:
    STELLA_TOOL_LLM_URL    tool-calling LLM base URL   (default http://123.37.5.219:8001/v1)
    STELLA_TOOL_LLM_MODEL  served model name           (default gemma-4-31B-it)
"""

from __future__ import annotations

import asyncio
from typing import Any

from src.stella_kb import config

from .dart_agent import _CHANNEL_OPEN, _clean
from .prompts import load as load_prompt


def _strip_channel(text: str) -> str:
    """Remove gemma-4 channel control tokens WITHOUT stripping surrounding whitespace — for
    per-delta use while streaming (``_clean`` also ``.strip()``s, which would drop the spaces
    between streamed tokens). The joined final answer still goes through full ``_clean``."""
    return _CHANNEL_OPEN.sub("", text or "").replace("<channel|>", "")


def _llm():
    """The tool-calling / compose model — same gemma-4 on :8001 the DART agent uses."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.tool_llm_model(), base_url=config.tool_llm_url(),
                      api_key="EMPTY", temperature=0)


def _tag(trace: list[dict], source: str) -> list[dict]:
    """Namespace a worker's trace entries (``agent`` → ``"wiki:planner"`` etc.), order kept."""
    out = []
    for e in trace:
        e = dict(e)
        e["agent"] = f"{source}:{e.get('agent', '')}".rstrip(":")
        out.append(e)
    return out


def _renumber(trace: list[dict]) -> list[dict]:
    """Assign a sequential global ``step`` over the merged (already execution-ordered) trace."""
    out = []
    for i, e in enumerate(trace):
        e = dict(e)
        e["step"] = i
        out.append(e)
    return out


def _count_calls(events: list[dict]) -> int:
    """How many handoff tool calls the supervisor made (the 'work' metric)."""
    return sum(1 for e in events if e.get("agent") == "supervisor" and e.get("action") == "call")


def _source(answers: dict) -> str:
    """Which backend(s) actually answered: ``"wiki"`` | ``"dart"`` | ``"dart+wiki"``."""
    return "+".join(sorted(answers)) if answers else "supervisor"


def _make_tools(store: Any, events: list[dict], answers: dict):
    """Build the two handoff tools, closing over the per-request ``store`` and the shared
    ``events``/``answers`` accumulators (so a tool's worker trace + answer are captured in
    execution order without parsing LangChain message objects)."""
    from langchain_core.tools import tool

    @tool
    async def consult_centroid_wiki(question: str) -> str:
        """센트로이드(Centroid Investment Partners / Management) 내부 밸류에이션 지식베이스에 질의합니다.
        DCF, AUM 추정, 펀드 관리수수료·성과보수, 기업가치(EV), 할인율 등 센트로이드/자사 펀드 관련 질문,
        그리고 회사명이 명시되지 않은 모든 재무·리포트·비율 계산 질문에 사용하세요."""
        from .core import arun  # deferred: avoid a core <-> supervisor import cycle

        events.append({"agent": "supervisor", "action": "call",
                       "arg": f"consult_centroid_wiki(question={question[:100]})", "thought": ""})
        out = await arun(question, store=store)
        events.extend(_tag(out.get("trace", []), "wiki"))
        events.append({"agent": "supervisor", "action": "result",
                       "arg": f"wiki: {(out.get('answer') or '')[:120]}", "thought": ""})
        answers["wiki"] = out.get("answer", "")
        return out.get("answer") or "(빈 답변)"

    @tool
    async def consult_dart(question: str) -> str:
        """국내 상장 기업(예: 삼성전자, 카카오)의 DART 공시·재무 정보를 조회합니다.
        질문에 특정 상장사 회사명이 명시된 경우에만 사용하세요."""
        from .dart_agent import _arun as dart_arun

        events.append({"agent": "supervisor", "action": "call",
                       "arg": f"consult_dart(question={question[:100]})", "thought": ""})
        out = await dart_arun(question)
        events.extend(_tag(out.get("trace", []), "dart"))
        events.append({"agent": "supervisor", "action": "result",
                       "arg": f"dart: {(out.get('answer') or '')[:120]}", "thought": ""})
        answers["dart"] = out.get("answer", "")
        return out.get("answer") or "(빈 답변)"

    return [consult_centroid_wiki, consult_dart]


def _build_agent(store: Any, events: list[dict], answers: dict):
    """Compile the phase-A dispatch agent (factored out so tests can stub it offline)."""
    from langchain.agents import create_agent

    return create_agent(model=_llm(), tools=_make_tools(store, events, answers),
                         system_prompt=load_prompt("supervisor"))


def _compose_msgs(question: str, answers: dict) -> list[tuple[str, str]]:
    """Phase-B compose prompt: the supervisor writes the final answer over the gathered tool
    outputs only (no fresh retrieval). Same system prompt as phase A + a compose directive."""
    blocks = []
    if answers.get("wiki"):
        blocks.append(f"[센트로이드 위키 자료]\n{answers['wiki']}")
    if answers.get("dart"):
        blocks.append(f"[DART 자료]\n{answers['dart']}")
    gathered = "\n\n".join(blocks) or "(수집된 자료 없음)"
    system = (load_prompt("supervisor")
              + "\n\n이제 아래 수집 자료만 근거로 한국어 최종 답변을 작성하세요. 자료에 없는 내용은 지어내지 마세요.")
    return [("system", system), ("user", f"질문: {question}\n\n{gathered}")]


async def _compose_stream(question: str, answers: dict):
    """Stream the final answer token by token (phase B), cleaning gemma channel tokens."""
    async for chunk in _llm().astream(_compose_msgs(question, answers)):
        delta = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        delta = _strip_channel(delta)   # NOT _clean: keep inter-token whitespace while streaming
        if delta:
            yield delta


async def _fallback(question: str, store: Any) -> dict:
    """Degrade to the cheap classifier + direct dispatch when the supervisor round fails."""
    from .core import arun, route

    try:
        src = await asyncio.to_thread(route, question)
    except Exception:  # noqa: BLE001 — routing must never hard-fail
        src = "wiki"
    if src == "dart":
        from .dart_agent import _arun as dart_arun
        return {"source": "dart", **(await dart_arun(question))}
    out = await arun(question, store=store)
    return {"source": "wiki", "answer": out["answer"], "trace": out["trace"], "steps": out["steps"]}


async def arun_supervised(question: str, store: Any = None) -> dict:
    """Answer via the handoff-tool supervisor. Returns ``{source, answer, trace, steps}``.

    Phase A runs the ``create_agent`` tool loop (its tools record their worker trace + answer);
    the agent's own terminal message is the composed answer. On any failure, falls back to
    ``route`` + direct dispatch so the request still gets a grounded answer."""
    events: list[dict] = []
    answers: dict = {}
    try:
        result = await _build_agent(store, events, answers).ainvoke({"messages": [("user", question)]})
    except Exception:  # noqa: BLE001 — tool-calling/endpoint failure → degrade gracefully
        return await _fallback(question, store)

    if not answers:  # supervisor called no tool → ground via the wiki rather than guess
        return await _fallback(question, store)

    msgs = result.get("messages", [])
    terminal = msgs[-1].content if msgs else ""
    answer = _clean(terminal if isinstance(terminal, str) else str(terminal))
    if not answer:  # empty terminal → fall back to the joined tool outputs
        answer = "\n\n".join(v for v in answers.values() if v)
    trace = _renumber(events + [{"agent": "supervisor", "action": "answer", "arg": "", "thought": ""}])
    return {"source": _source(answers), "answer": answer or "(답변 없음)",
            "trace": trace, "steps": _count_calls(events)}


def run_supervised(question: str, store: Any = None) -> dict:
    """Sync wrapper around :func:`arun_supervised` (CLI / sync ``core.answer``)."""
    return asyncio.run(arun_supervised(question, store=store))


async def astream_supervised(question: str, store: Any = None):
    """Async generator of ``step``/``token``/``answer`` events for the SSE endpoint.

    Phase A streams the dispatch loop, flushing each newly-recorded tool call/result as a
    ``step`` event; phase B re-composes the final answer with the same model, streamed as
    ``token`` events. On a phase-A failure, degrades to the direct wiki token stream."""
    events: list[dict] = []
    answers: dict = {}
    emitted = 0

    def _flush():
        nonlocal emitted
        out = []
        while emitted < len(events):
            e = events[emitted]
            out.append({"type": "step", "step": emitted, "agent": e["agent"],
                        "action": e["action"], "arg": e["arg"], "thought": e.get("thought", "")})
            emitted += 1
        return out

    try:
        agent = _build_agent(store, events, answers)
        async for _ in agent.astream({"messages": [("user", question)]}, stream_mode="updates"):
            for ev in _flush():
                yield ev
    except Exception:  # noqa: BLE001 — degrade to a direct wiki stream
        from .core import astream_run
        async for ev in astream_run(question, store=store, source="wiki"):
            yield ev
        return

    for ev in _flush():  # any trailing tool events
        yield ev

    if not answers:  # no tool fired → ground via the wiki stream instead of guessing
        from .core import astream_run
        async for ev in astream_run(question, store=store, source="wiki"):
            yield ev
        return

    parts: list[str] = []
    async for delta in _compose_stream(question, answers):
        parts.append(delta)
        yield {"type": "token", "text": delta}
    yield {"type": "answer", "answer": _clean("".join(parts)) or "(답변 없음)",
           "steps": _count_calls(events)}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "센트로이드 기업가치는 얼마인가요?"
    print(f"tool LLM: {config.tool_llm_url()} ({config.tool_llm_model()})\n")
    out = run_supervised(q)
    for e in out["trace"]:
        print(f"  [{e['agent']}] {e['action']}: {e['arg']}")
    print(f"\nsource: {out['source']}  steps: {out['steps']}\n")
    print(out["answer"])
