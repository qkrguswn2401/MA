"""Handoff-tool supervisor — tested deterministically (offline, no LLM / no network).

The phase-A dispatch agent (``supervisor._build_agent``) is stubbed with a fake whose
``ainvoke`` fills the shared ``events``/``answers`` the real tools would, so we can assert the
trace merge/renumber, the ``source`` bookkeeping, the per-request ``store`` threading into the
wiki tool, channel-token cleaning, and the route-fallback paths — without touching gemma-4.
The live end-to-end run is the ``__main__`` smoke (scripts/run_supervisor.sh), not here.
"""

from __future__ import annotations

import asyncio

from apps.agent import supervisor


# --- pure helpers ----------------------------------------------------------------------


def test_source_count_tag_renumber_helpers():
    assert supervisor._source({"wiki": "x", "dart": "y"}) == "dart+wiki"
    assert supervisor._source({"wiki": "x"}) == "wiki"
    assert supervisor._source({}) == "supervisor"

    events = [
        {"agent": "supervisor", "action": "call"},
        {"agent": "wiki:planner", "action": "plan"},
        {"agent": "supervisor", "action": "call"},
        {"agent": "supervisor", "action": "result"},
    ]
    assert supervisor._count_calls(events) == 2

    tagged = supervisor._tag([{"agent": "planner", "action": "plan", "arg": "x", "thought": ""}], "wiki")
    assert tagged[0]["agent"] == "wiki:planner"

    renum = supervisor._renumber([{"agent": "a"}, {"agent": "b"}, {"agent": "c"}])
    assert [e["step"] for e in renum] == [0, 1, 2]


# --- fake phase-A agent ----------------------------------------------------------------


def _fake_build(fill):
    """A ``_build_agent`` replacement whose agent.ainvoke runs ``fill(events, answers)`` (the
    side effects the real tools would have) and returns ``fill``'s value as the terminal text."""
    def build(store, events, answers):
        class _Agent:
            async def ainvoke(self, _payload):
                terminal = fill(events, answers)

                class _M:
                    content = terminal
                return {"messages": [_M()]}
        return _Agent()
    return build


def test_single_source_cleans_terminal_and_renumbers(monkeypatch):
    def fill(events, answers):
        events.append({"agent": "supervisor", "action": "call",
                       "arg": "consult_centroid_wiki(...)", "thought": ""})
        events.append({"agent": "wiki:planner", "action": "plan", "arg": "1", "thought": ""})
        events.append({"agent": "supervisor", "action": "result", "arg": "wiki: ...", "thought": ""})
        answers["wiki"] = "위키 답변"
        return "<|channel>thought\n<channel|>최종 답변"   # gemma channel tokens must be stripped

    monkeypatch.setattr(supervisor, "_build_agent", _fake_build(fill))
    out = supervisor.run_supervised("센트로이드 EV?")

    assert out["source"] == "wiki"
    assert out["answer"] == "최종 답변"
    assert out["steps"] == 1
    assert out["trace"][-1] == {"agent": "supervisor", "action": "answer", "arg": "", "thought": "", "step": 3}
    assert [e["step"] for e in out["trace"]] == list(range(len(out["trace"])))


def test_composite_two_sources(monkeypatch):
    def fill(events, answers):
        for name, src in (("consult_centroid_wiki", "wiki"), ("consult_dart", "dart")):
            events.append({"agent": "supervisor", "action": "call", "arg": f"{name}(...)", "thought": ""})
            events.append({"agent": "supervisor", "action": "result", "arg": f"{src}: ...", "thought": ""})
            answers[src] = src
        return "종합 답변"

    monkeypatch.setattr(supervisor, "_build_agent", _fake_build(fill))
    out = supervisor.run_supervised("센트로이드와 삼성전자 비교")

    assert out["source"] == "dart+wiki"
    assert out["steps"] == 2
    assert out["answer"] == "종합 답변"


def test_no_tool_call_falls_back_to_wiki(monkeypatch):
    monkeypatch.setattr(supervisor, "_build_agent", _fake_build(lambda e, a: "근거 없는 추측"))
    monkeypatch.setattr("apps.agent.core.route", lambda q: "wiki")

    async def fake_arun(q, store=None, **k):
        return {"answer": "위키 폴백", "trace": [], "steps": 0}

    monkeypatch.setattr("apps.agent.core.arun", fake_arun)
    out = supervisor.run_supervised("질문")

    assert out["source"] == "wiki"          # no tool fired → grounded wiki, not the model guess
    assert out["answer"] == "위키 폴백"


def test_dispatch_error_falls_back_to_route(monkeypatch):
    def build(store, events, answers):
        class _Agent:
            async def ainvoke(self, _):
                raise RuntimeError("tool-calling boom")
        return _Agent()

    monkeypatch.setattr(supervisor, "_build_agent", build)
    monkeypatch.setattr("apps.agent.core.route", lambda q: "dart")

    async def fake_dart(q):
        return {"answer": "다트 답변", "trace": [], "steps": 1}

    monkeypatch.setattr("apps.agent.dart_agent._arun", fake_dart)
    out = supervisor.run_supervised("삼성전자 매출?")

    assert out["source"] == "dart"
    assert out["answer"] == "다트 답변"


# --- the load-bearing closure: per-request store threads into the wiki tool -------------


def test_wiki_tool_threads_store(monkeypatch):
    seen = {}

    async def fake_arun(q, store=None, **k):
        seen["store"] = store
        return {"answer": "a", "trace": [{"agent": "planner", "action": "plan", "arg": "1", "thought": ""}],
                "steps": 2}

    monkeypatch.setattr("apps.agent.core.arun", fake_arun)
    events, answers = [], {}
    wiki_tool, _dart_tool = supervisor._make_tools("STORE_SENTINEL", events, answers)
    res = asyncio.run(wiki_tool.ainvoke({"question": "q"}))

    assert seen["store"] == "STORE_SENTINEL"   # the per-request store reached the worker
    assert answers["wiki"] == "a"
    assert res == "a"
    assert events[0]["agent"] == "supervisor" and events[0]["action"] == "call"
    assert events[1]["agent"] == "wiki:planner"   # worker trace namespaced
    assert events[-1]["action"] == "result"
