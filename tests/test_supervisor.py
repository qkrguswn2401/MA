"""Supervisor StateGraph — tested deterministically (offline, no LLM / no network).

The supervisor decides via a JSON completion (``supervisor.chat``) and dispatches to wiki/dart
worker nodes; here ``chat`` is scripted and the workers (``core.arun`` / ``dart._arun``)
are stubbed, so we drive the *real compiled graph* and assert routing, passthrough vs merge,
the trace/source bookkeeping, the per-request ``store`` threading, and the fallback path —
without touching gemma-4. The live end-to-end run is the ``__main__`` smoke, not here.
"""

from __future__ import annotations

import asyncio

from apps.agent.backends import supervisor


# --- pure helpers ----------------------------------------------------------------------


def test_source_count_tag_renumber_helpers():
    assert supervisor._source({"wiki": "x", "dart": "y"}) == "dart+wiki"
    assert supervisor._source({"wiki": "x"}) == "wiki"
    assert supervisor._source({}) == "supervisor"

    trace = [
        {"agent": "supervisor", "action": "call"},
        {"agent": "wiki:planner", "action": "plan"},
        {"agent": "supervisor", "action": "call"},
        {"agent": "supervisor", "action": "result"},
    ]
    assert supervisor._count_calls(trace) == 2

    tagged = supervisor._tag([{"agent": "planner", "action": "plan", "arg": "x", "thought": ""}], "wiki")
    assert tagged[0]["agent"] == "wiki:planner"

    renum = supervisor._renumber([{"agent": "a"}, {"agent": "b"}, {"agent": "c"}])
    assert [e["step"] for e in renum] == [0, 1, 2]

    assert supervisor._chunk("abcdef", size=4) == ["abcd", "ef"]
    assert supervisor._chunk("") == [""]


# --- the decision -----------------------------------------------------------------------


def test_decide_parses_json(monkeypatch):
    monkeypatch.setattr(supervisor, "chat",
                        lambda *a, **k: '{"next":"wiki","query":"엔터프라이즈 밸류","thought":"내부"}')
    d = supervisor._decide("질문", called=[], answers={})
    assert d == {"next": "wiki", "query": "엔터프라이즈 밸류", "thought": "내부"}


def test_decide_defaults_to_finish_on_garbage(monkeypatch):
    monkeypatch.setattr(supervisor, "chat", lambda *a, **k: "no json here")
    d = supervisor._decide("원래 질문", called=["wiki"], answers={"wiki": "a"})
    assert d["next"] == "FINISH"
    assert d["query"] == "원래 질문"          # falls back to the original question


def test_decide_survives_chat_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("endpoint down")
    monkeypatch.setattr(supervisor, "chat", boom)
    assert supervisor._decide("q", [], {})["next"] == "FINISH"


# --- scripted graph drivers -------------------------------------------------------------


def _script_chat(monkeypatch, decisions: list[str], merged: str = "종합 답변"):
    """Stub ``supervisor.chat``: hand back queued decision JSON in order; once the queue is
    drained (i.e. the compose merge call), return ``merged``."""
    queue = list(decisions)

    def fake(messages, **k):
        return queue.pop(0) if queue else merged
    monkeypatch.setattr(supervisor, "chat", fake)


def _stub_workers(monkeypatch, wiki="위키 답변", dart="다트 답변", seen=None):
    async def fake_arun(q, store=None, **k):
        if seen is not None:
            seen["wiki_q"], seen["store"] = q, store
        return {"answer": wiki, "trace": [{"agent": "planner", "action": "plan", "arg": "1", "thought": ""}],
                "steps": 1, "evidence": [{"page": "PL", "cell": "J7", "term": "관리수수료", "value": "12,411"}]}

    async def fake_dart(q):
        if seen is not None:
            seen["dart_q"] = q
        return {"answer": dart, "trace": [{"agent": "dart", "action": "call", "arg": "x", "thought": ""}],
                "steps": 1}

    monkeypatch.setattr("apps.agent.core.arun", fake_arun)
    monkeypatch.setattr("apps.agent.backends.dart._arun", fake_dart)


def test_single_wiki_passthrough(monkeypatch):
    # supervisor picks wiki, then FINISH → one source → answer is the worker's, VERBATIM.
    _script_chat(monkeypatch, ['{"next":"wiki","query":"q1","thought":"t"}',
                               '{"next":"FINISH","query":"","thought":"done"}'])
    seen = {}
    _stub_workers(monkeypatch, wiki="센트로이드 EV는 1206억", seen=seen)

    out = supervisor.run_supervised("센트로이드 기업가치?")
    assert out["source"] == "wiki"
    assert out["answer"] == "센트로이드 EV는 1206억"      # passthrough — not re-prosed
    assert out["steps"] == 1
    assert seen["wiki_q"] == "q1"                          # the supervisor's tailored sub-query
    assert out["evidence"] == [{"page": "PL", "cell": "J7", "term": "관리수수료", "value": "12,411"}]
    assert any(e["agent"] == "wiki:planner" for e in out["trace"])   # worker trace namespaced
    assert out["trace"][-1]["action"] == "passthrough"
    assert [e["step"] for e in out["trace"]] == list(range(len(out["trace"])))


def test_composite_two_sources_merge(monkeypatch):
    _script_chat(monkeypatch,
                 ['{"next":"wiki","query":"센트로이드","thought":"t"}',
                  '{"next":"dart","query":"삼성전자","thought":"t"}',
                  '{"next":"FINISH","query":"","thought":"done"}'],
                 merged="센트로이드와 삼성전자 비교 종합")
    seen = {}
    _stub_workers(monkeypatch, seen=seen)

    out = supervisor.run_supervised("센트로이드와 삼성전자 비교")
    assert out["source"] == "dart+wiki"
    assert out["answer"] == "센트로이드와 삼성전자 비교 종합"   # ≥2 sources → LLM merge
    assert out["steps"] == 2
    assert seen["dart_q"] == "삼성전자"


def test_grounds_when_supervisor_finishes_empty(monkeypatch):
    # supervisor says FINISH before calling anything → must still ground via the wiki, not finish empty.
    _script_chat(monkeypatch, ['{"next":"FINISH","query":"","thought":"몰라"}'])
    _stub_workers(monkeypatch, wiki="그래도 위키 답변")

    out = supervisor.run_supervised("애매한 질문")
    assert out["source"] == "wiki"
    assert out["answer"] == "그래도 위키 답변"


def test_graph_failure_falls_back_to_route(monkeypatch):
    def boom(_store):
        raise RuntimeError("graph build boom")
    monkeypatch.setattr(supervisor, "_build_supervisor", boom)
    monkeypatch.setattr("apps.agent.core.route", lambda q: "dart")

    async def fake_dart(q):
        return {"answer": "다트 폴백", "trace": [], "steps": 1}
    monkeypatch.setattr("apps.agent.backends.dart._arun", fake_dart)

    out = supervisor.run_supervised("삼성전자 매출?")
    assert out["source"] == "dart"
    assert out["answer"] == "다트 폴백"


# --- streaming: fast path vs graph ------------------------------------------------------


def _drain(agen):
    async def collect():
        return [ev async for ev in agen]
    return asyncio.run(collect())


def test_stream_fast_path_streams_real_wiki_tokens(monkeypatch):
    # route says wiki (the common single-domain case) → stream the wiki worker's REAL tokens
    # via core.astream_run; the supervisor graph must NOT be built (no decide round-trips).
    monkeypatch.setattr("apps.agent.core.route", lambda q: "wiki")

    async def fake_astream(q, store=None, source="wiki"):
        assert source == "wiki"          # never re-enter the supervisor (source="auto" would loop)
        yield {"type": "token", "text": "센트로이드 "}
        yield {"type": "token", "text": "EV 1206억"}
        yield {"type": "answer", "answer": "센트로이드 EV 1206억", "steps": 1}
    monkeypatch.setattr("apps.agent.core.astream_run", fake_astream)
    monkeypatch.setattr(supervisor, "_build_supervisor",
                        lambda store: (_ for _ in ()).throw(AssertionError("graph built on fast path")))

    evs = _drain(supervisor.astream_supervised("센트로이드 기업가치?"))
    assert [e["type"] for e in evs] == ["token", "token", "answer"]
    assert evs[-1]["answer"] == "센트로이드 EV 1206억"


def test_stream_dart_path_uses_graph(monkeypatch):
    # route says dart → the fast path is skipped; the supervisor graph runs (buffered) and the
    # single dart source is passed through, then chunk-replayed as tokens.
    monkeypatch.setattr("apps.agent.core.route", lambda q: "dart")
    _script_chat(monkeypatch, ['{"next":"dart","query":"삼성전자","thought":"t"}',
                               '{"next":"FINISH","query":"","thought":"done"}'])
    _stub_workers(monkeypatch, dart="삼성전자 매출 300조")

    evs = _drain(supervisor.astream_supervised("삼성전자 매출?"))
    types = [e["type"] for e in evs]
    assert "step" in types and types[-1] == "answer"
    assert evs[-1]["answer"] == "삼성전자 매출 300조"        # passthrough, not re-prosed


# --- nodes in isolation -----------------------------------------------------------------


def test_compose_node_passthrough_vs_merge(monkeypatch):
    one = asyncio.run(supervisor._compose_node({"question": "q", "answers": {"wiki": "단일"}}))
    assert one == {"answer": "단일", "source": "wiki",
                   "trace": [{"agent": "supervisor", "action": "passthrough", "arg": "wiki", "thought": ""}]}

    monkeypatch.setattr(supervisor, "chat", lambda *a, **k: "병합 결과")
    two = asyncio.run(supervisor._compose_node({"question": "q", "answers": {"wiki": "a", "dart": "b"}}))
    assert two["answer"] == "병합 결과" and two["source"] == "dart+wiki"

    empty = asyncio.run(supervisor._compose_node({"question": "q", "answers": {}}))
    assert empty == {"answer": "", "source": "none"}


def test_wiki_node_threads_store(monkeypatch):
    seen = {}
    _stub_workers(monkeypatch, wiki="a", seen=seen)
    cmd = asyncio.run(supervisor._make_wiki_node("STORE_SENTINEL")({"question": "q", "next_query": "tailored"}))

    assert seen["store"] == "STORE_SENTINEL"      # the per-request store reached the worker
    assert seen["wiki_q"] == "tailored"           # the supervisor's sub-query, not the raw question
    assert cmd.goto == "supervisor"
    assert cmd.update["answers"] == {"wiki": "a"}
    assert cmd.update["evidence"] == [{"page": "PL", "cell": "J7", "term": "관리수수료", "value": "12,411"}]
    assert cmd.update["trace"][0]["agent"] == "wiki:planner"
    assert cmd.update["trace"][-1]["action"] == "result"
