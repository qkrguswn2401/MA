"""End-to-end smoke tests against the live shared vLLM. Opt-in: ``pytest --run-llm``.

These assert on structure (the right page/cell is cited, the expected agents ran, fan-out
actually produced parallel branches) rather than exact wording, which varies by model run.
"""

from __future__ import annotations

import os

import pytest

from apps.agent.core import run

pytestmark = pytest.mark.llm


def test_mgt_ev_cites_dcf_exhibit():
    r = run("MGT 케이스 enterprise value는 얼마인가?")
    assert "DCF 장표 #1_MGT" in r["answer"]
    agents = {e["agent"] for e in r["trace"]}
    assert {"planner", "router", "retriever", "verifier", "synthesizer"} <= agents
    assert [e["step"] for e in r["trace"]] == list(range(len(r["trace"])))  # renumbered


def test_comparison_fans_out_to_parallel_branches():
    r = run("MGT와 DTT 케이스의 equity value를 비교하면?")
    # the planner should split this; each sub-question runs as its own branch (distinct sub)
    branch_subs = {e["sub"] for e in r["trace"] if 0 <= e.get("sub", -1) < 10**9}
    assert len(branch_subs) >= 2, "comparison should fan out into >= 2 solve branches"


def test_graph_query_comparison_fans_out_to_two_metrics():
    """The graph query layer resolves a comparison to >1 focal metric and cites both."""
    from src.stella_kb.graph import query

    if not os.path.exists(query.GRAPH_PATH):
        pytest.skip("graph not built — run `python -m src.stella_kb.graph.semantic`")
    g = query.load_graph()
    mids = query.resolve_all("관리보수와 성과보수를 비교하면?")
    assert {"Metric:management_fee", "Metric:performance_fee"} <= set(mids)
    ev = query.ask("관리보수와 성과보수를 비교하면?", synthesize=False, g=g)
    assert "Management Fee" in ev and "Performance Fee" in ev  # both metrics' evidence gathered


def test_graph_query_dual_case_equity_value():
    """A single dual-case metric answers an MGT-vs-DTT comparison, citing both cases."""
    from src.stella_kb.graph import query

    if not os.path.exists(query.GRAPH_PATH):
        pytest.skip("graph not built — run `python -m src.stella_kb.graph.semantic`")
    g = query.load_graph()
    ans = query.ask("MGT와 DTT 케이스의 equity value를 비교하면?", g=g)  # synthesize via prompt file
    assert "206,130" in ans or "206130" in ans   # MGT equity (frozen exhibit)
    assert "120,696" in ans or "120696" in ans   # DTT equity (live DCF)
