"""The agent pipeline: planner → (fan-out) solve → auditor, then synthesize() outside the graph.

This module is now a thin **facade** that re-exports the personas from their own files, so the
historical ``from .nodes import …`` / ``nodes.X`` surface keeps working:

    engine.py      shared LLM-call infra (_ask / _LLM_SEM / set_fanout), prompts, small helpers
    plan.py        planner_node — split the question into sub-questions
    solve.py       _route / _retrieve / _verify / solve_node — one fan-out branch
    audit.py       auditor_node — deterministic cross-evidence audit
    synthesize.py  synthesize / synthesize_stream — final answer, written AFTER the graph

The graph (planner → solve×N → auditor) ends at the auditor; the answer is written by
``synthesize``/``synthesize_stream`` from ``core`` so it can be streamed token by token. Every LLM
call passes through ``engine._LLM_SEM`` so no more than ``STELLA_FANOUT`` (default 4) requests hit
the shared vLLM at once. Deterministic wiki reads live in ``apps.agent.retrieval``.

Note for monkeypatching: ``_ask`` lives in ``engine`` and ``chat``/``chat_stream`` in
``synthesize`` — patch them there, not on this facade (re-exports won't redirect the real lookup).
"""

from __future__ import annotations

from .audit import auditor_node
from .engine import (
    PLANNER,
    RETRIEVER,
    ROUTER,
    SYNTHESIZER,
    VERIFIER,
    parse_action,
    set_fanout,
)
from .plan import planner_node
from .solve import (
    _ledger_evidence,
    _match_page,
    _retrieve,
    _route,
    _verify,
    solve_node,
)
from .synthesize import _synth_user, synthesize, synthesize_stream

__all__ = [
    "PLANNER", "ROUTER", "RETRIEVER", "VERIFIER", "SYNTHESIZER",
    "parse_action", "set_fanout",
    "planner_node", "solve_node", "auditor_node",
    "synthesize", "synthesize_stream",
    "_route", "_retrieve", "_verify", "_match_page", "_ledger_evidence", "_synth_user",
]
