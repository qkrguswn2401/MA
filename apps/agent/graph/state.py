"""The LangGraph state for the planner → (fan-out) solve → synthesizer pipeline.

The planner splits the question into sub-questions; each is dispatched to its own ``solve``
branch via the ``Send`` API and they run **concurrently** (bounded by a semaphore in
``nodes.py``). Branches write only to the ``operator.add`` channels below, which LangGraph
merges across the parallel barrier — so no branch clobbers another. Per-branch working state
(picked pages, retries) stays local inside ``solve_node``; the only per-branch fields on the
state are the ``Send`` payload (``sub``/``sub_idx``)."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

# These channels accumulate across all parallel branches — each branch returns only its own
# items and LangGraph concatenates (lists) / sums (steps). This is what makes the fan-out
# safe: branches never overwrite a shared list, they append to it.


class AgentState(TypedDict, total=False):
    """Running state threaded through the multi-agent graph."""
    question: str          # the original user question
    index_md: str          # the wiki INDEX (ToC) text, handed to planner/router
    wiki_dir: str          # per-request dataset dir for page/ledger reads (None → process default)
    plan: list             # [{ask, hint_terms, mode, direction}] from the planner
    sub: dict              # Send payload: the one sub-question this solve branch handles
    sub_idx: int           # Send payload: that sub-question's index (for trace grouping)
    evidence: Annotated[list, operator.add]  # accumulated [{page, cell, term, value, ask}]
    paths: Annotated[list, operator.add]     # provenance chains [{ask, direction, chain:[...]}]
    caveats: list          # auditor's cross-evidence red flags (single write, post-merge)
    answer: str            # the synthesizer's final Korean answer
    trace: Annotated[list, operator.add]     # per-turn record [{step, sub, agent, action, arg, thought}]
    steps: Annotated[int, operator.add]      # retriever reads consumed across branches (work done)
    max_steps: int         # per-branch read budget: initial read + up to (max_steps-1) retries
    verbose: bool
