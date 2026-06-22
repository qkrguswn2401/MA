"""Shared engine for the wiki pipeline personas (planner/solve/audit/synthesize).

Holds the one-LLM-call helper and the in-flight cap every persona shares: ``_ask`` (system+user
→ parsed JSON action + raw), guarded by ``_LLM_SEM`` so no more than ``STELLA_FANOUT`` (default 4)
requests hit the shared vLLM at once. ``set_fanout`` rebinds ``_FANOUT``/``_LLM_SEM`` at runtime;
callers reach these as ``engine._FANOUT`` / ``engine._LLM_SEM`` (module-qualified) so the rebind is
picked up at call time. Also the loaded prompt constants and the small pure helpers (``_rec``,
``_per``, ``_cell_on_page``, ``parse_action``).

The deterministic wiki reads live in ``apps.agent.retrieval``; the LLMs here only route and write
prose. The shared vLLM has no native tool-calling, hence the JSON-per-turn (ReAct-style) contract.
"""

from __future__ import annotations

import json
import re
import threading

from src.stella_kb import config
from src.stella_kb.llm import chat

from ...prompts import load as load_prompt

PLANNER = load_prompt("planner")
ROUTER = load_prompt("router")
RETRIEVER = load_prompt("retriever")
VERIFIER = load_prompt("verifier")
SYNTHESIZER = load_prompt("synthesizer")

_FANOUT = max(1, config.agent_fanout())  # concurrent LLM requests cap
_LLM_SEM = threading.Semaphore(_FANOUT)  # guards the shared guest vLLM from overload
_SYNTH_ORDER = 10**9  # sorts the synthesizer's trace entry last, after every branch


def set_fanout(n: int) -> None:
    """Resize the in-flight LLM cap. The library default (4) is deliberately polite to the
    shared guest vLLM; batch jobs (e.g. the eval, which fans out many questions at once) can
    raise it to match their worker count so workers aren't all blocked on a 4-slot semaphore.
    Call before launching the work; rebinding is picked up by ``_ask`` at call time."""
    global _FANOUT, _LLM_SEM
    _FANOUT = max(1, int(n))
    _LLM_SEM = threading.Semaphore(_FANOUT)


def _per(e: dict) -> str:
    """`` (2023)`` period suffix for an evidence row, blank when the value is a scalar."""
    p = (e.get("period") or "").strip()
    return f" ({p})" if p else ""


def _cell_on_page(celltok: str, text: str) -> bool:
    """Whether a bare cell ref (``E4``, ``AU4``) occurs on the page as a *whole* token.

    A plain substring check lets ``E4`` match ``E40``/``AE4`` and wave a hallucinated cell
    through — fatal for auditable provenance — so anchor the match on column/row boundaries.
    """
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(celltok)}(?![0-9])", text))


def parse_action(raw: str) -> dict | None:
    """Extract the single JSON object from a model turn (tolerates code fences/prose)."""
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        s = max(parts, key=len).lstrip("json").strip() if len(parts) >= 3 else s.strip("`")
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start : end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _ask(system: str, user: str, max_tokens: int) -> tuple[dict | None, str]:
    """One-shot LLM call: system + user → (parsed JSON action, raw text).

    Acquires ``_LLM_SEM`` so concurrent branches/pages never exceed the request cap — vLLM
    continuous-batches whatever does land at once, which is where the speed-up comes from.
    """
    with _LLM_SEM:
        raw = chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=max_tokens,
            timeout=120.0,
        )
    return parse_action(raw), raw


def _rec(sub: int, seq: int, agent: str, action: str, arg: str, thought: str) -> dict:
    """One trace record. ``sub``/``seq`` are the branch index and intra-branch order; the
    global ``step`` is reassigned in ``core`` after the parallel branches merge."""
    return {"step": seq, "sub": sub, "agent": agent, "action": action, "arg": arg, "thought": thought}
