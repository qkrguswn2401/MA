"""The agent pipeline: planner → (fan-out) solve → synthesizer.

The planner splits the question; each sub-question is dispatched to its own ``solve`` branch
(LangGraph ``Send``) and the branches run **concurrently**. A ``solve`` branch runs the three
sub-agents that used to be separate nodes — router → retriever → verifier — as a plain Python
retry loop, emitting their trace entries so all five personas stay visible. The retriever
itself fans out one LLM call per page. Every LLM call passes through ``_LLM_SEM`` so no more
than ``STELLA_FANOUT`` (default 4) requests hit the shared vLLM at once.

The deterministic wiki reads (``lookup``/``open_page``/``trace_links``) do all retrieval — the
LLMs only route and write prose. The shared vLLM has no native tool-calling, hence the
JSON-per-turn (ReAct-style) contract.
"""

from __future__ import annotations

import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from src.stella_kb.llm import chat

from ..io import lookup, open_page, trace_links
from ..prompts import load as load_prompt
from .state import AgentState

PLANNER = load_prompt("planner")
ROUTER = load_prompt("router")
RETRIEVER = load_prompt("retriever")
VERIFIER = load_prompt("verifier")
SYNTHESIZER = load_prompt("synthesizer")

_FANOUT = max(1, int(os.getenv("STELLA_FANOUT", "4")))  # concurrent LLM requests cap
_LLM_SEM = threading.Semaphore(_FANOUT)  # guards the shared guest vLLM from overload
_SYNTH_ORDER = 10**9  # sorts the synthesizer's trace entry last, after every branch


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
    return {"step": seq, "sub": sub, "agent": agent,
            "action": action, "arg": arg, "thought": thought}


# --------------------------------------------------------------------------- planner
def planner_node(state: AgentState) -> AgentState:
    """Break the question into a minimal list of sub-questions (each fans out to a branch)."""
    user = (f"INDEX:\n{state['index_md']}\n\nQuestion: {state['question']}\n\n"
            "Return the plan JSON.")
    act, _ = _ask(PLANNER, user, 600)
    plan = [p for p in ((act or {}).get("plan") or []) if isinstance(p, dict) and p.get("ask")]
    if not plan:  # parse miss / empty → fall back to a single pass-through sub-question
        plan = [{"ask": state["question"], "hint_terms": []}]
    for p in plan:  # normalize the routing controls so the solve branch can rely on them
        p["mode"] = "trace" if p.get("mode") == "trace" else "lookup"
        p["direction"] = "up" if p.get("direction") == "up" else "down"
    if state.get("verbose"):
        print(f"[planner] {len(plan)} sub-question(s) → fan out")
    return {
        "plan": plan,
        "trace": [_rec(-1, 0, "planner", "plan",
                       f"{len(plan)} sub-Q", (act or {}).get("thought", ""))],
    }


# ---------------------------------------------------------- per-sub-question sub-agents
def _route(sub: dict, tried: list, index: dict, index_md: str) -> tuple[list, dict | None, str]:
    """Pick the wiki page(s) for one sub-question; on a trace sub-Q expand along the DAG."""
    hints = sub.get("hint_terms") or []
    lookups = "\n\n".join(lookup(index, t) for t in hints) if hints else "(no hint terms)"
    avoid = (f"\nAlready tried for this sub-question and found insufficient — pick a "
             f"DIFFERENT page unless re-reading is clearly justified: {tried}") if tried else ""
    user = (f"INDEX:\n{index_md}\n\nLookup results:\n{lookups}\n\n"
            f"Sub-question: {sub['ask']}{avoid}\n\nReturn the pages JSON.")
    act, _ = _ask(ROUTER, user, 400)
    valid = set(index.get("pages", {}).keys())
    picks = [p for p in ((act or {}).get("pages") or []) if p in valid]  # drop hallucinations

    path = None
    if sub.get("mode") == "trace" and picks:
        direction = sub.get("direction", "down")
        chain = trace_links(index, picks[0], direction=direction)
        chain_pages = [c["sheet"] for c in chain
                       if c["has_page"] and c["sheet"] not in picks][:5]
        path = {"ask": sub["ask"], "direction": direction, "start": picks[0], "chain": chain}
        picks = picks + chain_pages
    return picks, path, (act or {}).get("thought", "")


def _retrieve(ask: str, pages: list) -> tuple[list, str]:
    """Open the pages and extract evidence — one LLM call PER PAGE, fanned out concurrently."""
    if not pages:
        return [], "(no pages selected)"
    texts = {p: open_page(p) for p in pages}

    def extract(page: str) -> list:
        user = (f"Sub-question: {ask}\n\nWIKI PAGE:\n{texts[page]}\n\n"
                "Return the evidence JSON.")
        act, _ = _ask(RETRIEVER, user, 800)
        out = []
        for e in (act or {}).get("evidence") or []:
            if not isinstance(e, dict):
                continue
            cell = str(e.get("cell", ""))
            celltok = cell.split("!")[-1]  # soft guard: the cell must be on THIS page
            if celltok and _cell_on_page(celltok, texts[page]):
                out.append({"page": e.get("page", "") or page, "cell": cell,
                            "term": e.get("term", ""), "value": str(e.get("value", "")),
                            "ask": ask})
        return out

    # branch threads spawn this pool too — the _LLM_SEM (not the worker count) is the real
    # cap, so total live threads can exceed _FANOUT but in-flight LLM requests never do.
    with ThreadPoolExecutor(max_workers=min(_FANOUT, len(pages))) as ex:
        per_page = list(ex.map(extract, pages))
    ev = [e for page_ev in per_page for e in page_ev]
    return ev, f"{len(ev)} fact(s) from {pages}"


def _verify(sub: dict, ev: list, path: dict | None) -> tuple[str, str]:
    """Judge whether the sub-question is answered. A traced chain is accepted as-is."""
    if sub.get("mode") == "trace" and path and path.get("chain"):
        return "ok", "provenance chain traced"
    ev_txt = "\n".join(f"- {e['term']} = {e['value']}  ({e['cell']}, {e['page']})"
                       for e in ev) or "(no evidence)"
    user = f"Sub-question: {sub['ask']}\n\nEvidence:\n{ev_txt}\n\nReturn the verdict JSON."
    act, _ = _ask(VERIFIER, user, 300)
    verdict = ((act or {}).get("verdict") or ("ok" if ev else "gap")).lower()
    return verdict, (act or {}).get("reason", "")


# ------------------------------------------------------------- solve (one fan-out branch)
def solve_node(state: AgentState, index: dict) -> AgentState:
    """Resolve ONE sub-question end to end (router → retriever → verifier, with retries).

    Runs as a parallel ``Send`` branch; returns only the ``operator.add`` channels, which
    LangGraph merges with the other branches at the barrier before the synthesizer."""
    sub = state["sub"]
    index_md = state["index_md"]              # the router prompt needs the ToC
    idx = state.get("sub_idx", 0)
    max_steps = max(1, state.get("max_steps", 3))  # per-branch read budget (initial + retries)
    verbose = state.get("verbose")

    tried: list = []
    evidence: list = []
    paths: list = []
    trace: list = []
    seen: set = set()                         # (page, cell) already captured — dedup retries
    reads = seq = 0
    while True:
        picks, path, rthought = _route(sub, tried, index, index_md)
        trace.append(_rec(idx, seq, "router", "route", ", ".join(picks) or "(none)", rthought))
        seq += 1
        if path:
            paths.append(path)

        ev, summary = _retrieve(sub["ask"], picks)
        for e in ev:                          # keep first sighting of each cell on this branch
            key = (e["page"], e["cell"])
            if key not in seen:
                seen.add(key)
                evidence.append(e)
        tried += picks
        reads += 1
        trace.append(_rec(idx, seq, "retriever", "read", summary, ""))
        seq += 1

        verdict, reason = _verify(sub, ev, path)
        trace.append(_rec(idx, seq, "verifier", "verify", verdict, reason))
        seq += 1

        if verdict != "gap" or reads >= max_steps:  # answered, or branch budget spent
            break

    if verbose:
        tag = f"[trace {sub.get('direction')}]" if sub.get("mode") == "trace" else ""
        print(f"[solve#{idx}]{tag} {sub['ask'][:42]} → {len(evidence)} ev, {len(paths)} path")
    return {"evidence": evidence, "paths": paths, "steps": reads, "trace": trace}


# ----------------------------------------------------------------------- synthesizer
def synthesizer_node(state: AgentState) -> AgentState:
    """Write the final cited Korean answer from the accumulated evidence + traced paths."""
    ev = state.get("evidence", [])
    ev_txt = "\n".join(
        f"- [{e['ask']}] {e['term']} = {e['value']}  ({e['cell']}, page {e['page']})" for e in ev
    ) or "(no evidence gathered)"

    # provenance chains traced over the formula DAG (sheet path; ⇒ marks a wiki page)
    path_txt = ""
    for pth in state.get("paths", []):
        arrow = "흘러가는" if pth["direction"] == "down" else "의존하는"
        hops = " → ".join(f"{c['sheet']}{'⇒page' if c['has_page'] else ''}" for c in pth["chain"])
        if hops:
            path_txt += f"\n- [{pth['ask']}] {pth['start']} 에서 {arrow} 경로: {pth['start']} → {hops}"
    path_block = f"\n\nProvenance chains (formula DAG, deterministic):{path_txt}" if path_txt else ""

    user = (f"Question: {state['question']}\n\nEvidence collected from the wiki:\n{ev_txt}"
            f"{path_block}\n\nWrite the final answer JSON.")
    act, raw = _ask(SYNTHESIZER, user, 700)
    text = ((act or {}).get("text") or raw or "").strip() or "(빈 답변)"
    return {
        "answer": text,
        "trace": [_rec(_SYNTH_ORDER, 0, "synthesizer", "answer", "", (act or {}).get("thought", ""))],
    }
