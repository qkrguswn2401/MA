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
import re
import threading
from concurrent.futures import ThreadPoolExecutor

from src.stella_kb import config
from src.stella_kb.llm import chat

from ..io import lookup, open_page, query_ledger, trace_links
from ..prompts import load as load_prompt
from .state import AgentState

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
def _match_page(raw_pick: str, valid: set, by_norm: dict) -> str | None:
    """Resolve a router-emitted page name to an exact INDEX key, tolerating the forms the
    model actually produces. The INDEX presents pages as ``[[page]]`` wikilinks, so the model
    frequently copies the brackets (and sometimes quotes); a strict ``p in valid`` then
    silently drops a perfectly good pick (e.g. ``[[BS]]`` ≠ ``BS``) and the branch starves.
    Strip ``[[ ]]``/quotes/whitespace, then fall back to a normalized (space/case-insensitive)
    match before giving up."""
    if not isinstance(raw_pick, str):
        return None
    p = raw_pick.strip().strip("\"'").strip()
    if p.startswith("[[") and p.endswith("]]"):
        p = p[2:-2].strip()
    if p in valid:
        return p
    return by_norm.get(re.sub(r"\s+", "", p).casefold())


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
    by_norm = {re.sub(r"\s+", "", v).casefold(): v for v in valid}
    picks, seen = [], set()
    for raw in (act or {}).get("pages") or []:        # tolerate [[wikilink]]/quote forms
        m = _match_page(raw, valid, by_norm)
        if m and m not in seen:                       # resolve + dedup; drop hallucinations
            seen.add(m)
            picks.append(m)

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
                            "term": e.get("term", ""), "period": str(e.get("period", "")),
                            "value": str(e.get("value", "")), "ask": ask})
        return out

    # branch threads spawn this pool too — the _LLM_SEM (not the worker count) is the real
    # cap, so total live threads can exceed _FANOUT but in-flight LLM requests never do.
    with ThreadPoolExecutor(max_workers=min(_FANOUT, len(pages))) as ex:
        per_page = list(ex.map(extract, pages))
    ev = [e for page_ev in per_page for e in page_ev]
    return ev, f"{len(ev)} fact(s) from {pages}"


def _ledger_evidence(picks: list, sub: dict) -> list:
    """For any ``*_거래내역`` page picked, run the deterministic ledger filter+sum.

    Transaction rows aren't on the wiki page (the time-series parse drops them), so the LLM
    retriever finds nothing there. This pulls them from the ledger sidecar and sums 출금 by
    적요 keyword (the sub-question's ``hint_terms``) deterministically — exact, cell-cited."""
    kws = [k for k in (sub.get("hint_terms") or []) if k]
    out: list = []
    for p in picks:
        if isinstance(p, str) and p.endswith("_거래내역"):
            out += query_ledger(p, kws, sub.get("ask", ""))
    return out


def _verify(sub: dict, ev: list, path: dict | None) -> tuple[str, str]:
    """Judge whether the sub-question is answered. A traced chain is accepted as-is."""
    if sub.get("mode") == "trace" and path and path.get("chain"):
        return "ok", "provenance chain traced"
    ev_txt = "\n".join(f"- {e['term']}{_per(e)} = {e['value']}  ({e['cell']}, {e['page']})"
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
        led = _ledger_evidence(picks, sub)    # deterministic 거래내역 filter+sum (rows not on page)
        if led:
            ev = ev + led
            summary += f"  +ledger({len(led)})"
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


# --------------------------------------------------------------------------- auditor
def _is_pdf_page(meta: dict) -> bool:
    """Whether a page came from the FDD PDF (vs the Excel workbook). PDF pages carry a
    `pdf …` kind / a `… (PDF)` section in the index; Excel pages don't. Used to tell a
    report *claim* apart from a source-of-truth Excel value."""
    blob = f"{meta.get('kind', '')} {meta.get('section', '')}".lower()
    return "pdf" in blob


def auditor_node(state: AgentState, index: dict) -> AgentState:
    """Deterministic cross-evidence audit between the solve barrier and the synthesizer.

    The per-branch verifier only asks "did THIS sub-question get evidence?" — it never sees
    the merged set, so it can't catch a reconciliation that cited the *same* cell for two
    opposed quantities (fabricated agreement), a report *claim* mistaken for source data, or a
    planned sub-question that found nothing. These are exactly the over-claiming failures.
    The checks are rule-based (no LLM) so they can't hallucinate and won't touch the answers
    that are already right; they only append caveats the synthesizer must honor."""
    ev = state.get("evidence", [])
    pages = index.get("pages", {})
    caveats: list[str] = []

    # 1) same (page,cell) used as evidence for >=2 distinct sub-questions. For a "A vs B"
    #    reconciliation this means one side was never really retrieved — the smoking gun
    #    behind fabricated "두 값이 일치한다" conclusions.
    cell_asks: dict[tuple, set] = {}
    ask_ev: dict[str, list] = {}
    for e in ev:
        cell_asks.setdefault((e["page"], e["cell"]), set()).add(e["ask"])
        ask_ev.setdefault(e["ask"], []).append(e)
    for (page, cell), asks in cell_asks.items():
        if len(asks) >= 2:
            ref = cell if "!" in cell else f"{page}!{cell}"   # cell may already carry the sheet
            caveats.append(
                f"동일 출처 셀 {ref} 이(가) 서로 다른 하위질문의 근거로 중복 사용됨 "
                f"({' / '.join(sorted(asks))}). 두 항목을 서로 다른 자료로 대사한 것이 아니므로 "
                f"'일치/동일하다'라고 단정하지 말 것 — 한쪽 출처는 실제로 확인되지 않았을 수 있음.")

    # 2) a sub-question whose evidence is ENTIRELY from PDF/report pages. The report is the
    #    thing being cross-checked, not the source of truth: a value on an FDD page is a
    #    *claim*, never proof that the underlying (Excel) data exists for that period/scope.
    for ask, items in ask_ev.items():
        if items and all(_is_pdf_page(pages.get(e["page"], {})) for e in items):
            caveats.append(
                f"하위질문 '{ask}' 의 근거가 PDF/리포트 페이지에서만 나옴 — 원본(Excel) 자료로는 "
                f"확인되지 않음. 리포트의 '주장'일 뿐, 원본 자료에 그 수치/시점이 존재한다는 "
                f"증거가 아님.")

    # 3) a planned sub-question that collected no evidence at all → that part is unverifiable.
    answered = set(ask_ev)
    for p in state.get("plan", []):
        if p.get("ask") and p["ask"] not in answered:
            caveats.append(f"하위질문 '{p['ask']}' 에 대한 근거를 수집하지 못함 — 해당 부분은 '확인 불가'.")

    if state.get("verbose"):
        print(f"[auditor] {len(caveats)} caveat(s)")
    thought = f"{len(caveats)} caveat" if caveats else "이상 없음"
    return {
        "caveats": caveats,
        "trace": [_rec(_SYNTH_ORDER - 1, 0, "auditor", "audit",
                       f"{len(caveats)} caveat(s)", thought)],
    }


# ----------------------------------------------------------------------- synthesizer
def synthesizer_node(state: AgentState) -> AgentState:
    """Write the final cited Korean answer from the accumulated evidence + traced paths."""
    ev = state.get("evidence", [])
    ev_txt = "\n".join(
        f"- [{e['ask']}] {e['term']}{_per(e)} = {e['value']}  ({e['cell']}, page {e['page']})"
        for e in ev
    ) or "(no evidence gathered)"

    # provenance chains traced over the formula DAG (sheet path; ⇒ marks a wiki page)
    path_txt = ""
    for pth in state.get("paths", []):
        arrow = "흘러가는" if pth["direction"] == "down" else "의존하는"
        hops = " → ".join(f"{c['sheet']}{'⇒page' if c['has_page'] else ''}" for c in pth["chain"])
        if hops:
            path_txt += f"\n- [{pth['ask']}] {pth['start']} 에서 {arrow} 경로: {pth['start']} → {hops}"
    path_block = f"\n\nProvenance chains (formula DAG, deterministic):{path_txt}" if path_txt else ""

    # deterministic audit flags (dup-cell-across-asks, pdf-only claims, unanswered sub-Qs) —
    # the synthesizer must honor these and not over-claim agreement past them.
    caveats = state.get("caveats", [])
    caveat_block = ("\n\n감사 경고(AUDIT — 반드시 반영, 무시 금지):\n"
                    + "\n".join(f"- {c}" for c in caveats)) if caveats else ""

    user = (f"Question: {state['question']}\n\nEvidence collected from the wiki:\n{ev_txt}"
            f"{path_block}{caveat_block}\n\nWrite the final answer JSON.")
    act, raw = _ask(SYNTHESIZER, user, 700)
    text = ((act or {}).get("text") or raw or "").strip() or "(빈 답변)"
    return {
        "answer": text,
        "trace": [_rec(_SYNTH_ORDER, 0, "synthesizer", "answer", "", (act or {}).get("thought", ""))],
    }
