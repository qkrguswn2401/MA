"""The agent pipeline: planner → (fan-out) solve → auditor, then synthesize() outside the graph.

The planner splits the question; each sub-question is dispatched to its own ``solve`` branch
(LangGraph ``Send``) and the branches run **concurrently**. A ``solve`` branch runs the three
sub-agents that used to be separate nodes — router → retriever → verifier — as a plain Python
retry loop, emitting their trace entries so all five personas stay visible. The retriever
itself fans out one LLM call per page. The graph ends at the ``auditor``; the final answer is
written by ``synthesize``/``synthesize_stream`` *after* the graph (called from ``core``) so it
can be streamed token by token. Every LLM call passes through ``_LLM_SEM`` so no more than
``STELLA_FANOUT`` (default 4) requests hit the shared vLLM at once.

The deterministic wiki reads (``lookup``/``open_page``/``trace_links``) do all retrieval — the
LLMs only route and write prose. The shared vLLM has no native tool-calling, hence the
JSON-per-turn (ReAct-style) contract.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

from src.stella_kb import config
from src.stella_kb.llm import chat, chat_stream

from ..io import (
    cross_ref_partners,
    extract_page_items,
    lookup,
    open_page,
    query_ledger,
    route_lookup,
    trace_links,
)
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
_CROSS_PAIR_CAP = 3   # max PDF↔Excel cross-ref partner pages added per sub-question (over-retrieval guard)


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


# --------------------------------------------------------------------------- planner
def planner_node(state: AgentState) -> AgentState:
    """Break the question into a minimal list of sub-questions (each fans out to a branch)."""
    user = f"INDEX:\n{state['index_md']}\n\nQuestion: {state['question']}\n\nReturn the plan JSON."
    act, _ = _ask(PLANNER, user, 400)
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
        "trace": [_rec(-1, 0, "planner", "plan", f"{len(plan)} sub-Q", (act or {}).get("thought", ""))],
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


def _route(sub: dict, tried: list, index: dict, index_md: str,
           wiki_dir: str | None = None) -> tuple[list, dict | None, str]:
    """Pick the wiki page(s) for one sub-question; on a trace sub-Q expand along the DAG.

    First attempt only, try the **curated routing table** (``routes.yaml``): if a hint term maps
    to existing pages, use them and **skip the router LLM** — the latency win (one fewer LLM call
    per sub-question, and a curated-correct page avoids a ``gap``→retry round). On a retry
    (``tried`` non-empty) the shortcut is bypassed so we don't re-pick the same pages; the LLM
    router runs with the ``avoid`` list instead. The trace-mode DAG expansion runs for both.
    """
    hints = sub.get("hint_terms") or []
    top_k = max(1, config.agent_router_top_k())  # max pages opened per round (recall vs cost)
    picks: list = []
    rthought = ""
    if not tried:  # only short-circuit the first try; retries must diverge via the LLM router
        picks = route_lookup(hints, index, wiki_dir)
        if picks:
            rthought = "routes.yaml 직결 — 라우터 LLM 생략"

    if not picks:  # no curated hit (or this is a retry) → fall back to the LLM router
        lookups = "\n\n".join(lookup(index, t) for t in hints) if hints else "(no hint terms)"
        avoid = (
            (
                f"\nAlready tried for this sub-question and found insufficient — pick a "
                f"DIFFERENT page unless re-reading is clearly justified: {tried}"
            )
            if tried
            else ""
        )
        user = (
            f"INDEX:\n{index_md}\n\nLookup results:\n{lookups}\n\n"
            f"Sub-question: {sub['ask']}{avoid}\n\n"
            f"답이 여러 페이지에 흩어져 있거나 비교·교차검증이면 관련 페이지를 한 번에 "
            f"최대 {top_k}개까지 고르세요(가능성 높은 순). Return the pages JSON."
        )
        act, _ = _ask(ROUTER, user, 400)
        valid = set(index.get("pages", {}).keys())
        by_norm = {re.sub(r"\s+", "", v).casefold(): v for v in valid}
        seen: set = set()
        for raw in (act or {}).get("pages") or []:  # tolerate [[wikilink]]/quote forms
            m = _match_page(raw, valid, by_norm)
            if m and m not in seen:  # resolve + dedup; drop hallucinations
                seen.add(m)
                picks.append(m)
        rthought = (act or {}).get("thought", "")

    picks = picks[:top_k]  # cap the router's page picks (recall/cost knob)
    path = None            # populated below for trace-mode sub-questions; intentionally None here
    if sub.get("mode") == "trace" and picks:
        direction = sub.get("direction", "down")
        chain = trace_links(index, picks[0], direction=direction)
        chain_pages = [c["sheet"] for c in chain if c["has_page"] and c["sheet"] not in picks][:5]
        path = {"ask": sub["ask"], "direction": direction, "start": picks[0], "chain": chain}
        picks = picks + chain_pages

    # cross-check pairing: attach each picked page's PDF↔Excel partner so a reconcile question
    # opens both the FDD report page and its Excel source. Capped, deduped — off by default.
    if config.agent_cross_ref_pairing() and picks:
        extra: list = []
        for p in picks:
            extra += cross_ref_partners(index, p, cap=2)
        extra = [p for p in dict.fromkeys(extra) if p not in picks][:_CROSS_PAIR_CAP]
        if extra:
            picks = picks + extra
            rthought = (rthought + " +cross-ref").strip()
    return picks, path, rthought


def _retrieve(ask: str, pages: list, wiki_dir: str | None = None,
              hint_terms: list | None = None) -> tuple[list, str]:
    """Open the pages and extract evidence — one LLM call PER PAGE, fanned out concurrently.

    When ``config.agent_deterministic_retrieve()`` is on, each page is first parsed with the
    deterministic ``extract_page_items`` (its ``value [cell]`` table); on a hit that page's
    evidence is taken verbatim and its LLM call is **skipped** (the latency win). Pages with no
    parseable table fall back to the LLM extractor below. Off by default → pure-LLM, unchanged.
    """
    if not pages:
        return [], "(no pages selected)"
    texts = {p: open_page(p, wiki_dir) for p in pages}

    det: dict[str, list] = {}
    if config.agent_deterministic_retrieve():
        for page in pages:
            items = extract_page_items(texts[page], hint_terms)
            if items:
                det[page] = [{"page": page, "cell": it["cell"], "term": it["term"],
                              "period": str(it.get("period", "")), "value": str(it["value"]),
                              "ask": ask} for it in items]
    llm_pages = [p for p in pages if p not in det]

    def extract(page: str) -> list:
        user = f"Sub-question: {ask}\n\nWIKI PAGE:\n{texts[page]}\n\nReturn the evidence JSON."
        # Pages now carry full raw grids (matrices/dense tables), so a multi-cell answer can
        # need many evidence rows — give the extractor headroom so its JSON isn't truncated.
        act, _ = _ask(system=RETRIEVER, user=user, max_tokens=1500)
        out = []
        for e in (act or {}).get("evidence") or []:
            if not isinstance(e, dict):
                continue
            cell = str(e.get("cell", ""))
            celltok = cell.split("!")[-1]  # soft guard: the cell must be on THIS page
            if celltok and _cell_on_page(celltok, texts[page]):
                out.append(
                    {
                        "page": e.get("page", "") or page,
                        "cell": cell,
                        "term": e.get("term", ""),
                        "period": str(e.get("period", "")),
                        "value": str(e.get("value", "")),
                        "ask": ask,
                    }
                )
        return out

    # branch threads spawn this pool too — the _LLM_SEM (not the worker count) is the real
    # cap, so total live threads can exceed _FANOUT but in-flight LLM requests never do.
    per_page = []
    if llm_pages:
        with ThreadPoolExecutor(max_workers=min(_FANOUT, len(llm_pages))) as ex:
            per_page = list(ex.map(extract, llm_pages))
    ev = [e for page_ev in per_page for e in page_ev] + [e for evlist in det.values() for e in evlist]
    det_note = f" ({len(det)} page(s) deterministic)" if det else ""
    return ev, f"{len(ev)} fact(s) from {pages}{det_note}"


def _ledger_evidence(picks: list, sub: dict, wiki_dir: str | None = None) -> list:
    """For any ``*_거래내역`` page picked, run the deterministic ledger filter+sum.

    Transaction rows aren't on the wiki page (the time-series parse drops them), so the LLM
    retriever finds nothing there. This pulls them from the ledger sidecar and sums 출금 by
    적요 keyword (the sub-question's ``hint_terms``) deterministically — exact, cell-cited."""
    kws = [k for k in (sub.get("hint_terms") or []) if k]
    out: list = []
    for p in picks:
        if isinstance(p, str) and p.endswith("_거래내역"):
            out += query_ledger(p, kws, sub.get("ask", ""), wiki_dir=wiki_dir)
    return out


def _verify(sub: dict, ev: list, path: dict | None) -> tuple[str, str]:
    """Judge whether the sub-question is answered. A traced chain is accepted as-is."""
    if sub.get("mode") == "trace" and path and path.get("chain"):
        return "ok", "provenance chain traced"
    ev_txt = (
        "\n".join(f"- {e['term']}{_per(e)} = {e['value']}  ({e['cell']}, {e['page']})" for e in ev) or "(no evidence)"
    )
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
    index_md = state["index_md"]  # the router prompt needs the ToC
    wiki_dir = state.get("wiki_dir")  # per-request dataset dir (None → process default)
    idx = state.get("sub_idx", 0)
    max_steps = max(1, state.get("max_steps", 3))  # per-branch read budget (initial + retries)
    verbose = state.get("verbose")

    tried: list = []
    evidence: list = []
    paths: list = []
    trace: list = []
    seen: set = set()  # (page, cell) already captured — dedup retries
    reads = seq = 0
    while True:
        picks, path, rthought = _route(sub, tried, index, index_md, wiki_dir)
        trace.append(_rec(idx, seq, "router", "route", ", ".join(picks) or "(none)", rthought))
        seq += 1
        if path:
            paths.append(path)

        ev, summary = _retrieve(sub["ask"], picks, wiki_dir, sub.get("hint_terms"))
        led = _ledger_evidence(picks, sub, wiki_dir)  # deterministic 거래내역 filter+sum (rows not on page)
        if led:
            ev = ev + led
            summary += f"  +ledger({len(led)})"
        for e in ev:  # keep first sighting of each fact on this branch
            # Dedup by the full fact grain, not just (page, cell): PDF pages tag EVERY row with
            # the same page-level tag (e.g. [FDD8]), so a bare (page, cell) key collapses an
            # entire time series (FY24…FY29) to one row. Include period+term so distinct rows
            # that legitimately share a tag survive, while true re-reads still dedup.
            key = (e["page"], e["cell"], e.get("period", ""), e.get("term", ""))
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
        # Key by the full fact grain (page, cell, period, term), not just (page, cell). On PDF
        # pages every row shares one page-level tag, so a coarse key falsely flags FY24 vs FY25
        # of the SAME series as "the same cell cited twice". With period+term, only a genuine
        # collision (one identical data point feeding two opposed asks) fires this caveat.
        cell_asks.setdefault((e["page"], e["cell"], e.get("period", ""), e.get("term", "")), set()).add(e["ask"])
        ask_ev.setdefault(e["ask"], []).append(e)
    for (page, cell, _period, _term), asks in cell_asks.items():
        if len(asks) >= 2:
            ref = cell if "!" in cell else f"{page}!{cell}"  # cell may already carry the sheet
            caveats.append(
                f"동일 출처 셀 {ref} 이(가) 서로 다른 하위질문의 근거로 중복 사용됨 "
                f"({' / '.join(sorted(asks))}). 두 항목을 서로 다른 자료로 대사한 것이 아니므로 "
                f"'일치/동일하다'라고 단정하지 말 것 — 한쪽 출처는 실제로 확인되지 않았을 수 있음."
            )

    # 2) a sub-question whose evidence is ENTIRELY from PDF/report pages. The PDF *is* the
    #    source for "what does the report show?" questions, so this is NOT a reason to decline —
    #    answer with the report figure, but flag that it's report-based (unit/asof/definition
    #    may differ from the Excel source). It only becomes "확인 불가" if there is no evidence
    #    at all (handled by check 3).
    for ask, items in ask_ev.items():
        if items and all(_is_pdf_page(pages.get(e["page"], {})) for e in items):
            caveats.append(
                f"하위질문 '{ask}' 의 근거는 리포트(PDF) 페이지에서 나옴 — 리포트 기준 수치이므로 "
                f"그대로 답하되, 엑셀 원본과 단위·정의·기준일이 다를 수 있음을 한 줄로 덧붙일 것. "
                f"(원본 미확인을 이유로 '확인 불가' 처리하지 말 것.)"
            )

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
        "trace": [_rec(_SYNTH_ORDER - 1, 0, "auditor", "audit", f"{len(caveats)} caveat(s)", thought)],
    }


# ----------------------------------------------------------------------- synthesizer
# The synthesizer is NOT a graph node — the graph ends at the auditor. Synthesis runs after the
# graph so the final answer can be **streamed** token by token (LangGraph would only hand back the
# node's state once the whole answer is already generated). Both the buffered ``run`` path and the
# SSE ``stream_run`` path build the same prompt via ``_synth_user`` and call the same model; the
# prompt now returns plain Korean prose (no JSON wrapper), so its tokens stream straight to the user.

_SYNTH_FALLBACK = "evidence는 수집되었으나 최종 답변 정리에 실패했습니다."


def _synth_user(state: AgentState) -> str:
    """Build the synthesizer user prompt from the merged evidence, traced paths, and audit caveats."""
    ev = state.get("evidence", [])
    ev_txt = (
        "\n".join(f"- [{e['ask']}] {e['term']}{_per(e)} = {e['value']}  ({e['cell']}, page {e['page']})" for e in ev)
        or "(no evidence gathered)"
    )

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
    caveat_block = (
        ("\n\n감사 경고(AUDIT — 반드시 반영, 무시 금지):\n" + "\n".join(f"- {c}" for c in caveats)) if caveats else ""
    )

    return (
        f"Question: {state['question']}\n\nEvidence collected from the wiki:\n{ev_txt}"
        f"{path_block}{caveat_block}\n\n최종 답변을 작성하세요."
    )


def _synth_trace() -> dict:
    """The synthesizer's trace record (sorts last via ``_SYNTH_ORDER``)."""
    return _rec(_SYNTH_ORDER, 0, "synthesizer", "answer", "", "")


def synthesize(state: AgentState) -> tuple[str, dict]:
    """Buffered final answer: ``(answer_text, trace_record)``. Used by the non-streaming
    ``run``/``arun`` and the eval. Prose out — no JSON parsing, so nothing to salvage."""
    with _LLM_SEM:
        raw = chat(
            [{"role": "system", "content": SYNTHESIZER},
             {"role": "user", "content": _synth_user(state)}],
            max_tokens=900, timeout=120.0,
        )
    return (raw or "").strip() or _SYNTH_FALLBACK, _synth_trace()


def synthesize_stream(state: AgentState) -> Iterator[str]:
    """Stream the final answer as text deltas (token level). Same prompt/model as
    :func:`synthesize`; the SSE path joins these into the canonical answer. Holds ``_LLM_SEM``
    for the one in-flight request, like every other model call."""
    with _LLM_SEM:
        emitted = False
        for delta in chat_stream(
            [{"role": "system", "content": SYNTHESIZER},
             {"role": "user", "content": _synth_user(state)}],
            max_tokens=900, timeout=120.0,
        ):
            emitted = True
            yield delta
        if not emitted:  # empty generation → at least surface the fallback
            yield _SYNTH_FALLBACK
