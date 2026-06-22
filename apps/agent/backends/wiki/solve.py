"""Solve branch — resolve ONE sub-question end to end: router → retriever → verifier, with
retries. Runs as a parallel ``Send`` branch; the retriever itself fans out one LLM call per page.
Returns only the ``operator.add`` channels, which LangGraph merges at the barrier before the
synthesizer. The deterministic wiki reads (``lookup``/``open_page``/``trace_links``) do all
retrieval — the LLMs only route and extract."""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

from src.stella_kb import config

from ...retrieval import (
    cross_ref_partners,
    extract_page_items,
    lookup,
    open_page,
    query_ledger,
    route_lookup,
    trace_links,
)
from . import engine
from .engine import RETRIEVER, ROUTER, VERIFIER, _cell_on_page, _per, _rec
from .state import AgentState

_CROSS_PAIR_CAP = 3   # max PDF↔Excel cross-ref partner pages added per sub-question (over-retrieval guard)


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
        act, _ = engine._ask(ROUTER, user, 400)
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
        act, _ = engine._ask(system=RETRIEVER, user=user, max_tokens=1500)
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
        with ThreadPoolExecutor(max_workers=min(engine._FANOUT, len(llm_pages))) as ex:
            per_page = list(ex.map(extract, llm_pages))
    ev = [e for page_ev in per_page for e in page_ev]
    ev += [e for evlist in det.values() for e in evlist]
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
    act, _ = engine._ask(VERIFIER, user, 300)
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
