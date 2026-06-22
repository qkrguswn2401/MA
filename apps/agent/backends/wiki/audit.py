"""Auditor node — deterministic cross-evidence audit between the solve barrier and the
synthesizer. Rule-based (no LLM), so it can't hallucinate; it only appends caveats the
synthesizer must honor."""

from __future__ import annotations

from .engine import _SYNTH_ORDER, _rec
from .state import AgentState


def auditor_node(state: AgentState, index: dict) -> AgentState:
    """Deterministic cross-evidence audit between the solve barrier and the synthesizer.

    The per-branch verifier only asks "did THIS sub-question get evidence?" — it never sees
    the merged set, so it can't catch a reconciliation that cited the *same* cell for two
    opposed quantities (fabricated agreement), or a planned sub-question that found nothing.
    These are exactly the over-claiming failures. The checks are rule-based (no LLM) so they
    can't hallucinate and won't touch the answers that are already right; they only append
    caveats the synthesizer must honor. (``index`` is kept for the build.py call signature.)"""
    ev = state.get("evidence", [])
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

    # 2) a planned sub-question that collected no evidence at all → that part is unverifiable.
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
