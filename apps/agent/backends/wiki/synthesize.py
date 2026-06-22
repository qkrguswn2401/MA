"""Synthesizer — the final Korean answer, written *after* the graph (the graph ends at the
auditor). Run outside the graph so the answer can be **streamed** token by token: LangGraph
would only hand back a node's state once the whole answer is already generated. Both the buffered
``run`` path and the SSE ``stream_run`` path build the same prompt via ``_synth_user`` and call
the same model; the prompt returns plain Korean prose (no JSON), so its tokens stream to the user.
"""

from __future__ import annotations

from collections.abc import Iterator

from src.stella_kb.llm import chat, chat_stream

from . import engine
from .engine import SYNTHESIZER, _per, _rec
from .state import AgentState

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
    return _rec(engine._SYNTH_ORDER, 0, "synthesizer", "answer", "", "")


def synthesize(state: AgentState) -> tuple[str, dict]:
    """Buffered final answer: ``(answer_text, trace_record)``. Used by the non-streaming
    ``run``/``arun`` and the eval. Prose out — no JSON parsing, so nothing to salvage."""
    with engine._LLM_SEM:
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
    with engine._LLM_SEM:
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
