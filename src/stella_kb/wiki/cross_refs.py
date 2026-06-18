"""Directed PDF↔Excel cross-references — ``derives_from`` (PDF→Excel) + ``cited_by`` (Excel→PDF).

The FDD report (PDF) is a downstream exhibit that **cites** the Excel valuation model (the source
of truth), so the edge is directed **PDF → Excel**. Strictly **bipartite**: never PDF↔PDF, never
Excel↔Excel (Excel pages already carry the formula DAG via depends_on/feeds_into).

Decision — connect ``F`` (pdf) → ``E`` (excel) iff:
  Gate 0  SAME ENTITY    — F's FDD deck must be about the Excel model's entity (Centroid/Stella).
                           A CAESAR (Celadon) or LIFE (KDB생명) deck is a *different company*, so
                           it never links no matter how many aliases (WACC, 관리수수료, …) overlap.
  Tier A  FUND IDENTITY  — F names fund X (the ``pdf_pages._xrefs`` bridge) and E is fund X's
                           Excel source. Self-entity-gating (only Centroid funds match). Strong.
  Tier B  SPECIFIC METRIC— F and E share an alias term carried by ≤ K pages (a specific line item,
                           not a generic category like 관리보수 on 12 funds). Entity-gated.
  Tier C  LLM JUDGE      — OPTIONAL, off by default: a whitelist-guarded, content-cached judge
                           that confirms/rejects the *ambiguous* Tier-B candidates (paraphrase /
                           value-match) the rules can't settle. It may only confirm a candidate
                           already on the deterministic shortlist — no invented edges.

Every edge records a ``via`` reason (fund / metric:<term> / llm) for audit. Runs post-merge in
``pdf_pages`` where both page sets coexist; idempotent (clears prior edges first).
"""
from __future__ import annotations

import re

_ENTITY_TERMS = ("센트로이드", "centroid", "stella")  # the Excel model's entity (Project Stella = Centroid)
_DECK_RE = re.compile(r"\[([^\]]+)\]")


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s).casefold()


def _is_pdf(meta: dict) -> bool:
    return meta.get("source") == "PDF"


def _deck_of(page_name: str, documents: dict) -> str | None:
    """The deck a PDF page belongs to: the ``[DECK]`` tag in its name, else the sole document."""
    m = _DECK_RE.search(page_name)
    if m and m.group(1) in documents:
        return m.group(1)
    return next(iter(documents)) if len(documents) == 1 else (m.group(1) if m else None)


def _matching_decks(documents: dict, entity_terms: tuple[str, ...]) -> set[str]:
    """Decks whose document node is about the Excel entity (Gate 0 whitelist)."""
    terms = [t.casefold() for t in entity_terms]
    out = set()
    for doc, node in (documents or {}).items():
        blob = f"{node.get('title', '')} {node.get('description', '')}".casefold()
        if any(t in blob for t in terms):
            out.add(doc)
    return out


def build_cross_refs(index: dict, entity_terms: tuple[str, ...] = _ENTITY_TERMS,
                     k: int = 4, cap: int = 6, judge=None) -> dict:
    """Compute directed PDF→Excel cross-refs into ``index`` (mutates). Returns a report.

    ``k`` = max pages a shared alias term may span to still count as a *specific* metric (Tier B).
    ``cap`` = max derives_from per PDF page. ``judge`` (optional) = a callable
    ``(pdf_name, excel_name, index) -> bool`` confirming an ambiguous candidate (Tier C)."""
    from .pdf_pages import _xrefs  # deferred: pdf_pages calls us in its __main__

    pages = index.get("pages", {})
    documents = index.get("documents", {})
    ai = index.get("alias_index", {})

    # idempotency: clear any prior edges
    for p in pages.values():
        p.pop("derives_from", None)
        p.pop("cited_by", None)

    pages_per_term = {t: {h["page"] for h in hits} for t, hits in ai.items()}
    matching = _matching_decks(documents, entity_terms)
    excel = {nm for nm, e in pages.items() if not _is_pdf(e)}

    n_edges = judged = 0
    for fname, f in pages.items():
        if not _is_pdf(f):
            continue
        via: dict[str, str] = {}  # excel page -> reason

        # Tier A — fund identity (self-entity-gating; works even if the deck isn't matched)
        for e in _xrefs(f, index):
            if e in excel:
                via.setdefault(e, "fund")

        # Tier B — specific shared metric, but ONLY for same-entity decks (Gate 0)
        if _deck_of(fname, documents) in matching:
            f_aliases = f.get("aliases") or []
            f_items = f.get("items") or []
            f_terms = {_norm(t) for t in f_aliases}
            f_terms |= {_norm(it.get("label") or "") for it in f_items}
            for t in f_terms:
                hosts = pages_per_term.get(t)
                if not hosts or len(hosts) > k:        # skip unknown / generic (category) terms
                    continue
                cand = [p for p in hosts if p in excel]
                if not cand:
                    continue
                if judge is not None:                  # Tier C: confirm the ambiguous candidate
                    for e in cand:
                        if e not in via:
                            judged += 1
                            if judge(fname, e, index):
                                via[e] = f"llm:{t}"
                else:
                    for e in cand:
                        via.setdefault(e, f"metric:{t}")

        if not via:
            continue
        derives = sorted(via)[:cap]
        f["derives_from"] = [{"page": e, "via": via[e]} for e in derives]
        for e in derives:                              # inverse view on the Excel side
            pages[e].setdefault("cited_by", []).append(fname)
            n_edges += 1

    # stable order for the cited_by lists
    for e in excel:
        if pages[e].get("cited_by"):
            pages[e]["cited_by"] = sorted(set(pages[e]["cited_by"]))
    return {
        "edges": n_edges,
        "judged": judged,
        "pdf_with_links": sum(1 for p in pages.values() if p.get("derives_from")),
        "excel_cited": sum(1 for nm in excel if pages[nm].get("cited_by")),
    }


_JUDGE_SYS = (
    "당신은 M&A 가치평가 지식베이스에서 FDD 보고서(PDF) 페이지와 엑셀 원천 페이지가 같은 항목을 "
    "다루는지 판정합니다. FDD 페이지의 수치가 이 엑셀 페이지에서 도출된 것이면 derives=true. "
    "단순히 비슷한 용어가 겹치는 정도면 false. 같은 회사·같은 항목이어야 합니다. "
    'JSON만 출력: {"derives": true|false, "why": "<한 문장>"}'
)


def make_llm_judge(wiki_dir: str | None = None):
    """A cached, whitelist-guarded Tier-C judge ``(pdf_name, excel_name, index) -> bool``.

    Only ever called on a deterministic Tier-B candidate (so it can only *confirm* an existing
    shortlist entry — never invent an edge), and the LLM call is content-cached so a rebuild stays
    deterministic. A judge failure → ``False`` (conservative: don't link)."""
    from ..config import pdf_structure_cache
    from ..llm import _json_span, cached_chat

    def judge(pdf_name: str, excel_name: str, index: dict) -> bool:
        f = index["pages"].get(pdf_name, {})
        e = index["pages"].get(excel_name, {})
        f_txt = f.get("desc") or f.get("title") or pdf_name
        e_items = ", ".join((it.get("label") or "") for it in (e.get("items") or [])[:20])
        user = (f"FDD 보고서 페이지: {pdf_name}\n요약: {f_txt}\n\n"
                f"엑셀 원천 페이지: {excel_name}\n항목: {e_items}\n\nJSON:")
        try:
            raw = cached_chat(
                [{"role": "system", "content": _JUDGE_SYS}, {"role": "user", "content": user}],
                cache_dir=pdf_structure_cache(), max_tokens=200, timeout=120)
            obj = _json_span(raw, "{", "}")
            return bool(isinstance(obj, dict) and obj.get("derives"))
        except Exception:  # noqa: BLE001 — a judge failure must not break the build; don't link
            return False

    return judge
