"""Query-compounding: persist a valuable agent answer back onto its wiki page.

The third LLM-wiki idea after lint + incremental ingest — a query result becomes a permanent
page section instead of dying in chat history, so knowledge accrues. Here the answer is built
**into the main page** (a ``## Q&A (compounded)`` appendix), per the chosen design.

The hard part is that ``compile`` regenerates every page from the workbook on each rebuild —
a section written straight into the markdown would be wiped. So the **source of truth is a
sidecar** ``<wiki>/qa/<page>.jsonl`` (append-only), and the page's Q&A section is *rendered
from* it. Both sides use this module: ``compile`` re-renders the section on every rebuild (so
it survives), and the agent's write path re-renders the live page immediately (so it shows
without waiting for a rebuild).

Provenance guard (the reason this is safe-ish despite mixing LLM prose into a grounded page):
an answer is only persisted if it carries real cell evidence (:func:`is_grounded`), and the
target page is the one its evidence most cites. A stray ``[[link]]`` an answer might invent is
still caught downstream by ``lint`` (``broken_link`` scans page bodies).

This module is workbook-free and stdlib-only, so both the build pipeline (``compile``) and the
query agent (``apps/agent``) can import it without pulling in openpyxl.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

SECTION = "## Q&A (compounded)"
_INTRO = ("> 에이전트가 생성한 누적 Q&A — 근거 셀이 있는 답변만 보존됩니다 "
          "(원천: 질의 시점의 그래프/페이지 근거).")


def _safe(page: str) -> str:
    return page.replace("/", "_")


def qa_dir(wiki_dir: str | Path) -> Path:
    return Path(wiki_dir) / "qa"


def qa_path(wiki_dir: str | Path, page: str) -> Path:
    return qa_dir(wiki_dir) / f"{_safe(page)}.jsonl"


# --------------------------------------------------------------------------- store

def load_qa(wiki_dir: str | Path, page: str) -> list[dict]:
    """Every persisted Q&A entry for one page (oldest first); ``[]`` if none. Tolerant of a
    corrupt line so one bad write never hides the rest."""
    path = qa_path(wiki_dir, page)
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def append_qa(wiki_dir: str | Path, page: str, entry: dict) -> None:
    """Append one entry to the page's sidecar (creating ``qa/`` as needed)."""
    path = qa_path(wiki_dir, page)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- grounding

def is_grounded(entry: dict) -> bool:
    """A persistable answer: a real answer string + at least one piece of cell evidence.

    The provenance gate — refuses to compound an answer that cites nothing (the failure mode
    that would launder a hallucination into a page read later as ground truth)."""
    ans = (entry.get("answer") or "").strip()
    if not ans or ans == "(no answer)":
        return False
    evidence = entry.get("evidence") or []
    return any(e.get("cell") for e in evidence)


# --------------------------------------------------------------------------- render

def _cells(entry: dict, target_page: str | None = None) -> list[str]:
    """Deduped cell refs for one entry, as ``page!cell`` (or bare ``cell`` on the target page)."""
    evidence = entry.get("evidence") or []
    out: list[str] = []
    for e in evidence:
        cell = e.get("cell")
        if not cell:
            continue
        pg = e.get("page")
        ref = f"{pg}!{cell}" if pg and pg != target_page else cell
        if ref not in out:
            out.append(ref)
    return out


def render_qa_section(entries: list[dict], target_page: str | None = None) -> str:
    """Markdown for the ``## Q&A (compounded)`` block (without trailing newline)."""
    lines = [SECTION, "", _INTRO, ""]
    for e in entries:
        q = " ".join((e.get("question") or "").split())
        a = " ".join((e.get("answer") or "").split())
        lines.append(f"- **Q:** {q}")
        lines.append(f"  - **A:** {a}")
        cells = _cells(e, target_page)
        if cells:
            lines.append("  - 근거: " + ", ".join(f"`{c}`" for c in cells))
        if e.get("created_at"):
            lines.append(f"  - _{e['created_at']}_")
    return "\n".join(lines)


def upsert_qa_section(page_md: str, entries: list[dict], target_page: str | None = None) -> str:
    """Return ``page_md`` with its Q&A section set to ``entries`` (rendered from the sidecar).

    The section is always the page's tail: an existing one is replaced wholesale, otherwise it's
    appended. Empty ``entries`` removes the section — so clearing the sidecar cleans the page.
    Idempotent: re-running with the same entries yields the same markdown (no duplication), which
    is what lets ``compile`` re-render it on every rebuild safely.
    """
    idx = page_md.find(SECTION)
    head = page_md[:idx] if idx != -1 else page_md
    head = head.rstrip()
    if not entries:
        return head + "\n"
    return head + "\n\n" + render_qa_section(entries, target_page) + "\n"


def new_entry(question: str, answer: str, evidence: list[dict]) -> dict:
    """Build a sidecar entry from an agent result (trims evidence to the fields we keep)."""
    ev = [
        {"page": e.get("page"), "cell": e.get("cell"), "term": e.get("term"), "value": e.get("value")}
        for e in (evidence or [])
        if e.get("cell")
    ]
    return {"question": question, "answer": answer, "evidence": ev,
            "created_at": date.today().isoformat()}


def target_page(evidence: list[dict]) -> str | None:
    """The page an answer most cites — where its compounded Q&A attaches."""
    from collections import Counter

    pages = (e.get("page") for e in (evidence or []) if e.get("page") and e.get("cell"))
    c = Counter(pages)
    return c.most_common(1)[0][0] if c else None
