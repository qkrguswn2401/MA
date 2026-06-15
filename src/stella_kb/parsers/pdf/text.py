"""pymupdf + pdfplumber 텍스트 추출 — vision describe 의 **참조 텍스트** 소스.

vision 경로는 페이지 PNG 와 함께 이 텍스트를 모델에 같이 넘긴다(철자·숫자 authoritative).
text-strategy PDF 면 이 결과만으로도 충분(무료·결정론). pymupdf 본문 + pdfplumber 표.
"""
from __future__ import annotations

import logging
from pathlib import Path

import fitz  # pymupdf
import pdfplumber

from ...config import max_table_pages
from .state import SourceAbbrev, SourcePage

log = logging.getLogger("stella_kb.parsers.pdf.text")


def _format_table(rows: list[list[str | None]], idx: int) -> str | None:
    cleaned: list[str] = []
    for row in rows:
        cells = [(c or "").strip().replace("\n", " ") for c in row]
        if any(cells):
            cleaned.append(" | ".join(cells))
    if not cleaned:
        return None
    return f"[표 {idx}]\n" + "\n".join(cleaned)


def parse_pdf(path: Path, abbrev: SourceAbbrev = "ETC") -> list[SourcePage]:
    """pymupdf 본문 + pdfplumber 표. 표는 같은 실제 페이지 텍스트 뒤에 부착.

    인용 ``[ABBR p.X]`` 가 실제 PDF 페이지를 유지하도록 표를 별도 페이지로 쪼개지 않는다.
    표 추출은 비싸므로 ``MNA_PARSE_MAX_TABLE_PAGES`` (default 80) 페이지까지만.
    """
    path = Path(path)
    pages: list[SourcePage] = []
    doc = fitz.open(str(path))
    table_cap = max_table_pages()
    try:
        with pdfplumber.open(str(path)) as plumber_doc:
            total = len(doc)
            if total > table_cap:
                log.warning("pdfplumber 표 추출: %d 페이지 > cap %d — 초과분 표 skip (file=%s)",
                            total, table_cap, path.name)
            for i, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                tables_md: list[str] = []
                raw_tables = (
                    plumber_doc.pages[i - 1].extract_tables() or [] if i <= table_cap else []
                )
                for t_idx, raw in enumerate(raw_tables, start=1):
                    md = _format_table(raw, t_idx)
                    if md:
                        tables_md.append(md)
                if tables_md:
                    text = (text + "\n\n" if text else "") + "\n\n".join(tables_md)
                if text:
                    pages.append(SourcePage(
                        abbrev=abbrev, file=path, page=i, text=text,
                        word_page_start=i, word_page_end=i,
                    ))
    finally:
        doc.close()
    return pages
