"""Stage 5: build the navigable index / table-of-contents over the wiki pages.

This is the entry point an agent reads first to decide *which page to open* — the
"key" of the vectorless KB. Deterministic (no LLM): it joins the sheet-name taxonomy,
the parsed metadata, and the formula DAG into two artifacts:

  - ``data/wiki/index.json``  — machine-readable: a section->group->page tree, a
    per-page entry (title/case/unit/aliases/items/links), and an **alias index**
    (``term -> [{page, cell}]``) that resolves a KO/EN query term to page+cell with no
    embeddings (the words->node resolver).
  - ``data/wiki/INDEX.md``    — the human/agent table of contents: sections with
    ``[[page]]`` wikilinks and key aliases.

Classification is by sheet-name **tokens**, not divider position: ``_raw.xlsx`` is
missing the ` Biz Plan>>`/`Fin.Model>>` divider tabs, so position-walking would mis-group
the fund/macro sheets. Tokens (``장표``, ``_비용``/``_거래내역``/``_관리보수``, ``EIU``,
``4.1``/``4.2``) are the schema (see docs/workbook_analysis.md §0).

Usage (from repo root, venv active; needs data/parsed + data/wiki/pages):
    python -m src.stella_kb.wiki.index
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import openpyxl

from ..config import (
    wiki_index_json,
    wiki_index_md,
    wiki_pages_dir,
    wiki_parsed_dir,
    wiki_workbook,
)
from ..graph.extract import build_dependency_graph
from .compile import all_items, page_currency, sheet_links, usable_tables, value_series

WORKBOOK = wiki_workbook()
PARSED_DIR = wiki_parsed_dir()
PAGES_DIR = wiki_pages_dir()
OUT_JSON = wiki_index_json()
OUT_MD = wiki_index_md()

# The wiki index covers **only** sheets present in the canonical `_raw` workbook. Engine
# sheets that live solely in the full `(Updated)` workbook (DCF, AUM Projection, the carry
# sheet, …) belong to the graph paradigm, not here — so the index never ingests them.


def classify(name: str) -> dict:
    """Map a sheet name to its logical {section, group, kind, case} via name tokens."""
    if "EIU" in name:
        return {"section": "거시 가정 (Macro · EIU)", "group": "EIU", "kind": "macro", "case": None}
    if "장표" in name or name in {"Football Chart", "Bridge"}:
        case = "MGT" if name.endswith("_MGT") else "DTT" if name.endswith("_DTT") else None
        fam = name.split("장표")[0].strip() or name
        return {"section": "PPT 장표 (Exhibits)", "group": fam, "kind": "exhibit", "case": case}
    for suf, kind in (
        ("_비용", "비용 (costs)"),
        ("_거래내역", "거래내역 (ledger)"),
        ("_관리보수", "관리보수 (fee schedule)"),
    ):
        if name.endswith(suf):
            return {"section": "Biz Plan (per-fund)", "group": name[: -len(suf)], "kind": kind, "case": None}
    if name == "성과보수, 배당금":
        return {
            "section": "Fin.Model (밸류에이션 엔진)",
            "group": "성과보수·배당금",
            "kind": "fee model",
            "case": None,
        }
    if name == "IRR":
        return {"section": "Biz Plan (per-fund)", "group": "IRR", "kind": "return model", "case": None}
    if name.startswith("4.1") or name == "PL_FY24(A)":
        return {
            "section": "BSPL (재무제표)",
            "group": "4.1 Centroid Investment Partners",
            "kind": "statement",
            "case": None,
        }
    if name.startswith("4.2"):
        return {"section": "BSPL (재무제표)", "group": "4.2 Centroid Management", "kind": "statement", "case": None}
    return {"section": "기타 (Other)", "group": name, "kind": None, "case": None}


def _norm(term: str) -> str:
    return re.sub(r"\s+", "", term).casefold()


def _desc(sheet: str) -> str | None:
    """First sentence of the page's grounded "What this is" prose, for the ToC.

    Reuses the already-LLM-written Korean blurb in ``data/wiki/pages/<sheet>.md`` so the
    index gains a discriminating one-liner per sheet with no extra LLM call. Titles alone
    collide ("Income Statement" ×2, "DCF Summary" ×2); this disambiguates by purpose.
    """
    p = PAGES_DIR / f"{sheet}.md"
    if not p.exists():
        return None
    m = re.search(r"## What this is\s*\n+(.+?)(?:\n##|\Z)", p.read_text(encoding="utf-8"), re.S)
    if not m:
        return None
    para = m.group(1).strip()
    if not para or para.startswith("_("):  # scaffold / "prose unavailable"
        return None
    first = re.split(r"(?<=다)\.\s", para)[0].strip().rstrip(".") + "."
    first = re.sub(rf"^'{re.escape(sheet)}'\s*시트", "이 시트", first)  # drop redundant sheet name
    return first


def load_all_values() -> dict[str, dict]:
    """``{sheet: {coord: value}}`` for the whole `_raw` workbook in one open (for data status).

    The keys are exactly the `_raw` sheet names — also used as the membership set that keeps
    the index `_raw`-only.
    """
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True, read_only=True)
    out = {
        ws.title: {c.coordinate: c.value for row in ws.iter_rows() for c in row if c.value is not None}
        for ws in wb.worksheets
    }
    wb.close()
    return out


def _period(tables: list) -> str | None:
    """Year range string across all of a sheet's table axes, e.g. ``2021–2029``."""
    years = [v for t in tables for v in (t.get("year_axis") or {}).get("columns", {}).values()
             if isinstance(v, int)]
    return f"{min(years)}–{max(years)}" if years else None


def _data_status(tables: list, vals: dict) -> str:
    """Routing signal: are the line-item cells real numbers, errors, or empty? (all tables)."""
    real = ref = 0
    for t in tables:
        axis_cols = (t.get("year_axis") or {}).get("columns") or {}
        for it in t.get("line_items") or []:
            for _, v, _ in value_series(vals, it, axis_cols):
                if isinstance(v, bool):
                    continue
                if isinstance(v, (int, float)):
                    real += 1
                elif isinstance(v, str) and v.startswith("#"):  # #REF!, #DIV/0!, ...
                    ref += 1
    if real == 0 and ref > 0:
        return "none"
    if ref > 0:
        return "partial"
    return "full" if real > 0 else "—"


def build_sheet_dag(path: str) -> dict[str, dict[str, list[str]]]:
    """The workbook's **sheet-level formula DAG** — the agent's provenance substrate.

    Collapses the cell-level dependency edges to ``{sheet: {depends_on, feeds_into}}``.
    Built from `_raw` (the index is `_raw`-only): edges still surface references that `_raw`
    cells make *to* engine sheets (e.g. ``='AUM Projection'!…``) as edge targets — that's
    information physically present in `_raw` — but no engine sheet appears as a source node.
    Tracing through the full Fin.Model engine is the graph paradigm's job, not the wiki's.
    Cross-sheet only; cycles kept (Excel has bidirectional refs — the BFS de-dups).
    """
    dg = build_dependency_graph(path)
    links: dict[str, dict[str, set]] = {}
    for prec, dep in dg.edges:
        sp, sd = prec.rsplit("!", 1)[0], dep.rsplit("!", 1)[0]
        if sp == sd:
            continue
        links.setdefault(sd, {"depends_on": set(), "feeds_into": set()})
        links.setdefault(sp, {"depends_on": set(), "feeds_into": set()})
        links[sd]["depends_on"].add(sp)
        links[sp]["feeds_into"].add(sd)
    return {s: {k: sorted(v) for k, v in d.items()} for s, d in links.items()}


def build_index() -> dict:
    all_vals = load_all_values()
    raw_sheets = set(all_vals)  # the canonical `_raw` sheet set — the index covers only these

    # Ingest only parsed JSONs whose sheet exists in `_raw`. Any full-workbook-only artifact
    # (e.g. a stale `성과보수, 배당금.json`) is skipped, so nothing outside `_raw` leaks in.
    parsed: dict[str, dict] = {}
    for p in sorted(PARSED_DIR.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("sheet", p.stem) in raw_sheets:
            parsed[p.stem] = d
    pages_on_disk = {p.stem for p in PAGES_DIR.glob("*.md")}
    links = sheet_links()

    pages: dict[str, dict] = {}
    tree: dict[str, dict[str, list]] = {}
    aliases: dict[str, list] = {}

    for sheet, data in parsed.items():
        cls = classify(sheet)
        meta = data.get("meta") or {}
        vals = all_vals.get(sheet, {})
        tables = usable_tables(data, vals)
        items = all_items(data, vals)

        # per-page alias set + feed the global alias index (term -> page+cell)
        page_aliases: list[str] = []
        for it in items:
            cell = it.get("label_cell")
            terms = [it.get("label"), it.get("label_ko"), it.get("label_en"), *(it.get("aliases") or [])]
            for t in terms:
                if not t:
                    continue
                if t not in page_aliases:
                    page_aliases.append(t)
                key = _norm(t)
                bucket = aliases.setdefault(key, [])
                if not any(h["page"] == sheet and h["cell"] == cell for h in bucket):
                    bucket.append({"page": sheet, "cell": cell, "term": t})

        link = links.get(sheet, {})
        entry = {
            "sheet": sheet,
            "title": meta.get("title") or sheet,
            "desc": _desc(sheet),
            "section": cls["section"],
            "group": cls["group"],
            "kind": cls["kind"],
            "case": meta.get("case") or cls["case"],
            "unit": page_currency(meta, items, sheet)[0],
            "period": _period(tables),
            "data_status": _data_status(tables, vals),
            "n_items": len(items),
            "has_page": sheet in pages_on_disk,
            "aliases": page_aliases,
            "items": [
                {
                    "label": it.get("label"),
                    "ko": it.get("label_ko"),
                    "cell": it.get("label_cell"),
                    "role": it.get("role"),
                }
                for it in items
            ],
            "depends_on": [s for s in link.get("depends_on", []) if s in parsed],
            "feeds_into": [s for s in link.get("feeds_into", []) if s in parsed],
        }
        pages[sheet] = entry
        tree.setdefault(cls["section"], {}).setdefault(cls["group"], []).append(sheet)

    sheet_dag = build_sheet_dag(WORKBOOK)
    return {"tree": tree, "pages": pages, "alias_index": aliases, "sheet_dag": sheet_dag}


def render_md(index: dict) -> str:
    pages, tree = index["pages"], index["tree"]
    n_alias = len(index["alias_index"])
    out = [
        "# Project Stella — Wiki Index (ToC)",
        "",
        f"> {len(pages)} pages · {n_alias} alias terms · vectorless lookup: resolve a "
        "KO/EN term via the alias index → open the `[[page]]` → follow links.",
        "",
    ]

    from .pdf_pages import SECTION as PDF_SECTION
    documents = index.get("documents") or {}

    for section in sorted(tree):
        # PDF/FDD section → two-layer per-deck index: a detailed document description (upper)
        # + that deck's table of contents (lower), so the router picks the report, then the page.
        if section == PDF_SECTION and documents:
            out += [f"## {section} — 문서별", ""]
            for doc in sorted(documents):
                d = documents[doc]
                out += [f"### 📄 {doc} — {d.get('title', doc)} ({d.get('n_pages', 0)}p)", ""]
                if d.get("description"):
                    out += [f"> {d['description']}", ""]
                out += ["#### 목차", ""]
                for t in d.get("toc", []):
                    out.append(f"- **[[{t['page']}]]** — {t.get('title','')}")
                    if t.get("summary"):
                        out.append(f"  - {t['summary']}")
                out.append("")
            continue

        out += [f"## {section}", ""]
        for group in sorted(tree[section]):
            sheets = sorted(tree[section][group])
            if len(sheets) > 1 or group not in sheets:
                out.append(f"### {group}")
            for s in sheets:
                e = pages[s]
                meta = [e["kind"] or ""]
                if e["case"]:
                    meta.append(f"case {e['case']}")
                if e["period"]:
                    meta.append(e["period"])
                if e["unit"]:
                    meta.append(e["unit"])
                out.append(f"- **[[{s}]]** — {e['title']}")
                if e.get("desc"):
                    out.append(f"  - {e['desc']}")
                out.append(f"  - {' · '.join(m for m in meta if m)}")
                rel = []
                if e["depends_on"]:
                    rel.append("← " + ", ".join(f"[[{d}]]" for d in e["depends_on"][:4]))
                if e["feeds_into"]:
                    rel.append("→ " + ", ".join(f"[[{d}]]" for d in e["feeds_into"][:4]))
                if rel:
                    out.append(f"  - links: {'  '.join(rel)}")
                if e.get("xref"):  # FDD → Excel source pages (cross-check hop)
                    out.append("  - 교차검증(엑셀 원천) → "
                               + ", ".join(f"[[{x}]]" for x in e["xref"][:5]))
            out.append("")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    index = build_index()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(index), encoding="utf-8")

    from .ledger import write_ledgers  # row sidecars for *_거래내역 (transaction ledgers)
    led = write_ledgers(WORKBOOK, [s for s in index["pages"] if s.endswith("_거래내역")],
                        OUT_JSON.parent / "ledgers")
    print(f"  ledgers: {sum(led.values())} rows across {len(led)} sheet(s) -> "
          f"{OUT_JSON.parent / 'ledgers'}")

    n_pages = len(index["pages"])
    n_alias = len(index["alias_index"])
    n_sections = len(index["tree"])
    n_dag = len(index["sheet_dag"])
    print(f"index: {n_pages} pages, {n_sections} sections, {n_alias} alias terms, " f"{n_dag}-sheet provenance DAG")
    print(f"  -> {OUT_JSON}")
    print(f"  -> {OUT_MD}")
