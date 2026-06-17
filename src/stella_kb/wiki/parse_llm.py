"""LLM parse pass: a sheet's Markdown grid -> a grounded structural schema.

This is the *interpretation* half of the pipeline (the *extraction* half — cells,
values, formula edges — stays mechanical in ``extract.py``). It replaces the brittle
hand-curated layout logic in ``metrics.py`` (per-sheet ``fiscal_year_axis`` offsets,
hand-keyed anchor cells) with an LLM reading the 2D grid produced by
``dump_md.py``.

The contract follows CLAUDE.md's rules for LLM use:
  - The model **interprets structure** (which row is the year axis, which rows are
    line items, what each line means, KO/EN aliases). It returns **cell references**,
    never numbers — values are read from openpyxl, never transcribed by the model.
  - Every reference the model emits is **grounded** against the real workbook cells
    (OpenKB whitelist pattern, applied to parsing): a label must actually sit at the
    cell the model cites; an axis cell must actually hold a year. Ungrounded claims are
    dropped and recorded, never trusted.

Output per sheet -> ``data/parsed/<sheet>.json``:
    {meta:{title,unit,case}, year_axis:{row,columns:{COL:year}}, line_items:[...],
     grounding:{...}}

Usage (from repo root, venv active; needs the local vLLM up — see llm.py):
    python -m src.stella_kb.wiki.dump_md --all      # produce data/md/ first
    python -m src.stella_kb.wiki.parse_llm "DCF"    # parse one sheet
    python -m src.stella_kb.wiki.parse_llm --all    # parse every dumped sheet
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import re
from datetime import date, datetime
from pathlib import Path

import openpyxl
from openpyxl.utils import column_index_from_string

from .. import config
from ..llm import chat
from ..prompts import load as load_prompt

from ..config import wiki_md_dir, wiki_parsed_dir, wiki_workbook

WORKBOOK = wiki_workbook()

MD_DIR = wiki_md_dir()
OUT_DIR = wiki_parsed_dir()

_CELL = re.compile(r"^([A-Z]{1,3})(\d+)$")


# --------------------------------------------------------------------------- prompt

_SYSTEM = load_prompt("parse_system")


def _values_grid(md: str) -> str:
    """The values grid (+ merged ranges), without the long formulas appendix."""
    return md.split("\n## Formulas", 1)[0]


def _json_from(raw: str) -> dict | None:
    """Extract the first JSON object from a model reply (tolerates ```json fences)."""
    s = raw.strip()
    if "```" in s:
        parts = s.split("```")
        s = max(parts, key=len).lstrip("json").strip() if len(parts) >= 3 else s.strip("`")
    start, end = s.find("{"), s.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        return json.loads(s[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


# --------------------------------------------------------------------------- grounding

def load_values(sheet: str) -> dict[str, object]:
    """``{coordinate: cached value}`` for a sheet — the ground truth for validation."""
    wb = openpyxl.load_workbook(WORKBOOK, data_only=True, read_only=True)
    ws = wb[sheet]
    vals = {c.coordinate: c.value for row in ws.iter_rows() for c in row
            if c.value is not None}
    wb.close()
    return vals


def _norm(text: object) -> str:
    return re.sub(r"\s+", "", str(text)).casefold()


_MONTH3 = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"


def _year_at(value: object) -> object | None:
    """The fiscal period a header cell represents, if any.

    Returns an ``int`` year for annual columns, a verbatim label string for interim
    columns (``"6M24"``, ``"1H24"``) and the terminal markers, or ``None``. Recognizes both
    the full-year/date forms the valuation model uses (``2024``, a datetime) **and** the
    abbreviated fiscal labels this corpus uses — ``FY20``, ``Dec-20``, ``'24``, ``6M 24`` —
    which the 4-digit-only matcher silently dropped (leaving an empty year axis).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (datetime, date)):
        return value.year
    if isinstance(value, (int, float)) and 2000 <= int(value) <= 2040:
        return int(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if s in {"T.V.", "Terminal", "n/a"}:
        return s
    low = s.casefold()

    m = re.search(r"\b(20[0-3]\d)\b", low)              # full 4-digit year (e.g. 2024)
    if m:
        year = int(m.group(1))
    else:                                               # 2-digit year in a fiscal context
        for pat in (r"fy\s*'?(\d{2})\b",               # FY20, FY 20
                    rf"(?:{_MONTH3})[-\s]'?(\d{{2}})\b",  # Dec-20, Jun 24
                    r"[1-4]\s*[hq]\s*'?(\d{2})\b",       # 1H24, 2Q 24
                    r"\d\d?\s*m\s*'?(\d{2})\b",          # 6M 24, 12M24
                    r"'(\d{2})\b"):                      # '24
            m = re.search(pat, low)
            if m:
                year = 2000 + int(m.group(1))
                break
        else:
            return None

    # half/quarter/month-count columns are interim — keep a distinct verbatim label so they
    # don't collide with the annual column of the same year (e.g. FY24 vs 6M24).
    q = re.search(r"([1-4])\s*([hq])(?![a-z])", low) or re.search(r"(\d\d?)\s*(m)(?![a-z])", low)
    if q:
        return f"{q.group(1)}{q.group(2).upper()}{year % 100:02d}"
    return year


def _axis_columns(row: object, values: dict[str, object]) -> dict[str, object]:
    """Derive the ``{column: period}`` map for an axis row — the **largest contiguous run**
    of year-bearing cells in that row.

    Restricting to one contiguous run is what keeps a *neighbouring* table's header from
    bleeding in: e.g. the headcount sheet's 월평균 header row also carries the unrelated
    인당 관리보수 (per-head fee) year header further right, separated by blank/text columns.
    Grabbing every year-like cell mixed the fee columns into the average table; the largest
    run isolates the real axis (ties → leftmost). A normal single-block year axis is one run,
    so this is a no-op there.
    """
    if not row:
        return {}
    found = []  # (col_index, col_letter, period)
    for coord, val in values.items():
        m = _CELL.match(coord)
        if m and int(m.group(2)) == row:
            period = _year_at(val)
            if period is not None:
                found.append((column_index_from_string(m.group(1)), m.group(1), period))
    if not found:
        return {}
    found.sort()
    runs = [[found[0]]]
    for prev, cur in zip(found, found[1:]):
        if cur[0] == prev[0] + 1:
            runs[-1].append(cur)
        else:
            runs.append([cur])
    best = max(runs, key=len)  # largest run; max keeps the first (leftmost) on a tie
    return {letter: period for _, letter, period in best}


def _ground_items(line_items: list, row: object, values: dict[str, object]) -> tuple[list, list]:
    """Keep only line items whose cited cell really holds their label; return (kept, dropped).

    A label is grounded if the cited cell's text contains the label (or vice versa); a cell
    sitting in the axis row itself can't be a line item.
    """
    kept, dropped = [], []
    for item in line_items:
        cell = (item.get("label_cell") or "").upper()
        label = item.get("label") or item.get("label_en") or ""
        cell_text = values.get(cell)
        m = _CELL.match(cell)
        in_axis_row = bool(m) and row is not None and int(m.group(2)) == row
        ok = bool(m) and not in_axis_row and cell_text is not None and label and (
            _norm(label) in _norm(cell_text) or _norm(cell_text) in _norm(label))
        (kept if ok else dropped).append(
            item if ok else {"label": label, "label_cell": cell, "cell_text": cell_text})
    return kept, dropped


def ground(parsed: dict, values: dict[str, object]) -> dict:
    """Drop every claim that doesn't match a real cell; keep a report of what fell out.

    Mutates ``parsed`` in place and returns a grounding summary. The axis column->period map
    is *derived* from the cells in the LLM-named row (the model never counts columns, so
    off-by-one is impossible). Supports both the single-table shape (top-level ``year_axis``
    + ``line_items``) and the multi-table shape (a ``tables`` list, for sheets that stack
    several tables on different axes — e.g. a monthly roster plus an annual 월평균 table).
    """
    if parsed.get("tables"):
        per_table, dropped_all = [], []
        for t in parsed["tables"]:
            row = (t.get("year_axis") or {}).get("row")
            t["year_axis"] = {"row": row, "columns": _axis_columns(row, values)}
            t["line_items"], dropped = _ground_items(t.get("line_items") or [], row, values)
            dropped_all += dropped
            per_table.append({"title": t.get("title"), "kept_items": len(t["line_items"]),
                              "kept_axis_cols": len(t["year_axis"]["columns"])})
        return {"tables": per_table, "dropped_items": dropped_all,
                "kept_items": sum(r["kept_items"] for r in per_table),
                "kept_axis_cols": sum(r["kept_axis_cols"] for r in per_table)}

    row = (parsed.get("year_axis") or {}).get("row")
    parsed["year_axis"] = {"row": row, "columns": _axis_columns(row, values)}
    parsed["line_items"], dropped = _ground_items(parsed.get("line_items") or [], row, values)
    return {"axis_row_ok": bool(parsed["year_axis"]["columns"]),
            "kept_axis_cols": len(parsed["year_axis"]["columns"]),
            "kept_items": len(parsed["line_items"]), "dropped_items": dropped}


# --------------------------------------------------------------------------- driver

def parse_sheet(sheet: str, timeout: float = 600.0) -> dict:
    """Parse one sheet: read its md dump, call the LLM, ground the result."""
    md_path = MD_DIR / f"{sheet.replace('/', '_')}.md"
    if not md_path.exists():
        raise FileNotFoundError(f"{md_path} — run `python -m src.stella_kb.wiki.dump_md` first")
    grid = _values_grid(md_path.read_text(encoding="utf-8"))

    raw = chat(
        [{"role": "system", "content": _SYSTEM},
         {"role": "user", "content": f"Worksheet: {sheet!r}\n\n{grid}\n\nJSON:"}],
        max_tokens=8192, timeout=timeout,
    )
    parsed = _json_from(raw)
    if parsed is None:
        return {"sheet": sheet, "error": "no JSON parsed", "raw": raw[:2000]}

    parsed["sheet"] = sheet
    parsed["grounding"] = ground(parsed, load_values(sheet))
    return parsed


def _parse_and_write(name: str) -> str:
    """Parse one sheet and write its JSON; return a one-line status (thread worker)."""
    try:
        result = parse_sheet(name)
    except Exception as e:  # noqa: BLE001 — report per-sheet, keep going
        return f"!! {name}: {type(e).__name__}: {e}"
    if "error" in result:
        return f"!! {name}: {result['error']}"
    out = OUT_DIR / f"{name.replace('/', '_')}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    g = result["grounding"]
    return (f"{name}: {g['kept_items']} items, {g['kept_axis_cols']} axis cols "
            f"(dropped {len(g['dropped_items'])} items) -> {out.name}")


if __name__ == "__main__":
    import sys

    # Divider tabs dump to near-empty stubs; skip them in a full run.
    args = sys.argv[1:]
    if args and args[0] == "--all":
        sheets = [p.stem for p in sorted(MD_DIR.glob("*.md"))
                  if p.stat().st_size > 200]
    else:
        sheets = args or ["DCF"]

    # Concurrent requests — vLLM batches them, so throughput >> sequential. Bounded to
    # keep load light on the shared endpoint (override with STELLA_CONCURRENCY).
    workers = max(1, min(config.parse_concurrency(), len(sheets)))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"parsing {len(sheets)} sheets with {workers} concurrent workers ...")
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for msg in ex.map(_parse_and_write, sheets):
            print(msg, flush=True)
