"""Curated builder for the per-fund **성과보수·배당금 (GP carry & distribution)** wiki page.

The `성과보수, 배당금` sheet is a native engine sheet that lives **only in the full
``(Updated)`` workbook** — it is absent from the canonical ``_raw`` WORKBOOK, so the normal
``dump_md → parse_llm → compile`` pipeline never produces a page for it. Its layout is also
hostile to the generic parser: six fund blocks laid out **side by side** (제2호=B열대,
옐로씨=Q열대, 제5호=AC열대, 제7호=AR열대, 제7-1호=BG열대, 제8호=BU열대), each block with a
**different value-column offset**, and several scenario re-computations stacked vertically.

So this module hand-anchors the headline figures (the metrics.py pattern: a curated cell
table, every value tied to an exact, verified cell) and emits the two artifacts the wiki/
index pipeline expects:

  - ``data/parsed/성과보수, 배당금.json`` — feeds ``index.py`` (page entry + alias index).
  - ``data/wiki/pages/성과보수, 배당금.md`` — what the agent's ``open_page`` reads.

Run after the main wiki build, then re-run ``index.py`` to register it:
    python -m src.stella_kb.wiki.carry
    python -m src.stella_kb.wiki.index
"""

from __future__ import annotations

import json
from pathlib import Path

import openpyxl

from .. import FULL_WORKBOOK  # the full workbook (engine sheets dropped from `_raw`)

SHEET = "성과보수, 배당금"
PARSED_OUT = Path("data/parsed") / f"{SHEET}.json"
PAGE_OUT = Path("data/wiki/pages") / f"{SHEET}.md"

# Per-fund anchors. `val` is the summary-block value column; in every block the headline
# rows are 성과보수 (carry) MGT=row4 / DTT=row6 and 재산분배액 (distribution) MGT=row7 /
# DTT=row9. `exit` pins the MGT-case Exit assumptions (row 5), whose columns differ per
# block. `dtt_mult` is the one DTT Exit input that diverges (제8호 only).
FUNDS: list[dict] = [
    {"name": "제2호 바이아웃", "alias": "제2호", "block": "센트로이드제2호바이아웃", "val": "E", "sumlbl": "C4",
     "exit": {"ebitda": "J5", "extra": ("Exit Sales", "K5"), "mult": "L5",
              "date": "M5", "hurdle": "N5", "excess": "O5"}},
    {"name": "옐로씨 제1호", "alias": "옐로씨", "block": "센트로이드옐로씨제1호", "val": "S", "sumlbl": "Q4",
     "exit": {"ebitda": "V5", "extra": None, "mult": "W5",
              "date": "X5", "hurdle": "Y5", "excess": "Z5"}},
    {"name": "제5호 바이아웃", "alias": "제5호", "block": "센트로이드제5호바이아웃", "val": "AF", "sumlbl": "AC4",
     "exit": {"ebitda": "AJ5", "extra": ("EV/Hole", "AK5"), "mult": "AK5",
              "date": "AL5", "hurdle": "AM5", "excess": "AN5"}},
    {"name": "제7호 바이아웃", "alias": "제7호", "block": "센트로이드제7호바이아웃", "val": "AU", "sumlbl": "AR4",
     "exit": {"ebitda": "AY5", "extra": None, "mult": "AZ5",
              "date": "BA5", "hurdle": "BB5", "excess": "BC5"}},
    {"name": "제7-1호 바이아웃", "alias": "제7-1호", "block": "센트로이드제7-1호바이아웃", "val": "BI", "sumlbl": "BG4",
     "exit": {"ebitda": "BM5", "extra": None, "mult": "BN5",
              "date": "BO5", "hurdle": "BP5", "excess": "BQ5"}},
    {"name": "제8호 바이아웃", "alias": "제8호", "block": "센트로이드제8호바이아웃", "val": "BW", "sumlbl": "BU4",
     "exit": {"ebitda": "CB5", "extra": None, "mult": "CC5", "dtt_mult": "CC6",
              "date": "CD5", "hurdle": "CE5", "excess": "CF5"}},
]

_COMMON_ALIASES = ["성과보수", "carry", "carried interest", "performance fee",
                   "초과수익", "재산분배액", "GP 성과보수"]


def _carry_cells(f: dict) -> dict:
    """The summary-block headline cells for one fund."""
    v = f["val"]
    return {"carry_mgt": f"{v}4", "carry_dtt": f"{v}6",
            "dist_mgt": f"{v}7", "dist_dtt": f"{v}9"}


def _fmt(v: object) -> str:
    """Render a cell value for the page: thousands-grouped numbers, ISO dates, raw else."""
    if isinstance(v, bool) or v is None:
        return "" if v is None else str(v)
    if isinstance(v, (int, float)):
        return f"{v:,.4f}".rstrip("0").rstrip(".") if v % 1 else f"{int(v):,}"
    s = str(v)
    return s[:10] if s[4:5] == "-" and s[:4].isdigit() else s  # 2026-12-31 00:00:00 -> date


def _read(ws, cell: str) -> object:
    return ws[cell].value


def build_parsed() -> dict:
    """The parsed schema — line_items only register the page + aliases in the index."""
    items = []
    for f in FUNDS:
        c = _carry_cells(f)
        items.append({
            "label": f"{f['block']} 성과보수",
            "label_en": f"{f['alias']} Performance Fee",
            "label_ko": "성과보수",
            "aliases": [f["alias"], f["block"], f["name"], *_COMMON_ALIASES],
            # label_cell points at the summary 성과보수 label cell so value_series finds the
            # value to its right and index.py marks data_status = full.
            "label_cell": f["sumlbl"],
            "value_row": 4,
            "role": "revenue",
        })
    return {"meta": {"title": "성과보수·배당금 (GP Performance Fee & Distribution)",
                     "unit": "KRWm", "case": None},
            "year_axis": {"row": None, "columns": {}},
            "line_items": items,
            "sheet": SHEET}


def build_page(ws) -> str:
    carry = {f["alias"]: (_read(ws, _carry_cells(f)["carry_mgt"]),
                          _read(ws, _carry_cells(f)["carry_dtt"])) for f in FUNDS}
    earners = [f["name"] for f in FUNDS
               if isinstance(carry[f["alias"]][0], (int, float)) and carry[f["alias"]][0]]
    earners_txt = " · ".join(earners) if earners else "없음"

    lines = [
        "---",
        f"sheet: {SHEET}",
        "section: Fin.Model (밸류에이션 엔진)",
        "unit: KRWm",
        "aliases: [성과보수, Performance Fee, carry, carried interest, GP 성과보수, 초과수익, "
        "Excess Return, 재산분배액, Property Distribution, 배당금, Exit EBITDA, Exit Multiple, "
        "Hurdle rate, 기준수익률, Excessive Rate, 제2호, 옐로씨, 제5호, 제7호, 제7-1호, 제8호, "
        "MGT Case, DTT Case]",
        "---",
        "",
        "# 성과보수·배당금 (GP Performance Fee & Distribution)",
        "",
        "## What this is",
        "",
        "이 시트는 펀드별 Exit 가정(Exit EBITDA·Multiple·시점)으로부터 GP가 받는 성과보수(carry)와 "
        "재산분배액을 산정하는 엔진입니다. 각 펀드 블록은 가로로 나열되며 블록마다 값 컬럼 오프셋이 "
        "다릅니다(제2호=E열, 옐로씨=S열, 제5호=AF열, 제7호=AU열, 제7-1호=BI열, 제8호=BW열). 상단 "
        "요약 4행의 성과보수가 모델로 흘러가는 헤드라인 carry이며, MGT 케이스는 4행, DTT 케이스는 "
        f"6행입니다. **여섯 펀드 중 {earners_txt}만 hurdle을 초과해 carry가 발생하고, 나머지는 "
        "0입니다.** 제7호 MGT 성과보수는 Operating Revenue(Revenue 장표 #1)의 집계 성과보수와 "
        "일치합니다. 차이나1호·제3호는 이 시트에 carry 블록이 없습니다(청산/회수 완료).",
        "",
        "## 성과보수 (Performance Fee / Carry)",
        "",
        "| Fund | KO | role | cell (MGT) | MGT | cell (DTT) | DTT |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in FUNDS:
        c = _carry_cells(f)
        lines.append(f"| {f['name']} | 성과보수 | revenue | `{c['carry_mgt']}` | "
                     f"{_fmt(_read(ws, c['carry_mgt']))} | `{c['carry_dtt']}` | "
                     f"{_fmt(_read(ws, c['carry_dtt']))} |")

    lines += ["", "## 재산분배액 (Property Distribution to GP)", "",
              "| Fund | KO | role | cell (MGT) | MGT | cell (DTT) | DTT |",
              "|---|---|---|---|---|---|---|"]
    for f in FUNDS:
        c = _carry_cells(f)
        lines.append(f"| {f['name']} | 재산분배액 | revenue | `{c['dist_mgt']}` | "
                     f"{_fmt(_read(ws, c['dist_mgt']))} | `{c['dist_dtt']}` | "
                     f"{_fmt(_read(ws, c['dist_dtt']))} |")

    lines += ["", "## Exit 가정 (Exit Assumptions, MGT Case 5행)", "",
              "| Fund | Exit EBITDA/기준 | cell | Multiple | cell | Exit 시점 | cell | "
              "Hurdle | cell | Excess | cell |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in FUNDS:
        e = f["exit"]
        eb = _fmt(_read(ws, e["ebitda"]))
        if e.get("extra"):
            lbl, xc = e["extra"]
            eb += f" ({lbl} {_fmt(_read(ws, xc))} `{xc}`)"
        mult = _fmt(_read(ws, e["mult"]))
        if e.get("dtt_mult"):
            mult += f" (DTT {_fmt(_read(ws, e['dtt_mult']))} `{e['dtt_mult']}`)"
        lines.append(
            f"| {f['name']} | {eb} | `{e['ebitda']}` | {mult} | `{e['mult']}` | "
            f"{_fmt(_read(ws, e['date']))} | `{e['date']}` | "
            f"{_fmt(_read(ws, e['hurdle']))} | `{e['hurdle']}` | "
            f"{_fmt(_read(ws, e['excess']))} | `{e['excess']}` |")

    lines += ["", "## Links", "", "- **Depends on:** —",
              "- **Feeds into:** [[Revenue 장표 #1]]", ""]
    return "\n".join(lines)


def build() -> None:
    wb = openpyxl.load_workbook(FULL_WORKBOOK, data_only=True, read_only=True)
    ws = wb[SHEET]
    PARSED_OUT.parent.mkdir(parents=True, exist_ok=True)
    PAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    PARSED_OUT.write_text(json.dumps(build_parsed(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
    PAGE_OUT.write_text(build_page(ws), encoding="utf-8")
    wb.close()


if __name__ == "__main__":
    build()
    print(f"wrote {PARSED_OUT}")
    print(f"wrote {PAGE_OUT}")
