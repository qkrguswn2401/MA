"""Lift the parsed wiki schema into the Tier-1 *semantic* property graph.

The cell DAG from :mod:`extract` is the substrate — 4k formula cells, too fine to reason
over. This module builds the graph an analyst actually queries (~hundreds of nodes) from
``data/parsed/*.json`` (the grounded line-items the wiki pipeline already produced), and
collapses the cell→cell dependency edges *up* to metric→metric by following each
line-item's ``value_row`` anchor.

Two metric grains are produced:
  - ``Metric``  — one per (sheet, label): the queryable unit, anchored to real cells.
  - ``Concept`` — one per distinct label across the workbook (e.g. one "관리보수" the same
    line item appears under every fund). Each ``Metric`` ``INSTANCE_OF`` its ``Concept``.

Structural nodes: ``Sheet`` / ``Section`` / ``Fund`` / ``Entity`` / ``Period``.
Edges: ``DEFINED_IN`` (Metric→Sheet) · ``PART_OF`` (Sheet→Section) · ``BELONGS_TO``
(Metric→Fund/Entity) · ``COVERS`` (Sheet→Period) · ``INSTANCE_OF`` (Metric→Concept) ·
``DEPENDS_ON`` (Metric→Metric, collapsed from the cell DAG).

Usage (repo root, venv active; needs data/parsed/ + the workbook):
    python -m src.stella_kb.graph.lift            # build, report counts, export
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import networkx as nx

from .. import DATA_DIR, WORKBOOK
from .extract import build_dependency_graph
from ..wiki.index import classify  # section/group/kind/case by sheet-name tokens

PARSED_DIR = DATA_DIR / "v0.1" / "parsed"
OUT_JSON = DATA_DIR / "graph" / "stella_semantic.json"

_COL_ROW = re.compile(r"^([A-Z]+)(\d+)$")


def _norm(label: str) -> str:
    return re.sub(r"\s+", "", label or "").casefold()


def _fund_entity(sheet: str, cls: dict) -> tuple[str | None, str | None]:
    """(fund, entity) for a sheet, or (None, None). Funds are Biz Plan groups; the two
    BSPL groups are the Centroid entities."""
    if cls["section"] == "Biz Plan (per-fund)" and cls["group"] != "IRR":
        return cls["group"], None
    if cls["section"] == "BSPL (재무제표)":
        return None, cls["group"]
    return None, None


def build(parsed_dir: Path = PARSED_DIR, workbook: str = WORKBOOK) -> nx.DiGraph:
    parsed = {p.stem: json.loads(p.read_text(encoding="utf-8"))
              for p in sorted(parsed_dir.glob("*.json"))}
    g = nx.DiGraph()

    # (sheet, value_row) -> metric_id, so we can map a cell back to the metric it belongs to
    cell_row_to_metric: dict[tuple[str, int], str] = {}

    for sheet, data in parsed.items():
        cls = classify(sheet)
        meta = data.get("meta") or {}
        axis = (data.get("year_axis") or {}).get("columns") or {}
        fund, entity = _fund_entity(sheet, cls)

        sheet_id, section_id = f"Sheet:{sheet}", f"Section:{cls['section']}"
        g.add_node(sheet_id, type="Sheet", title=meta.get("title") or sheet,
                   case=meta.get("case") or cls["case"], unit=meta.get("unit"))
        if section_id not in g:
            g.add_node(section_id, type="Section", label=cls["section"])
        g.add_edge(sheet_id, section_id, type="PART_OF")

        if fund:
            fid = f"Fund:{fund}"
            g.add_nodes_from([(fid, {"type": "Fund", "label": fund})])
        if entity:
            eid = f"Entity:{entity}"
            g.add_nodes_from([(eid, {"type": "Entity", "label": entity})])

        # periods this sheet covers
        for yr in {v for v in axis.values() if isinstance(v, int)}:
            pid = f"Period:{yr}"
            if pid not in g:
                g.add_node(pid, type="Period", year=yr)
            g.add_edge(sheet_id, pid, type="COVERS")

        for it in data.get("line_items") or []:
            label = it.get("label_ko") or it.get("label") or it.get("label_en")
            if not label:
                continue
            mid = f"Metric:{sheet}::{_norm(label)}"
            if mid not in g:
                g.add_node(mid, type="Metric", label=label,
                           label_en=it.get("label_en"), role=it.get("role"),
                           sheet=sheet, cell=it.get("label_cell"),
                           value_row=it.get("value_row"),
                           aliases=it.get("aliases") or [])
            g.add_edge(mid, sheet_id, type="DEFINED_IN")
            if fund:
                g.add_edge(mid, f"Fund:{fund}", type="BELONGS_TO")
            if entity:
                g.add_edge(mid, f"Entity:{entity}", type="BELONGS_TO")

            # concept layer: same label across the workbook -> one shared concept
            cid = f"Concept:{_norm(label)}"
            if cid not in g:
                g.add_node(cid, type="Concept", label=label)
            g.add_edge(mid, cid, type="INSTANCE_OF")

            vr = it.get("value_row")
            if isinstance(vr, int):
                cell_row_to_metric[(sheet, vr)] = mid

    _collapse_dependencies(g, workbook, cell_row_to_metric)
    return g


def _collapse_dependencies(g: nx.DiGraph, workbook: str,
                           cell_row_to_metric: dict[tuple[str, int], str]) -> None:
    """Walk the cell DAG: for each metric's value cells, trace precedents (through
    unanchored intermediate cells) until another metric's cell is reached, and add a
    ``DEPENDS_ON`` edge metric→metric. Transitive so real flows aren't lost between
    distant anchor rows."""
    dg = build_dependency_graph(workbook)
    cells = dg.cells                                  # cell_id -> CellInfo (formula cells)

    def cell_metric(cell_id: str) -> str | None:
        sheet, _, ref = cell_id.partition("!")
        m = _COL_ROW.match(ref)
        return cell_row_to_metric.get((sheet, int(m.group(2)))) if m else None

    # which formula cells belong to a metric (start points for the walk)
    metric_cells: dict[str, list[str]] = collections.defaultdict(list)
    for cid in cells:
        mid = cell_metric(cid)
        if mid:
            metric_cells[mid].append(cid)

    weights: dict[tuple[str, str], int] = collections.Counter()
    for mid, starts in metric_cells.items():
        for start in starts:
            seen, stack = set(), list(cells[start].precedents)
            while stack:
                p = stack.pop()
                if p in seen:
                    continue
                seen.add(p)
                src = cell_metric(p)
                if src and src != mid:
                    weights[(src, mid)] += 1        # src drives mid
                    continue                        # stop at the first anchored ancestor
                if p in cells:                      # unanchored intermediate -> keep walking
                    stack.extend(cells[p].precedents)

    for (src, dst), w in weights.items():
        g.add_edge(src, dst, type="DEPENDS_ON", weight=w)


def report(g: nx.DiGraph) -> str:
    ntypes = collections.Counter(d["type"] for _, d in g.nodes(data=True))
    etypes = collections.Counter(d["type"] for *_, d in g.edges(data=True))
    lines = [f"semantic graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges",
             "  nodes: " + ", ".join(f"{t}={c}" for t, c in ntypes.most_common()),
             "  edges: " + ", ".join(f"{t}={c}" for t, c in etypes.most_common())]
    return "\n".join(lines)


if __name__ == "__main__":
    g = build()
    print(report(g))
    OUT_JSON.write_text(
        json.dumps(nx.node_link_data(g, edges="links"), ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  -> {OUT_JSON}")
