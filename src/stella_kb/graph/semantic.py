"""Lift the cell-level dependency DAG into a semantic property graph.

Two graphs are produced from the same source:

* **cell graph** — the raw backbone: one node per formula cell, ``DEPENDS_ON`` edges
  straight from extract.py. Large (~14k nodes) but exact.
* **semantic graph** — aggregated to the grain a knowledge base is queried at:
  ``Section`` / ``Sheet`` / ``Fund`` / ``Entity`` nodes, with sheet→sheet ``DEPENDS_ON``
  edges weighted by how many cell dependencies cross them.

Sheet classification is **rule-based** (v1): sheets are sectioned by the ``>>`` divider
tabs in workbook order, then funds/entities are recognised by name. This is precise but
brittle to renames — an LLM labelling pass can replace `classify_sheets` later (see the
OpenKB reference) without touching the graph-building code.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import openpyxl

from .extract import DependencyGraph, build_dependency_graph
from .ids import nid, sheet_of
from .metrics import attach_metrics


# --- sheet classification -------------------------------------------------------------

@dataclass
class SheetClass:
    sheet: str
    section: str            # PPT | Fin.Model | Biz Plan | BSPL | (raw divider name)
    kind: str               # exhibit | engine | fund | statement | divider
    fund: str | None = None     # owning fund, for Biz Plan sheets
    entity: str | None = None   # owning entity, for BSPL sheets


_SECTIONS = {  # divider tab (contains '>>') -> canonical section name
    "PPT": "PPT",
    "Fin.Model": "Fin.Model",
    "Biz Plan": "Biz Plan",
    "BSPL": "BSPL",
}
_SECTION_KIND = {
    "PPT": "exhibit",
    "Fin.Model": "engine",
    "Biz Plan": "fund",
    "BSPL": "statement",
}


def _section_for_divider(name: str) -> str | None:
    """Map a divider tab name to a canonical top-level section, if it is one."""
    for key, canon in _SECTIONS.items():
        if key in name:
            return canon
    return None


def _fund_of(sheet: str) -> str | None:
    """Biz Plan sheets are ``<fund>_<비용|거래내역|관리보수>``; return ``<fund>``."""
    if sheet == "IRR":
        return None  # aggregator, not a fund
    return sheet.split("_", 1)[0] if "_" in sheet else sheet


def classify_sheets(path: str) -> dict[str, SheetClass]:
    """Walk sheets in workbook order, assigning section/kind/fund/entity by the ``>>``
    dividers and name patterns. Returns ``{sheet_name: SheetClass}`` for non-divider sheets.
    """
    wb = openpyxl.load_workbook(path, read_only=True)
    names = wb.sheetnames
    wb.close()

    out: dict[str, SheetClass] = {}
    section = "PPT"
    entity: str | None = None  # tracked within BSPL via ">>4.x" sub-dividers

    for name in names:
        if ">>" in name:
            canon = _section_for_divider(name)
            if canon:
                section = canon
                entity = None
            elif name.lstrip(">").startswith(("4.1", "4.2")):
                entity = name.lstrip("> ")  # e.g. "4.1센트로이드인베스트먼트파트너스"
            continue

        kind = _SECTION_KIND.get(section, "engine")
        sc = SheetClass(sheet=name, section=section, kind=kind)
        if section == "Biz Plan":
            sc.fund = _fund_of(name)
            if sc.fund is None:
                sc.kind = "engine"  # IRR aggregator
        elif section == "BSPL":
            sc.entity = entity
        out[name] = sc
    return out


# --- graph construction ---------------------------------------------------------------

def build_cell_graph(dg: DependencyGraph) -> nx.DiGraph:
    """Raw cell-level ``DEPENDS_ON`` backbone. Precedent cells that hold no formula of
    their own (pure inputs) still appear as nodes via the edge endpoints."""
    g = nx.DiGraph()
    for cid, info in dg.cells.items():
        g.add_node(cid, type="Cell", sheet=sheet_of(cid),
                   formula=info.formula, value=info.value)
    for prec, dep in dg.edges:
        if prec not in g:
            g.add_node(prec, type="Cell", sheet=sheet_of(prec))  # input cell
        g.add_edge(prec, dep, type="DEPENDS_ON")
    return g


def build_semantic_graph(path: str, dg: DependencyGraph | None = None,
                         with_metrics: bool = True) -> nx.DiGraph:
    """Aggregate the cell DAG to Section/Sheet/Fund/Entity nodes, then lift named Metrics.

    With ``with_metrics`` (default), `metrics.attach_metrics` adds the financial line items
    (AUM, fees, EV, WACC, FCFF) as `Metric` nodes wired to their source cells and to each
    other via ``DRIVES``/``ASSUMPTION_OF`` — see metrics.py.
    """
    if dg is None:
        dg = build_dependency_graph(path)
    classes = classify_sheets(path)

    g = nx.DiGraph()

    def link(sheet_id: str, node_type: str, name: str, rel: str) -> None:
        """Connect a sheet to a Section/Fund/Entity node, creating the target if new."""
        target = nid(node_type, name)
        if target not in g:
            g.add_node(target, type=node_type)
        g.add_edge(sheet_id, target, type=rel)

    # structural nodes + PART_OF / BELONGS_TO edges
    for sheet, sc in classes.items():
        sid = nid("Sheet", sheet)
        g.add_node(sid, type="Sheet", kind=sc.kind, section=sc.section)
        link(sid, "Section", sc.section, "PART_OF")
        if sc.fund:
            link(sid, "Fund", sc.fund, "BELONGS_TO")
        if sc.entity:
            link(sid, "Entity", sc.entity, "BELONGS_TO")

    # aggregate cell dependencies into weighted sheet->sheet DEPENDS_ON edges
    for prec, dep in dg.edges:
        ps, ds = nid("Sheet", sheet_of(prec)), nid("Sheet", sheet_of(dep))
        if ps == ds or ps not in g or ds not in g:
            continue  # drop intra-sheet edges and refs to divider/unknown sheets
        if g.has_edge(ps, ds):
            g[ps][ds]["weight"] += 1
        else:
            g.add_edge(ps, ds, type="DEPENDS_ON", weight=1)

    if with_metrics:
        attach_metrics(g, path)
    return g


def export(g: nx.DiGraph, path: str) -> None:
    """Write the graph as node-link JSON (``.json``) or GraphML (``.graphml``).

    JSON is the default/recommended form: it round-trips Korean labels, datetimes (via
    ``default=str``), and the list/None attrs on Metric nodes that GraphML rejects. GraphML
    needs scalar attrs only, so it flattens those first.
    """
    if path.endswith(".json"):
        import json
        data = nx.node_link_data(g, edges="edges")  # explicit key silences the nx FutureWarning
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    else:
        h = g.copy()
        for _, d in h.nodes(data=True):
            for k, v in list(d.items()):
                if v is None:
                    del d[k]
                elif isinstance(v, (list, tuple)):
                    d[k] = ", ".join(map(str, v))
                elif not isinstance(v, (str, int, float, bool)):
                    d[k] = str(v)
        nx.write_graphml(h, path)


if __name__ == "__main__":
    from .. import FULL_WORKBOOK, DATA_DIR  # graph layer needs the full 63-sheet model

    dg = build_dependency_graph(FULL_WORKBOOK)
    sg = build_semantic_graph(FULL_WORKBOOK, dg)
    print(f"semantic graph: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")
    for t in ("Section", "Entity", "Fund", "Metric", "Period"):
        members = [n for n, d in sg.nodes(data=True) if d.get("type") == t]
        print(f"  {t}: {len(members)}  {members[:6]}")
    edge_kinds = {}
    for *_, d in sg.edges(data=True):
        edge_kinds[d.get("type")] = edge_kinds.get(d.get("type"), 0) + 1
    print("edge types:", edge_kinds)
    print("top sheet->sheet dependencies:")
    top = sorted((d["weight"], u, v) for u, v, d in sg.edges(data=True)
                 if d.get("type") == "DEPENDS_ON" and "weight" in d)
    for w, u, v in top[-8:]:
        print(f"  {u} -> {v}  (x{w})")

    out = str(DATA_DIR / "graph" / "stella_graph.json")
    export(sg, out)
    print(f"\nexported -> {out}")
