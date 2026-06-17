"""Query layer: resolve a question to a Metric node, traverse the graph for evidence,
synthesize an answer. Follows the DCI loop (resolve -> inspect -> synthesize) with the
project's split: the **LLM only maps words->nodes and writes the final prose**; all
evidence comes from deterministic graph traversal, and every number carries its source cell.

    from src.stella_kb.graph.query import ask
    print(ask("What is the equity value and what drives it?"))

Run ``python -m src.stella_kb.graph.semantic`` once first to write ``data/stella_graph.json``.
"""

from __future__ import annotations

import json

import networkx as nx

from .. import DATA_DIR
from .. import llm
from ..prompts import load as load_prompt
from .ids import name_of

GRAPH_PATH = str(DATA_DIR / "graph" / "stella_graph.json")


# --- load ------------------------------------------------------------------------------

def load_graph(path: str = GRAPH_PATH) -> nx.DiGraph:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


# --- deterministic retrieval -----------------------------------------------------------

def _label(g: nx.DiGraph, n: str) -> str:
    return g.nodes[n].get("label", name_of(n))


def series(g: nx.DiGraph, mid: str) -> list[tuple]:
    """``[(year, value, cell), ...]`` from this metric's HAS_VALUE edges, year-sorted."""
    out = []
    for _, p, d in g.out_edges(mid, data=True):
        if d.get("type") == "HAS_VALUE":
            out.append((g.nodes[p].get("year"), d.get("value"), d.get("cell")))
    return sorted(out, key=lambda t: (1, 0) if isinstance(t[0], str) else (0, t[0]))


def source_cells(g: nx.DiGraph, mid: str) -> list[str]:
    return [name_of(v) if v.startswith("Sheet:") else v
            for _, v, d in g.out_edges(mid, data=True)
            if d.get("type") == "DEFINED_IN"]


def drivers(g: nx.DiGraph, mid: str, max_depth: int = 6) -> list[tuple]:
    """Reverse DRIVES/ASSUMPTION_OF walk: what feeds ``mid``. ``[(depth, label, rel), ...]``."""
    out, seen = [], set()

    def up(node, depth):
        if depth > max_depth:
            return
        for u, _, d in g.in_edges(node, data=True):
            rel = d.get("type")
            if rel in ("DRIVES", "ASSUMPTION_OF") and (u, node) not in seen:
                seen.add((u, node))
                out.append((depth, _label(g, u), rel))
                up(u, depth + 1)

    up(mid, 0)
    return out


def evidence(g: nx.DiGraph, mid: str) -> str:
    """A compact, grounded evidence block for one metric — the only thing the LLM sees."""
    n = g.nodes[mid]
    lines = [f"Metric: {n.get('label')} (id={name_of(mid)}, category={n.get('category')}"
             + (f", case={n.get('case')}" if n.get("case") else "") + ")"]
    if n.get("label_ko"):
        lines.append(f"Korean label: {n['label_ko']}")
    if n.get("value") is not None:
        cells = ", ".join(source_cells(g, mid)) or "—"
        if n.get("value_mgt") is not None:          # dual-case: show both MGT and DTT
            case = n.get("case") or "DTT"
            lines.append(f"Value ({case} case): {n['value']}  [cells: {cells}]")
            lines.append(f"Value (MGT case): {n['value_mgt']}  [cell: {n.get('cell_mgt', '—')}]")
        else:
            lines.append(f"Value: {n['value']}  [cells: {cells}]")
    s = series(g, mid)
    if s:
        lines.append("By period:")
        for yr, val, cell in s:
            vs = f"{val:,.1f}" if isinstance(val, (int, float)) else str(val)
            lines.append(f"  {yr}: {vs}  [{cell}]")
    dr = drivers(g, mid)
    if dr:
        lines.append("Drives/assumptions feeding it (depth · label · relation):")
        for depth, lbl, rel in dr:
            lines.append(f"  {'  ' * depth}- {lbl} ({rel})")
    return "\n".join(lines)


# --- resolve + answer ------------------------------------------------------------------

def resolve(question: str) -> str | None:
    """Question -> a single Metric node id (``Metric:...``) via the whitelist-guarded mapper."""
    r = llm.resolve_metric(question)
    return f"Metric:{r['id']}" if r.get("id") else None


def resolve_all(question: str, max_metrics: int = 4) -> list[str]:
    """Question -> the **set** of focal Metric node ids (multi-hop fan-out, whitelist-guarded).

    Comparative / cross-metric questions yield several ids; single-metric ones yield one.
    Falls back to the single-metric resolver if the fan-out finds nothing, so a plain
    question never regresses.
    """
    mids = [f"Metric:{i}" for i in llm.resolve_metrics(question, max_metrics)]
    if not mids:
        one = resolve(question)
        if one:
            mids = [one]
    return mids


def ask(question: str, synthesize: bool = True, g: nx.DiGraph | None = None,
        max_metrics: int = 4) -> str:
    """Resolve focal metric(s) -> gather graph evidence for each -> synthesize a cited answer.

    Multi-hop: a comparison resolves to >1 focal metric, each metric's evidence is gathered
    deterministically (and is itself multi-hop — ``drivers`` walks the DRIVES/ASSUMPTION_OF
    chain), and the LLM writes one joint answer over all the evidence blocks.
    """
    if g is None:
        g = load_graph()
    mids = [m for m in resolve_all(question, max_metrics) if m in g]
    if not mids:
        return "Could not resolve the question to a known metric in the graph."
    blocks = [evidence(g, m) for m in mids]
    ev = "\n\n".join(blocks)
    if not synthesize:
        return ev
    sys = load_prompt("query_synthesis_system")
    plural = "s" if len(mids) > 1 else ""
    user = f"Question: {question}\n\nEvidence ({len(mids)} metric{plural}):\n{ev}\n\nAnswer:"
    return llm.chat([{"role": "system", "content": sys}, {"role": "user", "content": user}],
                    max_tokens=500)


if __name__ == "__main__":
    g = load_graph()
    print(f"loaded graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")
    for q in [
        "What is the equity value and what drives it?",
        "관리수수료 추이가 어떻게 되나요?",          # "how does the management fee trend?"
        "What discount rate (WACC) is used?",
        "관리보수와 성과보수를 비교하면?",            # comparative -> multi-metric fan-out
        "Compare EBITDA and FCFF over the projection.",
    ]:
        print("Q:", q)
        print("  focal metrics:", resolve_all(q))
        print(ask(q, g=g))
        print("-" * 70)
