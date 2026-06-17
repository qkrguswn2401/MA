"""Render the Tier-1 semantic graph (:mod:`lift`) as a self-contained interactive HTML.

No Python plotting deps — reads ``data/stella_semantic.json`` and emits one HTML file with
the graph data inlined and `vis-network` loaded from a CDN. Nodes are colored by type and
each node type / edge type can be toggled, so the 694-node graph stays explorable.

Output goes to ``frontend/web/graph.html`` so the running FastAPI server serves it at
``/ui/graph.html`` (it also works opened directly via file://, since the data is inlined).

Two graphs can be rendered (same node-link template, different sources):
  - ``wiki``    — :mod:`lift`'s wiki-grounded graph (``stella_semantic.json``) -> graph.html
  - ``curated`` — :mod:`semantic`'s DCF-anchor graph (``stella_graph.json``)   -> graph_curated.html

Usage (repo root, venv active; build the source JSON first):
    python -m src.stella_kb.graph.lift && python -m src.stella_kb.graph.viz          # wiki (default)
    python -m src.stella_kb.graph.semantic && python -m src.stella_kb.graph.viz curated
    python -m src.stella_kb.graph.viz both
"""

from __future__ import annotations

import json
from pathlib import Path

from .. import DATA_DIR, ROOT

# named views: key -> (source json, output html, page title)
VIEWS = {
    "wiki": (DATA_DIR / "graph" / "stella_semantic.json", ROOT / "frontend" / "web" / "graph.html",
             "Semantic Graph (wiki-grounded · lift)"),
    "curated": (DATA_DIR / "graph" / "stella_graph.json", ROOT / "frontend" / "web" / "graph_curated.html",
                "DCF Anchor Graph (curated · semantic)"),
}

# node type -> color; the structural types are muted, the metric grains pop
COLORS = {
    "Metric": "#5b8cff", "Concept": "#9b5bff", "Sheet": "#3ecf8e",
    "Section": "#2bb39a", "Fund": "#e0a341", "Entity": "#ff6b6b", "Period": "#7a8290",
}
# Concept + Period make the graph hairy (star INSTANCE_OF / COVERS edges) — off by default
DEFAULT_OFF = {"Concept", "Period"}
EDGE_OFF = {"INSTANCE_OF", "COVERS"}


def to_vis(graph: dict) -> tuple[list, list]:
    """node-link JSON -> (vis nodes, vis edges)."""
    # semantic.py exports node-link JSON under "edges"; lift.py under "links" — accept both
    edge_list = graph.get("links") or graph.get("edges") or []
    deg: dict[str, int] = {}
    for e in edge_list:
        deg[e["source"]] = deg.get(e["source"], 0) + 1
        deg[e["target"]] = deg.get(e["target"], 0) + 1

    nodes = []
    for n in graph["nodes"]:
        t = n.get("type", "?")
        nid = n["id"]
        tip = f"{t}: {n.get('label', nid)}"
        if n.get("sheet"):
            tip += f"<br>sheet: {n['sheet']}"
        if n.get("cell"):
            tip += f"<br>cell: {n['sheet']}!{n['cell']}"
        if n.get("role"):
            tip += f"<br>role: {n['role']}"
        nodes.append({
            "id": nid, "label": n.get("label", nid), "group": t, "title": tip,
            "value": 1 + deg.get(nid, 0), "color": COLORS.get(t, "#888"),
            "hidden": t in DEFAULT_OFF,
        })

    edges = []
    for e in edge_list:
        et = e.get("type", "?")
        edges.append({
            "from": e["source"], "to": e["target"], "etype": et,
            "title": et, "arrows": "to",
            "hidden": et in EDGE_OFF,
            "color": {"color": "#ff8c42", "opacity": 0.9} if et == "DEPENDS_ON"
                     else {"color": "#3a3f4a", "opacity": 0.5},
        })
    return nodes, edges


def render(graph: dict, title: str = "Semantic Graph") -> str:
    nodes, edges = to_vis(graph)
    ntypes = sorted({n["group"] for n in nodes})
    etypes = sorted({e["etype"] for e in edges})
    legend = " ".join(
        f'<span class="lg"><i style="background:{COLORS.get(t, "#888")}"></i>{t}</span>'
        for t in ntypes)
    ncb = "".join(
        f'<label><input type="checkbox" data-ntype="{t}" '
        f'{"" if t in DEFAULT_OFF else "checked"}> {t}</label>' for t in ntypes)
    ecb = "".join(
        f'<label><input type="checkbox" data-etype="{t}" '
        f'{"" if t in EDGE_OFF else "checked"}> {t}</label>' for t in etypes)

    return _TEMPLATE.format(
        title=title,
        nodes=json.dumps(nodes, ensure_ascii=False),
        edges=json.dumps(edges, ensure_ascii=False),
        legend=legend, ncb=ncb, ecb=ecb,
        n=len(nodes), e=len(edges))


_TEMPLATE = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"/>
<title>Project Stella — {title}</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html,body{{margin:0;height:100%;background:#0f1115;color:#e6e9ef;
    font-family:-apple-system,"Noto Sans KR",sans-serif;}}
  #net{{position:absolute;top:0;left:0;right:0;bottom:0;}}
  #panel{{position:absolute;top:12px;left:12px;z-index:10;background:#171a21ee;
    border:1px solid #2a2f3a;border-radius:10px;padding:12px 14px;max-width:300px;font-size:12.5px;}}
  #panel h1{{font-size:14px;margin:0 0 6px;}}
  #panel .sub{{color:#9aa3b2;margin-bottom:8px;}}
  .grp{{margin:8px 0 2px;color:#9aa3b2;text-transform:uppercase;font-size:10.5px;letter-spacing:.05em;}}
  label{{display:inline-block;margin:2px 8px 2px 0;cursor:pointer;}}
  .lg{{display:inline-flex;align-items:center;gap:4px;margin-right:8px;}}
  .lg i{{width:10px;height:10px;border-radius:50%;display:inline-block;}}
  input[type=text]{{width:100%;box-sizing:border-box;background:#1e222b;border:1px solid #2a2f3a;
    color:#e6e9ef;border-radius:7px;padding:6px 8px;margin-top:6px;}}
  .hint{{color:#9aa3b2;margin-top:8px;font-size:11px;}}
</style></head>
<body>
<div id="panel">
  <h1>{title}</h1>
  <div class="sub">{n} nodes · {e} edges · 색상 = 노드 유형</div>
  <div>{legend}</div>
  <div class="grp">Node types</div><div>{ncb}</div>
  <div class="grp">Edge types</div><div>{ecb}</div>
  <input type="text" id="search" placeholder="노드 검색 (label)…"/>
  <div class="hint">DEPENDS_ON = 주황색 · 드래그/스크롤로 탐색 · Concept·Period 기본 숨김</div>
</div>
<div id="net"></div>
<script>
const rawNodes = {nodes};
const rawEdges = {edges};
const nodes = new vis.DataSet(rawNodes);
const edges = new vis.DataSet(rawEdges.map((e,i)=>({{id:i, ...e}})));
const net = new vis.Network(document.getElementById('net'), {{nodes, edges}}, {{
  nodes:{{shape:'dot', scaling:{{min:6,max:34}}, font:{{color:'#cdd6df', size:11}}}},
  edges:{{smooth:{{type:'continuous'}}, width:0.6}},
  physics:{{solver:'forceAtlas2Based',
    forceAtlas2Based:{{gravitationalConstant:-45, springLength:90, springConstant:0.05}},
    stabilization:{{iterations:180}}}},
  interaction:{{hover:true, tooltipDelay:120}},
}});
// type toggles
document.querySelectorAll('input[data-ntype]').forEach(cb=>cb.onchange=()=>{{
  const t=cb.dataset.ntype, on=cb.checked;
  nodes.update(rawNodes.filter(n=>n.group===t).map(n=>({{id:n.id, hidden:!on}})));
}});
document.querySelectorAll('input[data-etype]').forEach(cb=>cb.onchange=()=>{{
  const t=cb.dataset.etype, on=cb.checked;
  const upd=[]; edges.forEach(e=>{{ if(e.etype===t) upd.push({{id:e.id, hidden:!on}}); }});
  edges.update(upd);
}});
// search -> select+focus
document.getElementById('search').onkeydown=ev=>{{
  if(ev.key!=='Enter') return;
  const q=ev.target.value.trim().toLowerCase(); if(!q) return;
  const hit=rawNodes.find(n=>(n.label||'').toLowerCase().includes(q));
  if(hit){{ net.selectNodes([hit.id]); net.focus(hit.id,{{scale:1.1, animation:true}}); }}
}};
</script>
</body></html>
"""


def render_view(key: str) -> None:
    in_json, out_html, title = VIEWS[key]
    if not in_json.exists():
        builder = "lift" if key == "wiki" else "semantic"
        raise SystemExit(f"!! {in_json} missing — run `python -m src.stella_kb.graph.{builder}` first")
    graph = json.loads(in_json.read_text(encoding="utf-8"))
    n_edges = len(graph.get("links") or graph.get("edges") or [])
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(render(graph, title), encoding="utf-8")
    print(f"wrote {out_html}  ({len(graph['nodes'])} nodes, {n_edges} edges)  ->  /ui/{out_html.name}")


if __name__ == "__main__":
    import sys

    arg = sys.argv[1] if len(sys.argv) > 1 else "wiki"
    keys = list(VIEWS) if arg == "both" else [arg]
    if any(k not in VIEWS for k in keys):
        raise SystemExit(f"usage: viz [wiki|curated|both]  (got {arg!r})")
    for k in keys:
        render_view(k)
    print("serve: files are under frontend/web/ -> open http://localhost:8000/ui/<name>")
