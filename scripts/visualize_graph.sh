#!/usr/bin/env bash
#
# Visualize the query agent's FULL architecture.
#
# The agent is two backends behind an LLM router (apps/agent/core.py):
#   answer() → route() ─┬─ wiki  → LangGraph (planner → fan-out solve → synthesizer)
#                       └─ dart  → tool-calling agent over the DART MCP (dart.py)
# Only the *wiki* StateGraph is a compiled LangGraph; the route tier and the DART branch
# live in core.py, so get_graph() can't see them. This script therefore emits BOTH:
#   1. Full architecture (route + both backends) — the source of truth in
#      docs/agent_graph.md (mermaid + PNG). This is what you usually want.
#   2. Compiled wiki sub-graph — rendered live from apps.agent.backends.wiki.build, so the wiki
#      half is always verified against the committed code (drift check).
#
# Outputs:
#   - ASCII + Mermaid (both views) → terminal (no network)
#   - PNG of the full architecture → docs/agent_graph.png   (via mermaid.ink; skipped offline)
#   - opens docs/agent_graph.html (interactive Cytoscape, full architecture) if possible
#
# Usage (from anywhere):
#     scripts/visualize_graph.sh             # render both views + PNG, open the HTML
#     NO_OPEN=1 scripts/visualize_graph.sh   # don't try to open a browser
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root (script lives in scripts/)
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

OUT_PNG="docs/agent_graph.png"
HTML="docs/agent_graph.html"
ARCH_MD="docs/agent_graph.md"

echo "==> rendering the FULL agent architecture (route + wiki + dart) ..."
"$PY" - "$OUT_PNG" "$ARCH_MD" <<'PY'
import sys
from pathlib import Path

out_png, arch_md = sys.argv[1], sys.argv[2]

# --- 1. Full architecture: the source-of-truth mermaid lives in docs/agent_graph.md,
#        fenced between <!-- full-arch:begin --> / <!-- full-arch:end --> so the docs and
#        this render never drift. (The route tier + DART branch aren't in the compiled
#        LangGraph, so there's nothing to introspect — the diagram is authored.)
md = Path(arch_md).read_text(encoding="utf-8")
try:
    block = md.split("<!-- full-arch:begin -->", 1)[1].split("<!-- full-arch:end -->", 1)[0]
    full_mermaid = block.split("```mermaid", 1)[1].rsplit("```", 1)[0].strip()
except IndexError:
    sys.exit(f"!! couldn't find the full-arch:begin/end mermaid block in {arch_md}")

print("\n--- Mermaid (FULL architecture: route + wiki + dart) ---")
print(full_mermaid)

# --- 2. Compiled wiki sub-graph: rendered live from code as a drift check on the wiki half.
print("\n--- Compiled wiki sub-graph (live from apps.agent.backends.wiki.build) ---")
try:
    from apps.agent.backends.wiki import build_app
    from apps.agent.retrieval import INDEX_JSON, load_index

    if not INDEX_JSON.exists():
        print("   (skipped — data/wiki/index.json missing; build the wiki first: "
              "scripts/run_pipeline.sh)")
    else:
        g = build_app(load_index()).get_graph()
        try:
            print(g.draw_ascii())
        except Exception as e:                   # grandalf missing, etc.
            print(f"   (ascii unavailable: {e})")
        print(g.draw_mermaid())
except Exception as e:                           # import/build failure shouldn't kill the PNG
    print(f"   (compiled sub-graph unavailable: {type(e).__name__}: {e})")

# --- 3. PNG of the FULL architecture (mermaid.ink — needs internet).
try:
    from langchain_core.runnables.graph_mermaid import draw_mermaid_png
    png = draw_mermaid_png(full_mermaid)
    Path(out_png).write_bytes(png)
    print(f"\n--- wrote {out_png} ({len(png)} bytes, full architecture) ---")
except Exception as e:
    print(f"\n   (PNG skipped — no internet to mermaid.ink? {type(e).__name__}: {e})")
PY

if [ "${NO_OPEN:-0}" = "1" ]; then
  exit 0
fi

opener=""
for c in xdg-open open; do
  command -v "$c" >/dev/null 2>&1 && { opener="$c"; break; }
done

if [ -n "$opener" ]; then
  echo "==> opening $HTML (interactive Cytoscape — full architecture) ..."
  "$opener" "$HTML" >/dev/null 2>&1 || true
  [ -f "$OUT_PNG" ] && "$opener" "$OUT_PNG" >/dev/null 2>&1 || true
else
  echo "==> no browser opener (xdg-open/open). View manually:"
  echo "    interactive : $HTML"
  [ -f "$OUT_PNG" ] && echo "    full PNG    : $OUT_PNG"
  echo "    all views   : $ARCH_MD  (mermaid; renders on GitHub)"
fi
