#!/usr/bin/env bash
#
# Visualize the CURRENT query-agent graph, straight from the live code
# (apps/agent/graph/build.py — so it always reflects whatever is committed).
#
# Emits three views:
#   1. ASCII        → the terminal (always; no network)
#   2. Mermaid src  → the terminal (paste into any mermaid renderer / GitHub)
#   3. PNG          → docs/agent_graph.png   (via mermaid.ink; skipped if offline)
# then opens the interactive Cytoscape view (docs/agent_graph.html) if a browser opener
# exists. The compiled graph is the 3-node fan-out skeleton (planner → solve → synthesizer);
# the expanded pipeline (sub-agents inside solve) lives in docs/agent_graph.md / .html.
#
# Usage (from anywhere):
#     scripts/visualize_graph.sh          # ASCII + mermaid + PNG, open the HTML
#     NO_OPEN=1 scripts/visualize_graph.sh   # don't try to open a browser
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root (script lives in scripts/)
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

OUT_PNG="docs/agent_graph.png"
HTML="docs/agent_graph.html"

echo "==> rendering the compiled graph from apps.agent.graph.build ..."
"$PY" - "$OUT_PNG" <<'PY'
import sys

from apps.agent.graph import build_app
from apps.agent.io import INDEX_JSON, load_index

out_png = sys.argv[1]
if not INDEX_JSON.exists():
    sys.exit("!! data/wiki/index.json missing — build the wiki first (scripts/run_pipeline.sh)")

g = build_app(load_index()).get_graph()

print("\n--- ASCII (compiled topology) ---")
try:
    print(g.draw_ascii())
except Exception as e:                       # grandalf missing, etc.
    print(f"   (ascii unavailable: {e})")

print("\n--- Mermaid (compiled topology) ---")
print(g.draw_mermaid())

try:
    png = g.draw_mermaid_png()               # mermaid.ink — needs internet
    with open(out_png, "wb") as f:
        f.write(png)
    print(f"--- wrote {out_png} ({len(png)} bytes) ---")
except Exception as e:
    print(f"   (PNG skipped — no internet to mermaid.ink? {type(e).__name__})")
PY

if [ "${NO_OPEN:-0}" = "1" ]; then
  exit 0
fi

opener=""
for c in xdg-open open; do
  command -v "$c" >/dev/null 2>&1 && { opener="$c"; break; }
done

if [ -n "$opener" ]; then
  echo "==> opening $HTML (interactive Cytoscape view) ..."
  "$opener" "$HTML" >/dev/null 2>&1 || true
  [ -f "$OUT_PNG" ] && "$opener" "$OUT_PNG" >/dev/null 2>&1 || true
else
  echo "==> no browser opener (xdg-open/open). View manually:"
  echo "    interactive : $HTML"
  [ -f "$OUT_PNG" ] && echo "    compiled PNG: $OUT_PNG"
  echo "    expanded    : docs/agent_graph.md  (mermaid; renders on GitHub)"
fi
