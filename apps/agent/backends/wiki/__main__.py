"""Visualize the agent's LangGraph structure.

    python -m apps.agent.backends.wiki                 # print Mermaid to stdout
    python -m apps.agent.backends.wiki --out FILE.mmd  # also write the Mermaid source
    python -m apps.agent.backends.wiki --png FILE.png  # render a PNG (needs network: mermaid.ink)

Paste the Mermaid into https://mermaid.live or a Markdown ```mermaid block to view it.
ASCII (``get_graph().draw_ascii()``) is intentionally not offered: grandalf mis-renders
the ``agent → agent`` self-loop. The graph structure is index-independent, so we build it
with an empty index.
"""

from __future__ import annotations

import sys

from .build import build_app


def main(argv: list[str]) -> None:
    graph = build_app({}).get_graph()
    mermaid = graph.draw_mermaid()
    print(mermaid)

    if "--out" in argv:
        path = argv[argv.index("--out") + 1]
        with open(path, "w", encoding="utf-8") as f:
            f.write(mermaid)
        print(f"# wrote Mermaid source -> {path}", file=sys.stderr)

    if "--png" in argv:
        path = argv[argv.index("--png") + 1]
        try:
            graph.draw_mermaid_png(output_file_path=path)
            print(f"# wrote PNG -> {path}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 — PNG needs network / extra deps
            print(f"# PNG render failed ({type(e).__name__}: {e}); "
                  "use the Mermaid text instead.", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1:])
