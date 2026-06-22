"""CLI: ``python -m apps.agent "<question>"`` — demo the agent, printing the routing trace.

Routes each question to the wiki (Centroid) or DART (public company) backend automatically;
force one with ``--source wiki|dart|auto``. With no question, runs a few samples (one DART).
Needs ``data/wiki/`` built and the local vLLM up (see ``src/stella_kb/llm.py``); DART questions
also need the tool LLM (:8001) + the DART MCP server (see ``apps/agent/backends/dart.py``).
"""

from __future__ import annotations

import sys

from .core import answer


def main(argv: list[str]) -> None:
    source = "auto"
    if "--source" in argv:
        i = argv.index("--source")
        source = argv[i + 1] if i + 1 < len(argv) else "auto"
        argv = argv[:i] + argv[i + 2:]

    questions = argv or [
        "기업가치(Enterprise Value)는 얼마이고 어느 셀에서 오나요?",
        "관리수수료(operating revenue)는 어느 장표에 있나요?",
        "삼성전자 공시 알려줘",  # routes to DART
    ]
    for q in questions:
        print("=" * 78)
        print("Q:", q)
        print("-" * 78)
        out = answer(q, source=source, verbose=True)
        print(f"[backend: {out['source']}]")
        print(out["answer"])
        print()


if __name__ == "__main__":
    main(sys.argv[1:])
