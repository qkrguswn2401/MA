"""Wiki query agent — the *query* half of Project Stella, kept separate from the
``src/stella_kb`` *build* half.

``src/stella_kb`` compiles the workbook into the vectorless wiki (``data/wiki/``:
``INDEX.md``, ``index.json``, ``pages/*.md``). This package consumes that wiki at question
time: a LangGraph agent navigates the index and pages to answer M&A valuation questions.
It imports the build library only for the shared LLM client; it never rebuilds the wiki.

Layout:
    core.py       public API / facade: run / ask / answer(router) / stream_run
    datasets.py   dataset (wiki version) registry + cached WikiStore
    backends/     the agent backends — supervisor.py · dart.py · wiki/ (LangGraph)
    retrieval/    deterministic wiki access (lookup / open_page / trace_links) — no LLM
    api/          FastAPI HTTP API (/ask, /ask/stream SSE, /datasets, /health)
    prompts/      Korean prompt templates

    from apps.agent import ask
    python -m apps.agent "기업가치는 얼마인가요?"
"""

from .core import ask, run, stream_run

__all__ = ["ask", "run", "stream_run"]
