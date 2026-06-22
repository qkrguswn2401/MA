"""Agent prompt templates, kept out of code (one ``<name>.txt`` per prompt).

Mirrors ``src/stella_kb/prompts`` but scoped to the query agent, so the agent package is
self-contained. Load with ``load("planner")``; the path resolves relative to
this folder, so it works regardless of cwd.
"""

from pathlib import Path

_DIR = Path(__file__).resolve().parent


def load(name: str) -> str:
    """Return the text of prompt ``<name>.txt`` from this folder."""
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()
