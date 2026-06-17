"""Shared fixtures + the live-LLM opt-in gate.

The deterministic tests (the bulk) need no network. Tests that hit the shared guest vLLM are
marked ``@pytest.mark.llm`` and skipped unless ``--run-llm`` is passed — they are slow and
non-deterministic and the server is a guest resource that may be down. Fixtures that depend on
build artifacts (``index.json``, the full workbook) skip cleanly when those are absent so a
fresh checkout still runs the pure-logic suite green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# Resolve the canonical wiki + workbook through config, so the suite follows the versioned
# data layout (data/v0.1/...) instead of a hardcoded path.
from src.stella_kb import FULL_WORKBOOK  # noqa: E402
from src.stella_kb.config import agent_wiki_dir  # noqa: E402

INDEX_JSON = agent_wiki_dir() / "index.json"
FULL_WB = Path(FULL_WORKBOOK)


def pytest_addoption(parser):
    parser.addoption("--run-llm", action="store_true", default=False,
                     help="run tests that call the live shared vLLM")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-llm"):
        return
    skip_llm = pytest.mark.skip(reason="needs the live vLLM (pass --run-llm)")
    for item in items:
        if "llm" in item.keywords:
            item.add_marker(skip_llm)


@pytest.fixture(scope="session")
def index() -> dict:
    """The built wiki index (``data/wiki/index.json``); skip if it hasn't been generated."""
    if not INDEX_JSON.exists():
        pytest.skip("data/wiki/index.json not built (run src.stella_kb.wiki.index)")
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def full_workbook() -> str:
    """Path to the full 63-sheet workbook; skip if it isn't present."""
    if not FULL_WB.exists():
        pytest.skip("full workbook not present under data/raw/")
    return str(FULL_WB)
