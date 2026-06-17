"""Dataset (wiki version) registry + per-request store.

A *dataset* is one built wiki under a directory (``index.json`` + ``pages/`` + ``ledgers/``).
The HTTP API selects one per request by a short **id** (``"default"``, ``"v0.2"``, …) rather
than a filesystem path — the id is resolved here against a config-driven registry
(``config.yaml`` ``agent.datasets``), so a client can never point the agent at an arbitrary
directory. Each dataset's index + INDEX.md are cached so concurrent requests reuse them.

Per-request retrieval stays concurrency-safe because the chosen wiki dir is threaded through
the agent state (``AgentState.wiki_dir`` → ``open_page``/``query_ledger``), never set as a
process-global — so two in-flight requests can target different datasets at once.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from src.stella_kb.config import agent_wiki_dir, get

DEFAULT = "default"


def registry() -> dict[str, str]:
    """``{id: wiki_dir}`` from ``config.yaml`` ``agent.datasets``, always incl. ``default``
    (which falls back to ``agent_wiki_dir()`` / ``MNA_AGENT_WIKI`` when not listed)."""
    reg = {str(k): str(v) for k, v in (get("agent", "datasets", default={}) or {}).items()}
    reg.setdefault(DEFAULT, str(agent_wiki_dir()))
    return reg


def available() -> list[str]:
    """Sorted dataset ids a client may pass as ``dataset``."""
    return sorted(registry())


def resolve_dir(dataset: str | None) -> Path:
    """Map a dataset id to its wiki dir. ``None``/empty → the default. Raises ``KeyError``
    (with the unknown id) if it isn't registered — the API turns that into a 422."""
    reg = registry()
    key = dataset or DEFAULT
    if key not in reg:
        raise KeyError(key)
    return Path(reg[key])


@lru_cache(maxsize=8)
def _load_index(path: str, _mtime: float) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _load_md(path: str, _mtime: float) -> str:
    return Path(path).read_text(encoding="utf-8")


class WikiStore:
    """A resolved dataset: its wiki dir plus lazily-loaded, cached ``index`` and ``index_md``.

    Caching is keyed by (path, mtime) so a rebuilt wiki is picked up without a restart."""

    def __init__(self, dataset: str | None = None):
        self.dataset = dataset or DEFAULT
        self.wiki_dir = resolve_dir(dataset)
        self.index_json = self.wiki_dir / "index.json"
        self.index_md_path = self.wiki_dir / "INDEX.md"

    def exists(self) -> bool:
        return self.index_json.exists()

    @property
    def index(self) -> dict:
        return _load_index(str(self.index_json), self.index_json.stat().st_mtime)

    @property
    def index_md(self) -> str:
        return _load_md(str(self.index_md_path), self.index_md_path.stat().st_mtime)


@lru_cache(maxsize=8)
def get_store(dataset: str | None = None) -> WikiStore:
    """Cached :class:`WikiStore` for a dataset id (raises ``KeyError`` for unknown ids)."""
    return WikiStore(dataset)
