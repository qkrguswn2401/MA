"""Wiki I/O — deterministic access to the built wiki (``data/wiki/``). No LLM here."""

from .tools import (
    INDEX_JSON,
    INDEX_MD,
    LEDGERS_DIR,
    PAGES_DIR,
    WIKI_DIR,
    load_index,
    lookup,
    open_page,
    query_ledger,
    trace_links,
)

__all__ = [
    "WIKI_DIR", "INDEX_MD", "INDEX_JSON", "PAGES_DIR", "LEDGERS_DIR",
    "load_index", "lookup", "open_page", "trace_links", "query_ledger",
]
