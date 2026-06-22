"""Wiki retrieval — deterministic access to the built wiki (``data/wiki/``). No LLM here."""

from .tools import (
    INDEX_JSON,
    INDEX_MD,
    LEDGERS_DIR,
    PAGES_DIR,
    WIKI_DIR,
    cross_ref_partners,
    extract_page_items,
    load_index,
    load_routes,
    lookup,
    open_page,
    persist_answer,
    query_ledger,
    route_lookup,
    trace_links,
)

__all__ = [
    "WIKI_DIR", "INDEX_MD", "INDEX_JSON", "PAGES_DIR", "LEDGERS_DIR",
    "load_index", "lookup", "open_page", "trace_links", "query_ledger",
    "load_routes", "route_lookup", "extract_page_items", "persist_answer",
    "cross_ref_partners",
]
