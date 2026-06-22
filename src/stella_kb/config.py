"""Central configuration: ``config.yaml`` + environment overrides.

Precedence per value: **environment variable > config.yaml > built-in default**. Every legacy
``STELLA_*``/``MNA_*``/``RAGAS_*`` env var still overrides its config key, so scripts and
per-run overrides keep working unchanged. Secrets (``DART_MCP_TOKEN``, ``DART_API_KEY``) are
**not** here — read those from ``os.environ`` / ``.env`` directly.

Imported from anywhere in the repo:
    from src.stella_kb.config import llm_url, llm_model        # apps/agent, eval/
    from ..config import parse_concurrency                     # within src/stella_kb/*
Loads ``config.yaml`` at repo root (override path with ``STELLA_CONFIG``). PyYAML + stdlib
only, so it imports cleanly in the lean ``.venv-ragas`` too.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import yaml

from . import ROOT, WORKBOOK

_CONFIG_PATH = Path(os.environ.get("STELLA_CONFIG", str(ROOT / "config.yaml")))


@lru_cache(maxsize=1)
def _data() -> dict:
    if _CONFIG_PATH.exists():
        return yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) or {}
    return {}


def get(*path: str, env: str | None = None, default: Any = None,
        cast: Callable[[Any], Any] | None = None) -> Any:
    """Resolve one value: env var (if set) > config.yaml at ``path`` > ``default``.

    ``path`` is the nested-key path, e.g. ``get("llm", "url")``. ``cast`` (e.g. ``int``) is
    applied to whatever wins — important since env vars arrive as strings.
    """
    val: Any = None
    if env is not None and os.environ.get(env) is not None:
        val = os.environ[env]
    if val is None:
        node: Any = _data()
        for key in path:
            node = node.get(key) if isinstance(node, dict) else None
            if node is None:
                break
        val = node
    if val is None:
        val = default
    if val is None or cast is None:
        return val
    return cast(val)


# --- typed accessors: one place per setting (env name · yaml path · fallback default) -------

def llm_url() -> str:
    return get("llm", "url", env="STELLA_LLM_URL", default="http://123.37.5.219:8001/v1")


def llm_model() -> str:
    return get("llm", "model", env="STELLA_LLM_MODEL", default="gemma-4-31B-it")


def tool_llm_url() -> str:
    return get("llm", "tool", "url", env="STELLA_TOOL_LLM_URL", default=llm_url())


def tool_llm_model() -> str:
    return get("llm", "tool", "model", env="STELLA_TOOL_LLM_MODEL", default=llm_model())


def parse_concurrency() -> int:
    return get("concurrency", "parse", env="STELLA_CONCURRENCY", default=6, cast=int)


def agent_fanout() -> int:
    return get("concurrency", "fanout", env="STELLA_FANOUT", default=4, cast=int)


def agent_router_top_k() -> int:
    """Max pages the router opens for ONE sub-question in a single round. Opening several related
    pages at once (reads fan out in parallel) is cheaper than picking one and paying for a serial
    ``gap``→retry round, so this is the recall/latency knob. Default 3 keeps prior behavior."""
    return get("agent", "router_top_k", env="MNA_ROUTER_TOPK", default=3, cast=int)


def agent_cross_ref_pairing() -> bool:
    """When true, the router auto-attaches a routed page's PDF↔Excel cross-ref partner(s)
    (``derives_from``/``cited_by``) so a cross-check/reconcile question opens both the FDD report
    page and its Excel source. Capped to avoid over-retrieval. Default **off** — query-affecting,
    so enable for the A/B (env ``MNA_CROSSREF_PAIR``) before turning it on by default."""
    return get("agent", "cross_ref_pairing", env="MNA_CROSSREF_PAIR",
               default=False, cast=lambda v: str(v).lower() in ("1", "true", "yes", "on"))


def agent_deterministic_retrieve() -> bool:
    """When true, the retriever first tries a **deterministic** parse of a page's ``value [cell]``
    table (``retrieval.tools.extract_page_items``) and, on a hit, skips that page's LLM extraction — a
    latency win for already-structured pages. Pages with no parseable table fall back to the LLM.
    Default **off** — it changes evidence selection, so enable only after a clean quality A/B."""
    return get("agent", "deterministic_retrieve", env="MNA_DET_RETRIEVE",
               default=False, cast=lambda v: str(v).lower() in ("1", "true", "yes", "on"))


def eval_fanout(default: int = 8) -> int:
    return get("concurrency", "eval_fanout", env="STELLA_EVAL_FANOUT", default=default, cast=int)


def ragas_concurrency() -> int:
    return get("concurrency", "ragas", env="RAGAS_CONCURRENCY", default=6, cast=int)


def pdf_describe_concurrency() -> int:
    return get("concurrency", "pdf_describe", env="MNA_PDF_DESCRIBE_CONCURRENCY",
               default=4, cast=int)


def max_table_pages() -> int:
    return get("parsing", "max_table_pages", env="MNA_PARSE_MAX_TABLE_PAGES",
               default=80, cast=int)


def pdf_vision_cache() -> str:
    return get("cache", "pdf_vision", env="PDF_VISION_CACHE", default=".cache/pdf_vision")


def pdf_page_png_cache() -> str:
    return get("cache", "pdf_page_png", env="PDF_PAGE_PNG_CACHE", default=".cache/pages")


def pdf_structure_cache() -> str:
    """Disk cache for the PDF *structuring* LLM calls (structure_section / build_document), so a
    wiki rebuild is deterministic and doesn't perturb eval results."""
    return get("cache", "pdf_structure", env="PDF_STRUCTURE_CACHE", default=".cache/pdf_structure")


def wiki_parse_cache() -> str:
    """Disk cache for the wiki *parse* LLM calls (``parse_llm``: a sheet's grid → structure).

    Content-addressed (the key is the model + full prompt, i.e. the grid), so it doubles as the
    **incremental** mechanism: an unchanged sheet hashes to the same key → cache hit → no LLM
    call, while an edited sheet misses and re-parses. Also makes a rebuild deterministic (the
    shared vLLM is non-deterministic even at temp 0). Clear the dir to force a fresh parse."""
    return get("cache", "wiki_parse", env="WIKI_PARSE_CACHE", default=".cache/wiki_parse")


def wiki_prose_cache() -> str:
    """Disk cache for the wiki *compile* prose LLM calls (the page's 'What this is' blurb).

    Same content-addressed/incremental rationale as :func:`wiki_parse_cache`: keyed on the
    facts handed to the model, so a page whose values are unchanged reuses its prose for free."""
    return get("cache", "wiki_prose", env="WIKI_PROSE_CACHE", default=".cache/wiki_prose")


def dart_mcp_url() -> str:
    return get("dart", "mcp_url", env="DART_MCP_URL", default="http://127.0.0.1:8002/sse")


# --- wiki build I/O paths (env-overridable; defaults preserve the canonical data/ tree) -----
# The whole wiki pipeline (dump_md -> parse_llm -> compile -> index -> pdf_pages) reads its
# input workbook/PDFs and writes its md/parsed/wiki artifacts through these accessors, so a
# second corpus can be built into an isolated tree without touching the canonical build:
#     MNA_WIKI_WORKBOOK=<x.xlsx> MNA_WIKI_DATA=data/v0.2 MNA_WIKI_PDF_DIR=test_data/v0.2 \
#         python -m src.stella_kb.wiki.dump_md --all   (and the rest of the stages)
# Defaults reproduce the original hardcoded paths exactly, so existing runs/tests are unchanged.

def wiki_workbook() -> str:
    """Source workbook for the wiki Excel pipeline (dump_md/index)."""
    return get("wiki", "workbook", env="MNA_WIKI_WORKBOOK", default=WORKBOOK)


def alias_stopwords() -> list:
    """Extra structural alias terms to drop in the dedup pass, beyond the curated
    ``wiki.dedup.STRUCTURAL_STOPWORDS``. yaml: ``wiki.alias_stopwords`` (a list of terms)."""
    v = get("wiki", "alias_stopwords", default=[])
    return list(v) if isinstance(v, (list, tuple)) else []


def cross_ref_llm_judge() -> bool:
    """When true, the PDF→Excel cross-ref build (``wiki.cross_refs``) uses the cached, whitelist-
    guarded LLM judge to confirm ambiguous Tier-B candidates. Default **off** — deterministic
    fund-identity + specific-metric links only, pending a quality check on the judged edges."""
    return get("wiki", "cross_ref_llm_judge", env="MNA_CROSSREF_JUDGE",
               default=False, cast=lambda v: str(v).lower() in ("1", "true", "yes", "on"))


def wiki_data_dir() -> Path:
    """Base dir holding the wiki build artifacts (``md/`` ``parsed/`` ``wiki/``). Default is the
    canonical build under ``data/v0.1`` (each corpus version lives in its own ``data/<v>``)."""
    return Path(get("wiki", "data_dir", env="MNA_WIKI_DATA", default="data/v0.1"))


def wiki_pdf_dir() -> Path:
    """Dir scanned for FDD report PDFs to ingest. Default: ``<data_dir>/raw``."""
    return Path(get("wiki", "pdf_dir", env="MNA_WIKI_PDF_DIR",
                    default=str(wiki_data_dir() / "raw")))


def curation_dir() -> Path:
    """Repo-tracked root for hand-authored, **version-controlled** curation, laid out per dataset
    version: ``data/<version>/{decks,routes}.yaml`` — co-located with each version's build. These
    two yamls stay committed via a ``.gitignore`` exception even though the rest of ``data/`` is
    ignored (regenerable), so a fresh checkout still has the curation. Override the root with env
    ``MNA_CURATION_DIR`` / yaml ``curation.dir``."""
    return Path(get("curation", "dir", env="MNA_CURATION_DIR", default=str(ROOT / "data")))


def _version_token(d: Path | str) -> str:
    """Dataset-version token from a data/wiki dir, per the ``data/<version>/wiki`` convention:
    ``data/v0.2`` → ``v0.2`` and ``data/v0.2/wiki`` → ``v0.2``. Names the ``data/<version>/``
    subdir that pairs with the build/dataset."""
    p = Path(d)
    return p.parent.name if p.name == "wiki" else p.name


def wiki_decks_yaml() -> Path:
    """Curated **first-layer** deck index the PDF build reads to override the LLM-synthesized
    document node (per-deck ``title``/``description``). A hand-authored, git-committed input —
    precedence is curated > LLM > default — so the upper layer is deterministic and auditable
    (OpenKB curated-whitelist pattern). Default ``data/<version>/decks.yaml`` (version from
    the build's ``MNA_WIKI_DATA`` dir); explicit file via env ``MNA_WIKI_DECKS``; absent = pure
    LLM."""
    return Path(get("wiki", "decks", env="MNA_WIKI_DECKS",
                    default=str(curation_dir() / _version_token(wiki_data_dir()) / "decks.yaml")))


def wiki_md_dir() -> Path:
    return wiki_data_dir() / "md"


def wiki_parsed_dir() -> Path:
    return wiki_data_dir() / "parsed"


def wiki_pages_dir() -> Path:
    return wiki_data_dir() / "wiki" / "pages"


def wiki_index_json() -> Path:
    return wiki_data_dir() / "wiki" / "index.json"


def wiki_index_md() -> Path:
    return wiki_data_dir() / "wiki" / "INDEX.md"


def agent_wiki_dir() -> Path:
    """Wiki the query agent reads (index.json / pages / ledgers). Default ``data/wiki`` (the
    canonical valuation-model wiki); point it at another build (e.g. ``data/v0.2/wiki``) to
    serve or evaluate against a different corpus without touching the agent code."""
    return Path(get("agent", "wiki_dir", env="MNA_AGENT_WIKI", default="data/v0.1/wiki"))


def agent_routes_yaml(wiki_dir: str | Path | None = None) -> Path:
    """Curated routing table for the query agent — ``term → page(s)`` so a hit skips the router
    LLM. Resolved **per dataset**: ``data/<version>/routes.yaml`` (version from the served
    ``wiki_dir``, default the process wiki). Committed alongside ``decks.yaml``. An explicit env
    ``MNA_AGENT_ROUTES`` file overrides for single-dataset serving (don't set it when serving
    several datasets — it would force one table for all). Absent = pure-LLM routing."""
    base = Path(wiki_dir) if wiki_dir else agent_wiki_dir()
    return Path(get("agent", "routes", env="MNA_AGENT_ROUTES",
                    default=str(curation_dir() / _version_token(base) / "routes.yaml")))


if __name__ == "__main__":  # smoke: print the resolved config
    print(f"config file: {_CONFIG_PATH}  (exists={_CONFIG_PATH.exists()})")
    for name in ("llm_url", "llm_model", "tool_llm_url", "tool_llm_model", "parse_concurrency",
                 "agent_fanout", "eval_fanout", "ragas_concurrency", "pdf_describe_concurrency",
                 "max_table_pages", "pdf_vision_cache", "pdf_page_png_cache", "dart_mcp_url"):
        print(f"  {name:24s} = {globals()[name]()!r}")
