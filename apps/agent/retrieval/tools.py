"""Deterministic wiki access — the "code does all retrieval" half of the agent.

No LLM here. These are the read tools the agent drives over ``data/wiki/`` (built by
``src/stella_kb``): ``lookup`` resolves a KO/EN term to candidate pages via the
``alias_index`` (words→node), and ``open_page`` reads a page's grounded facts table off
disk. Every number a page surfaces already carries its ``Sheet!Cell``.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from src.stella_kb.config import agent_wiki_dir

WIKI_DIR = agent_wiki_dir()          # env MNA_AGENT_WIKI overrides (default data/wiki)
INDEX_MD = WIKI_DIR / "INDEX.md"
INDEX_JSON = WIKI_DIR / "index.json"
PAGES_DIR = WIKI_DIR / "pages"
LEDGERS_DIR = WIKI_DIR / "ledgers"   # per-fund 거래내역 row sidecars (src/stella_kb/wiki/ledger.py)


def load_index() -> dict:
    """The machine-readable index: ``{tree, pages, alias_index}``."""
    return json.loads(INDEX_JSON.read_text(encoding="utf-8"))


# --- curated routing table (routes.yaml) — deterministic term→page, skips the router LLM ----

@lru_cache(maxsize=8)
def _load_routes_cached(path: str, _mtime: float) -> dict:
    """Parse ``routes.yaml`` → ``{normalized_term: [page, ...]}``. Cached by (path, mtime) so an
    edit is picked up live but the file isn't re-read on every sub-question. Tolerant of an
    empty/malformed file (→ ``{}``) so a bad table never breaks routing — it just falls back to
    the LLM router."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a broken table must degrade to the LLM router, not crash
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, list[str]] = {}
    for k, v in data.items():
        pages = v if isinstance(v, list) else [v]
        out[_norm(str(k))] = [str(p) for p in pages if p]
    return out


def load_routes(wiki_dir: str | Path | None = None) -> dict:
    """Load the per-dataset curated routing table (``curation/<version>/routes.yaml``, resolved
    from ``wiki_dir`` by ``config.agent_routes_yaml``); ``{}`` if absent.

    Per-request ``wiki_dir`` (concurrency-safe — never a global), default the process wiki."""
    from src.stella_kb.config import agent_routes_yaml

    path = agent_routes_yaml(wiki_dir)
    if not path.exists():
        return {}
    return _load_routes_cached(str(path), path.stat().st_mtime)


def cross_ref_partners(index: dict, page: str, cap: int = 2) -> list[str]:
    """Directed PDF↔Excel cross-ref partners of ``page`` — an FDD page's Excel sources
    (``derives_from``), or an Excel page's citing FDD pages (``cited_by``). Deterministic; empty
    if none. Lets the router auto-pair both sides of a PDF×Excel cross-check question."""
    e = index.get("pages", {}).get(page, {})
    if e.get("source") == "PDF":
        return [d["page"] if isinstance(d, dict) else d for d in (e.get("derives_from") or [])][:cap]
    return list(e.get("cited_by") or [])[:cap]


def route_lookup(hint_terms: list[str], index: dict,
                 wiki_dir: str | Path | None = None) -> list[str]:
    """Deterministic curated routing: a sub-question's hint terms → pages, via ``routes.yaml``.

    A hit lets the agent skip the router LLM entirely (the latency win). Returns validated,
    order-preserving, deduped page names — only those that actually exist in ``index['pages']``
    (a stale curated target is silently dropped). Empty when there's no table or no term hits,
    in which case the caller falls back to the LLM router."""
    routes = load_routes(wiki_dir)
    if not routes:
        return []
    valid = index.get("pages", {})
    picks, seen = [], set()
    for t in (hint_terms or []):
        for p in routes.get(_norm(str(t)), []):
            if p in valid and p not in seen:
                seen.add(p)
                picks.append(p)
    return picks


def _norm(term: str) -> str:
    return re.sub(r"\s+", "", term).casefold()


def lookup(index: dict, term: str, limit: int = 12) -> str:
    """Resolve a term to candidate pages via the alias index (the words→node resolver).

    Exact normalized match first; falls back to substring containment either way so a
    query term (``관리수수료``) still finds a near-label (``관리보수``). Each hit is
    enriched with the page's metadata so the agent can disambiguate generic terms
    (``합계``, ``관리보수``) by fund/section group + period without opening every page.
    """
    ai = index["alias_index"]
    pages = index["pages"]
    key = _norm(term)

    seen, hits = set(), []
    for ak, bucket in ai.items():
        if key == ak or key in ak or ak in key:
            for h in bucket:
                sig = (h["page"], h["cell"])
                if sig in seen:
                    continue
                seen.add(sig)
                hits.append(h)

    if not hits:
        return f"LOOKUP {term!r} → no matching pages in the alias index."

    # rank exact-key hits first, then by page so collisions group together
    hits.sort(key=lambda h: (_norm(h["term"]) != key, h["page"]))
    lines = [f"LOOKUP {term!r} → {len(hits)} hit(s)"
             + (f" (showing {limit})" if len(hits) > limit else "") + ":"]
    for h in hits[:limit]:
        p = pages.get(h["page"], {})
        meta = " · ".join(m for m in (
            p.get("kind"), p.get("group"),
            (f"case {p['case']}" if p.get("case") else None),
            p.get("period"), p.get("unit"),
            (f"data: {p['data_status']}" if p.get("data_status") else None),
        ) if m)
        lines.append(f"- page {h['page']!r}  cell {h['cell']}  term {h['term']!r}"
                     + (f"  [{meta}]" if meta else ""))
    return "\n".join(lines)


def trace_links(index: dict, start: str, direction: str = "down",
                max_depth: int = 4, cap: int = 14) -> list[dict]:
    """Walk the sheet-level formula DAG from ``start`` — the deterministic provenance hop.

    BFS over ``index['sheet_dag']`` following ``feeds_into`` (``direction='down'`` — where a
    value *flows to*) or ``depends_on`` (``direction='up'`` — what a value *comes from*).
    Cycle-safe (Excel has bidirectional engine refs) and depth/size capped. Returns ordered
    ``[{sheet, depth, has_page}]`` — the auditable chain; ``has_page`` flags which hops the
    agent can actually open (engine sheets like ``DCF`` have no wiki page but still belong on
    the path).
    """
    from collections import deque

    dag = index.get("sheet_dag", {})
    pages = index.get("pages", {})
    key = "feeds_into" if direction == "down" else "depends_on"

    seen = {start}
    chain: list[dict] = []
    dq = deque([(start, 0)])
    while dq and len(chain) < cap:
        node, d = dq.popleft()
        if d >= max_depth:
            continue
        for nb in dag.get(node, {}).get(key, []):
            if nb in seen:
                continue
            seen.add(nb)
            chain.append({"sheet": nb, "depth": d + 1, "has_page": nb in pages})
            dq.append((nb, d + 1))
            if len(chain) >= cap:
                break
    return chain


def open_page(name: str, wiki_dir: str | Path | None = None) -> str:
    """Return a page's markdown (frontmatter trimmed to the essentials to save context).

    ``wiki_dir`` overrides the default wiki per call (the API threads the per-request dataset's
    dir here); ``None`` uses the process default ``PAGES_DIR``."""
    pages = (Path(wiki_dir) / "pages") if wiki_dir else PAGES_DIR
    path = pages / f"{name}.md"
    if not path.exists():
        return (f"OPEN {name!r} → no such page. Use the EXACT page name from the INDEX "
                "(it is the wikilink text).")
    text = path.read_text(encoding="utf-8")
    # drop the long `aliases:` frontmatter line; keep sheet/section/case/unit + body
    if text.startswith("---"):
        fm, _, body = text.partition("\n---\n")
        kept = [ln for ln in fm.splitlines() if not ln.startswith("aliases:")]
        text = "\n".join(kept) + "\n---\n" + body
    return f"OPEN {name!r}:\n{text}"


_CELL_VAL = re.compile(r"^\s*(.+?)\s*\[([^\]\s]+)\]\s*$")  # "46328767 [D6]" / "3.70% [FDD8]"


def extract_page_items(page_md: str, hint_terms: list[str] | None = None,
                       cap: int = 60) -> list[dict]:
    """Deterministically pull value-bearing rows from a wiki page's markdown table(s).

    The compile step renders every page fact as ``value [cell]`` in a pipe table with a header
    row (period columns for Excel time-series, a single ``value`` column for PDF figures). This
    parses those cells into ``[{term, period, value, cell}]`` — no LLM, exact cell provenance —
    so the retriever can skip its per-page LLM call for pages that are already structured (the
    same "code does retrieval" win as the ledger sidecar, but for ordinary tables).

    ``hint_terms`` (optional) keeps only rows whose label matches a hint (normalized substring);
    omitted → every row. Empty list when the page has no parseable ``value [cell]`` table (e.g.
    a prose-only page), so the caller falls back to the LLM extractor. ``cap`` bounds the rows.
    """
    hints = [_norm(h) for h in (hint_terms or []) if h]
    out: list[dict] = []
    header: list[str] = []
    for line in page_md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            header = []  # table ended; reset so a later table re-detects its own header
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and all(c and set(c) <= {"-", ":"} for c in cells):  # the |---|---| separator
            continue
        if not header:  # first pipe row of a block is its header
            header = cells
            continue
        label = cells[0].strip(" *`")
        if not label or (hints and not any(h in _norm(label) for h in hints)):
            continue
        for ci, cell in enumerate(cells[1:], start=1):
            m = _CELL_VAL.match(cell)
            if not m:  # not a "value [ref]" cell (label/role/blank columns)
                continue
            value, ref = m.group(1).strip(), m.group(2)
            if not value:
                continue
            hdr = header[ci] if ci < len(header) else ""
            period = hdr if re.match(r"^(FY)?\s*\d{4}", hdr) else ""
            out.append({"term": label, "period": period, "value": value, "cell": ref})
            if len(out) >= cap:
                return out
    return out


def persist_answer(question: str, answer: str, evidence: list[dict],
                   wiki_dir: str | Path | None = None, page: str | None = None) -> dict:
    """Compound a valuable answer onto its wiki page (the query→permanent-page step).

    Writes the answer to the page's append-only sidecar (``<wiki>/qa/<page>.jsonl``, the source
    of truth) and re-renders the page's ``## Q&A (compounded)`` section from it so it shows
    immediately *and* survives the next ``compile`` rebuild. The target page defaults to the one
    the evidence most cites. Persisted **only if grounded** (real answer + ≥1 cell of evidence)
    and the target page exists — so an ungrounded/uncited answer is refused, never laundered
    onto a page. ``wiki_dir`` is per-request (concurrency-safe), default the process wiki.

    Returns ``{ok, page, reason?, n_qa?}``."""
    from src.stella_kb.wiki import qa

    base = Path(wiki_dir) if wiki_dir else WIKI_DIR
    entry = qa.new_entry(question, answer, evidence)
    if not qa.is_grounded(entry):
        return {"ok": False, "page": page, "reason": "ungrounded (no cell evidence)"}
    page = page or qa.target_page(evidence)
    if not page:
        return {"ok": False, "page": None, "reason": "no target page in evidence"}
    page_path = base / "pages" / f"{page}.md"
    if not page_path.exists():
        return {"ok": False, "page": page, "reason": f"target page {page!r} not found"}

    qa.append_qa(base, page, entry)
    entries = qa.load_qa(base, page)
    page_path.write_text(
        qa.upsert_qa_section(page_path.read_text(encoding="utf-8"), entries, target_page=page),
        encoding="utf-8")
    return {"ok": True, "page": page, "n_qa": len(entries)}


def query_ledger(page: str, keywords: list[str], ask: str = "", cap: int = 10,
                 wiki_dir: str | Path | None = None) -> list[dict]:
    """Deterministic filter+sum over a ``*_거래내역`` ledger sidecar → evidence items.

    Transaction ledgers are dropped by the time-series parse (rows aren't on the wiki page), so
    this reads the row sidecar (``LEDGERS_DIR/<page>.json``), filters rows whose 적요 contains any
    keyword, and **sums 출금 per currency** (USD→원화 via each row's rate) — no LLM arithmetic.
    Returns evidence ``{page, cell, term, value, ask}``: the matched rows (capped), per-currency
    and grand totals (with the contributing cells as provenance), and a ``0건 — 해당 적요 없음``
    marker per keyword with no matches (the "그런 적요 자체가 없다" signal). Empty if no sidecar
    or no usable keywords."""
    from src.stella_kb.wiki.ledger import query_ledger_rows

    ledgers = (Path(wiki_dir) / "ledgers") if wiki_dir else LEDGERS_DIR
    path = ledgers / f"{page}.json"
    kws = [str(k) for k in (keywords or []) if k and len(str(k)) >= 2]
    if not path.exists() or not kws:
        return []
    q = query_ledger_rows(json.loads(path.read_text(encoding="utf-8")), kws)
    out: list[dict] = []
    cells_by_cur: dict[str, list[str]] = {}
    for m in q["matched"]:
        cur, ref = m.get("currency", "KRW"), str(m.get("ref", "")).split("!")[-1]
        cells_by_cur.setdefault(cur, []).append(ref)
    for m in q["matched"][:cap]:
        out.append({"page": page, "cell": str(m.get("ref", "")).split("!")[-1],
                    "term": f"{m.get('desc', '')} ({m.get('currency')} 출금)", "period": "",
                    "value": f"{m.get('outflow')}", "ask": ask})
    kw_label = "·".join(kws)
    for cur, total in q["totals_krw"].items():
        out.append({"page": page, "cell": "+".join(cells_by_cur.get(cur, [])) or "—",
                    "term": f"{kw_label} 출금 합계 ({cur}{'→원화' if cur == 'USD' else ''})",
                    "period": "", "value": f"{total:,}", "ask": ask})
    if q["totals_krw"]:
        out.append({"page": page, "cell": "ledger-sum",
                    "term": f"{kw_label} 출금 총합(원화 환산)", "period": "",
                    "value": f"{q['krw_total']:,}", "ask": ask})
    for kw in q["absent"]:
        out.append({"page": page, "cell": "—", "term": f"적요 '{kw}' 검색", "period": "",
                    "value": "0건 — 해당 적요 없음", "ask": ask})
    return out
