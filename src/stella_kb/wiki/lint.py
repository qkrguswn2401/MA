"""Stage 6 (maintenance): lint a built wiki for internal consistency.

The build pipeline (dump → parse → compile → index) *constructs* the wiki; this pass
*audits* it — the third operation of the LLM-wiki pattern (ingest / query / **lint**),
the health check that keeps the KB trustworthy as it grows and is edited incrementally.

Fully deterministic and offline (no LLM, no workbook): it reads only the built artifacts
under a wiki dir — ``index.json``, ``pages/*.md``, ``INDEX.md`` — plus the dataset's
curated ``routes.yaml`` if present. So it runs in the test suite and as a CI gate, and is
cheap enough to run after every incremental rebuild.

Checks (severity):
  - ``broken_link``      (error) — a ``[[target]]`` that is neither a page nor a real
    sheet name. A link to a real-but-out-of-scope engine sheet (no wiki page, e.g.
    ``[[AUM Projection]]``) is reported separately as ``dangling_sheet_link`` (info),
    since the wiki is ``_raw``-only by design and those are expected.
  - ``missing_page``     (error) — an ``index['pages']`` entry whose ``.md`` file is gone
    (index drifted from disk — the agent's lookup/ToC would point at an unopenable page).
  - ``orphan_alias``     (error) — an ``alias_index`` hit whose page no longer exists on
    disk: ``lookup()`` would resolve a term to a page the agent then fails to open.
  - ``orphan_page``      (warn)  — a page file with no ``index['pages']`` entry: openable
    by exact name but invisible to lookup and the ToC.
  - ``stale_route``      (warn)  — a curated ``routes.yaml`` target that isn't a real page
    (``route_lookup`` drops it silently; surface it so curation can be fixed).
  - ``contradiction``    (warn, **opt-in** ``--contradictions``) — two *different* pages
    that claim different values for the **same term + period**, within the **same
    section/group/kind/case**. The case key makes this **dual-case-aware**: MGT vs DTT (or
    FDD-deck vs Excel) values for the same metric are *expected* to differ and are never
    compared. It is off by default and heuristic: this corpus holds many *legitimate*
    parallel series that share a slot but aren't case-tagged (EIU **KR vs US** both classify
    to ``group: EIU``; exhibit scenarios ``#1/#2/#3``), so a term match across them is not a
    real clash. Treat its output as leads to inspect, not failures.

``--fix`` prunes the two kinds of drift it can mend safely without touching generated
prose: ``orphan_alias`` hits and ``missing_page`` entries are removed from ``index.json``
in place (useful between full rebuilds, when an incrementally-served index has drifted).
Broken links and contradictions are reported only — they live in generated/curated text a
rebuild owns.

Exit code is non-zero iff any *error*-severity finding exists, so CI can gate on it.

Usage (from repo root, venv active):
    python -m src.stella_kb.wiki.lint                  # lint the default agent wiki
    python -m src.stella_kb.wiki.lint data/v0.2/wiki   # lint a specific build
    python -m src.stella_kb.wiki.lint --fix            # prune the prunable drift
    python -m src.stella_kb.wiki.lint --contradictions # also run the opt-in value-clash check
    python -m src.stella_kb.wiki.lint --json           # machine-readable report
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import agent_routes_yaml, agent_wiki_dir

# severity ranks (for sorting / exit code); only "error" gates CI
SEVERITY = {"error": 2, "warn": 1, "info": 0}

_WIKILINK = re.compile(r"\[\[([^\]]+)\]\]")
_CODE_SPAN = re.compile(r"```.*?```|`[^`]*`", re.S)  # fenced + inline code — not real links
_NUMERIC = re.compile(r"^-?[\d,]*\.?\d+%?$")  # "1,234" / "3.70%" / "-0.5" — a comparable value
_CELL_VAL = re.compile(r"^\s*(.+?)\s*\[([^\]\s]+)\]\s*$")  # "46,328 [D6]" / "3.70% [FDD8]"


def _norm(term: str) -> str:
    return re.sub(r"\s+", "", str(term)).casefold()


def _link_target(raw: str) -> str:
    """The page name a ``[[...]]`` points at — drop only an Obsidian ``|alias`` suffix.

    NB: do *not* strip a trailing ``#…`` as an anchor — page names in this corpus legitimately
    contain ``#`` (``장표 #1``, ``Key Finding Summary #5``), so an anchor split would mangle a
    valid link into a phantom-broken one.
    """
    return raw.split("|", 1)[0].strip()


# --------------------------------------------------------------------------- page parsing

def _page_items(page_md: str) -> list[dict]:
    """Pull ``value [cell]`` rows from a page's markdown table(s) → ``[{term, period, value}]``.

    Mirrors ``apps.agent.retrieval.tools.extract_page_items`` (kept local so ``src/stella_kb`` stays
    free of an ``apps/`` import) — the compile step renders every fact as ``value [cell]`` in a
    pipe table with a header row, so this is an exact, LLM-free read of what the page asserts.
    """
    out: list[dict] = []
    header: list[str] = []
    for line in page_md.splitlines():
        s = line.strip()
        if not s.startswith("|"):
            header = []  # table ended; the next pipe block re-detects its own header
            continue
        cells = [c.strip() for c in s.strip("|").split("|")]
        if cells and all(c and set(c) <= {"-", ":"} for c in cells):  # the |---|---| separator
            continue
        if not header:  # first pipe row of a block is its header
            header = cells
            continue
        label = cells[0].strip(" *`")
        if not label:
            continue
        for ci, cell in enumerate(cells[1:], start=1):
            m = _CELL_VAL.match(cell)
            if not m:
                continue
            value = m.group(1).strip()
            if not value:
                continue
            hdr = header[ci] if ci < len(header) else ""
            period = hdr if re.match(r"^(FY)?\s*\d{4}", hdr) else ""
            out.append({"term": label, "period": period, "value": value})
    return out


def _wikilinks(text: str) -> set[str]:
    """Real ``[[target]]`` links in ``text`` — code spans stripped first so a ``[[page]]``
    shown as an example inside backticks (e.g. the INDEX.md header) isn't read as a link."""
    text = _CODE_SPAN.sub(" ", text)
    return {_link_target(m.group(1)) for m in _WIKILINK.finditer(text)}


# --------------------------------------------------------------------------- the checks

def _finding(check: str, severity: str, page: str, msg: str, **extra) -> dict:
    return {"check": check, "severity": severity, "page": page, "msg": msg, **extra}


def _known_sheets(index: dict, valid_pages: set[str]) -> set[str]:
    """Names that legitimately exist as workbook sheets even without a wiki page.

    Derived from the sheet-level DAG + the per-page link sets in the index (no workbook open),
    so a ``[[X]]`` to a real-but-out-of-scope engine sheet reads as *dangling*, not *broken*.
    """
    known = set(valid_pages) | set(index.get("pages", {}))
    for s, d in (index.get("sheet_dag") or {}).items():
        known.add(s)
        known.update(d.get("depends_on", []))
        known.update(d.get("feeds_into", []))
    for e in (index.get("pages") or {}).values():
        known.update(e.get("depends_on", []))
        known.update(e.get("feeds_into", []))
    return known


def _check_links(pages_dir: Path, index_md: Path, index: dict,
                 valid_pages: set[str]) -> list[dict]:
    known = _known_sheets(index, valid_pages)
    sources = [(p.stem, p) for p in sorted(pages_dir.glob("*.md"))]
    if index_md.exists():
        sources.append(("INDEX.md", index_md))
    findings: list[dict] = []
    for name, path in sources:
        for tgt in sorted(_wikilinks(path.read_text(encoding="utf-8"))):
            if tgt in valid_pages:
                continue
            if tgt in known:
                findings.append(_finding(
                    "dangling_sheet_link", "info", name,
                    f"links to [[{tgt}]] — a real sheet with no wiki page (out of scope)",
                    target=tgt))
            else:
                findings.append(_finding(
                    "broken_link", "error", name,
                    f"links to [[{tgt}]] — no such page or sheet", target=tgt))
    return findings


def _check_pages(index: dict, valid_pages: set[str]) -> list[dict]:
    findings: list[dict] = []
    indexed = set(index.get("pages", {}))
    for name in sorted(indexed - valid_pages):
        findings.append(_finding("missing_page", "error", name,
                                 "is in index['pages'] but has no pages/<name>.md on disk"))
    for name in sorted(valid_pages - indexed):
        findings.append(_finding("orphan_page", "warn", name,
                                 "has a page file but no index entry (invisible to lookup/ToC)"))
    return findings


def _check_aliases(index: dict, valid_pages: set[str]) -> list[dict]:
    findings: list[dict] = []
    for term, bucket in sorted((index.get("alias_index") or {}).items()):
        for hit in bucket:
            page = hit.get("page")
            if page not in valid_pages:
                findings.append(_finding(
                    "orphan_alias", "error", page or "—",
                    f"alias {hit.get('term', term)!r} → page {page!r} which is not on disk",
                    term=hit.get("term", term), cell=hit.get("cell")))
    return findings


def _check_routes(wiki_dir: Path, valid_pages: set[str]) -> list[dict]:
    path = agent_routes_yaml(wiki_dir)
    if not path.exists():
        return []
    import yaml

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a broken routes file is the agent's problem, not lint's
        return [_finding("stale_route", "warn", "routes.yaml", f"{path} is not valid YAML")]
    if not isinstance(data, dict):
        return []
    findings: list[dict] = []
    for term, v in data.items():
        targets = v if isinstance(v, list) else [v]
        for tgt in targets:
            if tgt and str(tgt) not in valid_pages:
                findings.append(_finding(
                    "stale_route", "warn", str(tgt),
                    f"route {term!r} → {tgt!r} which is not a real page", term=str(term)))
    return findings


def _check_cross_refs(index: dict, valid_pages: set[str]) -> list[dict]:
    """PDF→Excel cross-refs must stay bipartite + directed + inverse-consistent:
    ``derives_from`` only on PDF pages pointing at real Excel pages; ``cited_by`` only on Excel
    pages; and every ``F derives_from E`` mirrored in ``E.cited_by`` (no PDF↔PDF, no drift)."""
    pages = index.get("pages", {})

    def is_pdf(n: str) -> bool:
        return pages.get(n, {}).get("source") == "PDF"

    findings: list[dict] = []
    for name, e in pages.items():
        if e.get("derives_from") and not is_pdf(name):
            findings.append(_finding("cross_ref", "error", name,
                                     "derives_from on a non-PDF page (edge must be PDF→Excel)"))
        if e.get("cited_by") and is_pdf(name):
            findings.append(_finding("cross_ref", "error", name,
                                     "cited_by on a PDF page (edge must be Excel→PDF)"))
        for d in e.get("derives_from") or []:
            tgt = d.get("page") if isinstance(d, dict) else d
            if tgt not in valid_pages:
                findings.append(_finding("cross_ref", "error", str(tgt),
                                         f"derives_from {name!r} → {tgt!r} which is not a real page"))
            elif is_pdf(tgt):
                findings.append(_finding("cross_ref", "error", str(tgt),
                                         f"derives_from {name!r} → a PDF page (PDF↔PDF forbidden)"))
            elif name not in (pages.get(tgt, {}).get("cited_by") or []):
                findings.append(_finding("cross_ref", "warn", str(tgt),
                                         f"derives_from {name!r} → {tgt!r} not mirrored in its cited_by"))
    return findings


def _norm_value(v: str) -> str:
    """Comparable form of a rendered value: drop thousands separators and surrounding space."""
    return v.replace(",", "").strip()


def _check_contradictions(pages_dir: Path, index: dict) -> list[dict]:
    """Same term+period, different value, across two pages in the same section/group/kind/case.

    The (section, group, kind, case) key is the dual-case guard: MGT vs DTT — and any two pages
    in different sections (e.g. an FDD deck vs the Excel model) — carry distinct keys and are
    never compared, so a *legitimate* case/source divergence is not flagged; only an unlabelled
    clash within one logical slot is. Comparison is cross-page only (same-page sub-tables, e.g.
    a KRW and a USD account on one ledger, share a key by design and are excluded).
    """
    meta = index.get("pages", {})
    # (norm_term, period) -> { (section,group,kind,case): { norm_value: [(page, raw_value)] } }
    buckets: dict[tuple, dict[tuple, dict[str, list]]] = {}
    for path in sorted(pages_dir.glob("*.md")):
        page = path.stem
        m = meta.get(page, {})
        slot = (m.get("section"), m.get("group"), m.get("kind"), m.get("case"))
        for it in _page_items(path.read_text(encoding="utf-8")):
            val = it["value"]
            if not _NUMERIC.match(_norm_value(val)):
                continue  # only compare numeric assertions; skip dates / text / blanks
            key = (_norm(it["term"]), it["period"])
            by_slot = buckets.setdefault(key, {}).setdefault(slot, {})
            by_slot.setdefault(_norm_value(val), []).append((page, val))

    findings: list[dict] = []
    for (term, period), by_slot in buckets.items():
        for slot, by_val in by_slot.items():
            pages_in_slot = {p for hits in by_val.values() for p, _ in hits}
            if len(by_val) > 1 and len(pages_in_slot) > 1:  # >1 distinct value from >1 page
                detail = "; ".join(
                    f"{raw} ({pg})" for hits in by_val.values() for pg, raw in hits)
                per = f" [{period}]" if period else ""
                findings.append(_finding(
                    "contradiction", "warn", sorted(pages_in_slot)[0],
                    f"{term!r}{per} disagrees within {slot[0]} / {slot[1]}: {detail}",
                    term=term, period=period, pages=sorted(pages_in_slot)))
    return findings


# --------------------------------------------------------------------------- driver

def lint(wiki_dir: str | Path | None = None, contradictions: bool = False) -> dict:
    """Run the checks over one built wiki dir → a structured report.

    ``{findings: [...], counts: {error,warn,info}, ok: bool, wiki_dir: str}``; ``ok`` is False
    iff any error-severity finding exists (the CI gate). ``contradictions`` (off by default)
    adds the heuristic, noisier value-clash check.
    """
    base = Path(wiki_dir) if wiki_dir else agent_wiki_dir()
    index_json = base / "index.json"
    pages_dir = base / "pages"
    index_md = base / "INDEX.md"

    if not index_json.exists():
        raise FileNotFoundError(f"no index.json under {base} — build the wiki first")
    index = json.loads(index_json.read_text(encoding="utf-8"))
    valid_pages = {p.stem for p in pages_dir.glob("*.md")} if pages_dir.exists() else set()

    findings: list[dict] = []
    findings += _check_links(pages_dir, index_md, index, valid_pages)
    findings += _check_pages(index, valid_pages)
    findings += _check_aliases(index, valid_pages)
    findings += _check_routes(base, valid_pages)
    findings += _check_cross_refs(index, valid_pages)
    if contradictions:
        findings += _check_contradictions(pages_dir, index)

    findings.sort(key=lambda f: (-SEVERITY[f["severity"]], f["check"], f["page"]))
    counts = {s: sum(1 for f in findings if f["severity"] == s) for s in SEVERITY}
    return {"wiki_dir": str(base), "findings": findings, "counts": counts,
            "ok": counts["error"] == 0}


def apply_fixes(wiki_dir: str | Path | None = None) -> dict:
    """Prune the auto-fixable drift (``orphan_alias``, ``missing_page``) from ``index.json``.

    Safe because it only deletes index entries pointing at pages that no longer exist on disk —
    it never rewrites generated prose or curated files. Returns ``{aliases_pruned, pages_pruned,
    buckets_emptied}``. A full rebuild supersedes this; it exists to mend an index served live
    between rebuilds (the incremental path).
    """
    base = Path(wiki_dir) if wiki_dir else agent_wiki_dir()
    index_json = base / "index.json"
    pages_dir = base / "pages"
    index = json.loads(index_json.read_text(encoding="utf-8"))
    valid_pages = {p.stem for p in pages_dir.glob("*.md")} if pages_dir.exists() else set()

    aliases_pruned = buckets_emptied = pages_pruned = 0
    ai = index.get("alias_index") or {}
    for term in list(ai):
        kept = [h for h in ai[term] if h.get("page") in valid_pages]
        aliases_pruned += len(ai[term]) - len(kept)
        if kept:
            ai[term] = kept
        else:
            del ai[term]
            buckets_emptied += 1

    pg = index.get("pages") or {}
    for name in [n for n in pg if n not in valid_pages]:
        del pg[name]
        pages_pruned += 1

    index_json.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"aliases_pruned": aliases_pruned, "buckets_emptied": buckets_emptied,
            "pages_pruned": pages_pruned}


def render_report(report: dict) -> str:
    findings, counts = report["findings"], report["counts"]
    icon = {"error": "✗", "warn": "!", "info": "·"}
    lines = [f"wiki lint: {report['wiki_dir']}",
             f"  {counts['error']} error · {counts['warn']} warn · {counts['info']} info"]
    if not findings:
        lines.append("  clean — no findings.")
        return "\n".join(lines)
    by_check: dict[str, list[dict]] = {}
    for f in findings:
        by_check.setdefault(f["check"], []).append(f)
    for check in sorted(by_check, key=lambda c: -SEVERITY[by_check[c][0]["severity"]]):
        items = by_check[check]
        lines.append(f"\n[{check}] ({len(items)})")
        for f in items:
            lines.append(f"  {icon[f['severity']]} {f['page']}: {f['msg']}")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    do_fix = "--fix" in args
    as_json = "--json" in args
    do_contra = "--contradictions" in args
    pos = [a for a in args if not a.startswith("--")]
    wiki_dir = pos[0] if pos else None

    report = lint(wiki_dir, contradictions=do_contra)
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(render_report(report))

    if do_fix:
        fixed = apply_fixes(wiki_dir)
        print(f"\n--fix: pruned {fixed['aliases_pruned']} orphan alias hit(s) "
              f"({fixed['buckets_emptied']} term(s) emptied), {fixed['pages_pruned']} "
              f"missing page entr(y/ies) from index.json")
        report = lint(wiki_dir, contradictions=do_contra)  # re-lint so exit code reflects fixes

    sys.exit(0 if report["ok"] else 1)
