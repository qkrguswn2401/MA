"""Ingest a PDF report into wiki pages — the counterpart to the Excel pipeline.

A summary PDF (e.g. an FDD Executive Summary) is the *other* document in a PDF×Excel
cross-check. This stage lifts the PDF into wiki pages so the agent can open a PDF page
**and** an Excel page and compare — the same retrieval path, no agent changes.

Flow (mirrors dump_md -> parse_llm -> compile):
  1. the **vision PDF parser** (``parsers.pdf.describe_pdf`` — gemma multimodal) reads each
     page *image* into faithful **markdown** (tables as pipe-rows, charts, reading order).
     Slide-deck FDD reports parse far better this way than via text extraction.
  2. each PDF **page** becomes one **section** (``pdf_to_sections``); its label comes from the
     page's ``# Executive Summary | <name>`` heading — Company Snapshot, Key Finding Summary,
     Valuation Summary, … — one wiki page per PDF page.
  3. the LLM structures each section into {title, aliases, figures[], summary} — it interprets,
     never transcribes numbers (values copied verbatim from the markdown, CLAUDE.md rule).
  4. every figure value is **grounded**: its digits must appear in the section text, else dropped.
  5. each value is rendered with a ``[<tag>]`` source marker (the PDF analogue of the Excel
     ``[J6]`` cell) so the retriever's cell-on-page guard passes unchanged.

``build_pages`` returns the index pieces (page entries, alias additions, tree section) for the
caller to merge into an existing wiki ``index.json`` next to the Excel pages.

Requires the vision endpoint (gemma-4 vLLM, ``STELLA_LLM_URL``) + PyMuPDF/pdfplumber.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import pdf_structure_cache
from ..llm import _json_span, cached_chat
from ..prompts import load as load_prompt

_SYSTEM = load_prompt("pdf_page_system")
_DOC_SYSTEM = load_prompt("pdf_doc_system")
SECTION = "FDD 요약 보고서 (PDF)"
_HEADING = re.compile(r"^#{1,3}\s+(.*\S)\s*$", re.M)
# FDD pages carry an "Executive Summary | <topic>" breadcrumb — match it anywhere (heading or
# body). The capture stops at the next `|`/newline, so a trailing pipe never leaks into label.
_EXEC_CRUMB = re.compile(r"Executive Summary\s*\|\s*([^\n|]+)")


def _label_from_page(md: str) -> str:
    """Derive a section label from a vision page.

    Prefer the ``Executive Summary | <topic>`` breadcrumb (present on every FDD page, whether
    or not it's a markdown heading); fall back to the first ``#``/``##``/``###`` heading. Strips
    the boilerplate prefix, any ``[FDD]`` suffix, and stray trailing separators (``|``, ``-``)."""
    m = _EXEC_CRUMB.search(md) or _HEADING.search(md)
    if not m:
        return ""
    label = re.sub(r"Executive Summary\s*\|", "", m.group(1)).strip()
    label = re.sub(r"\s*\[FDD\].*$", "", label).strip()
    label = re.sub(r"[*_`]+", "", label)  # strip markdown emphasis (**bold**, _em_, `code`)
    return label.strip(" |·-\t")


def pdf_to_sections(pdf_path: str, min_chars: int = 200) -> list[tuple[str, str]]:
    """Vision-parse the PDF (gemma multimodal) into ``[(label, body), ...]`` — one per page.

    The vision parser emits faithful per-page markdown (tables, charts, reading order) for
    these slide-deck FDD reports, so each PDF page becomes one wiki section. Short pages
    (covers/dividers, < ``min_chars``) are dropped; a duplicate label gets a ``#n`` suffix so
    page names stay unique."""
    from ..parsers.pdf import describe_pdf

    pages, _ = describe_pdf(pdf_path)
    out: list[tuple[str, str]] = []
    seen: dict[str, int] = {}
    for sp in pages:
        body = sp.text.strip()
        if len(body) < min_chars:
            continue
        label = _label_from_page(body) or f"페이지 {sp.page}"
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} #{seen[label]}"
        out.append((label, body))
    return out


def _clean(s: object) -> str:
    return re.sub(r"[\s,]", "", str(s))


def _grounded(value: str, text: str) -> bool:
    """Keep a figure only if its (comma/space-stripped) value occurs in the section text —
    the OpenKB whitelist idea applied to PDF: the model may label, but can't invent numbers."""
    v = _clean(value)
    return len(v) >= 2 and v in _clean(text)


# --- FDD -> Excel cross-references (deterministic) ----------------------------------------
_FUND_NUM = re.compile(r"(\d+(?:-\d+)?)\s*호")  # number tied to 호 — used on clean group names
# A fund *reference* inside FDD prose: a 호-number whose name is closed by PEF/펀드/Fund within a
# short window. Requiring the fund word avoids matching legal/accounting clause numbers
# ("제2조 제3호") as funds — the generic guard against number-only false positives.
_FUND_REF = re.compile(r"(\d+(?:-\d+)?)\s*호[^,\n]{0,20}?(?:PEF|펀드|Fund)")


def _norm(term: object) -> str:
    return re.sub(r"\s+", "", str(term)).casefold()


def _fund_match(group: str, blob_flat: str, blob_nums: set[str]) -> bool:
    """Does an Excel per-fund **group name** identify the fund an FDD page references?

    Derived from the group name itself — no hardcoded fund list, so it generalizes to whatever
    per-fund groups exist: match on the **name-core** (``차이나1호`` → ``차이나``) as a substring,
    or on the **호-number** (``제2호`` → ``2``) shared with the page. Handles the naming variance
    (Excel ``차이나1호`` vs FDD ``센트로이드제1호차이나PEF``) without enumerating either side."""
    core = re.sub(r"[제\s&]", "", _FUND_NUM.sub("", str(group)))
    gnums = set(_FUND_NUM.findall(str(group)))
    return (len(core) >= 2 and core in blob_flat) or bool(gnums & blob_nums)


def _xrefs(entry: dict, index: dict, cap: int = 6) -> list[str]:
    """Excel source pages an FDD page cross-references, via **fund identity** — deterministic.

    A Biz Plan fund group whose name-core or 호-number (both derived from the group name, never
    hardcoded) appears in this page's figure labels/aliases → that fund's source pages
    (거래내역/비용). This is the bridge the alias index *can't* make: an FDD page names a fund as
    '센트로이드제1호차이나PEF' while the Excel group is '차이나1호', and ledger rows aren't aliased.
    Shared **line-item** terms (관리수수료/영업수익/Adjusted NAV …) are already cross-linked FDD↔Excel
    by the alias index, so they are deliberately not duplicated here. Capped."""
    pages = index.get("pages", {})
    fund_pages: dict[str, list[str]] = {}
    for nm, e in pages.items():
        section = str(e.get("section", ""))
        if e.get("source") != "PDF" and section.startswith("Biz Plan"):
            group = e.get("group")
            fund_pages.setdefault(group, []).append(nm)

    entry_items = entry.get("items") or []
    entry_aliases = entry.get("aliases") or []
    terms = (
        " ".join(it.get("label") or "" for it in entry_items)
        + " "
        + " ".join(entry_aliases)
    )
    blob_flat, blob_nums = _norm(terms), set(_FUND_REF.findall(terms))
    out: list[str] = []
    for group, fpages in fund_pages.items():
        if _fund_match(group, blob_flat, blob_nums):
            out.extend(nm for nm in fpages if nm not in out)
    return out[:cap]


def structure_section(label: str, text: str, timeout: float = 600.0) -> dict:
    """LLM-structure one section's markdown; drop ungrounded figures. ``{}`` if unusable."""
    raw = cached_chat(
        [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"PDF 섹션: {label!r}\n\n{text}\n\nJSON:"},
        ],
        cache_dir=pdf_structure_cache(),
        max_tokens=4500,
        timeout=timeout,
    )
    obj = _json_span(raw, "{", "}")
    if not isinstance(obj, dict):
        return {}
    raw_figures = obj.get("figures") or []
    obj["figures"] = [
        f for f in raw_figures
        if isinstance(f, dict) and f.get("value") and _grounded(f["value"], text)
    ]
    return obj


_NUMERIC = re.compile(r"^[\d\s.,%xX()\-+]+$")


def _extract_tables(md: str) -> list[str]:
    """Pull every markdown table block (>=2 consecutive pipe-rows) out of the vision page text.

    The vision parser transcribes charts/matrices/tables faithfully as pipe-rows, but the LLM
    structurer collapses 2-axis grids to row/column aggregates and drops cells it can't fit the
    ``{label,period,value}`` schema. Carrying the raw blocks onto the page verbatim keeps every
    cell (e.g. a 5×5 WACC×PGR heatmap, a 3-D loan matrix) retrievable."""
    blocks, cur = [], []
    for ln in md.splitlines():
        if ln.count("|") >= 2:
            cur.append(ln.rstrip())
        else:
            if len(cur) >= 2:
                blocks.append("\n".join(cur))
            cur = []
    if len(cur) >= 2:
        blocks.append("\n".join(cur))
    return blocks


def _table_terms(tables: list[str], cap: int = 24) -> list[str]:
    """Non-numeric cell texts (column headers + row labels) from the raw tables → alias terms,
    so a recovered table page is routable by its headers/entities (담보유형, peer names, …)."""
    terms: list[str] = []
    seen: set = set()
    for tbl in tables:
        for row in tbl.splitlines():
            for c in row.split("|"):
                c = re.sub(r"\s+", " ", c).strip(" *`")
                if 2 <= len(c) <= 30 and "--" not in c and not _NUMERIC.match(c):
                    k = _norm(c)
                    if k and k not in seen:
                        seen.add(k)
                        terms.append(c)
    return terms[:cap]


def _page_md(name: str, tag: str, label: str, s: dict, xref: list[str] | None = None,
             body: str | None = None) -> str:
    title = s.get("title") or label
    # aliases are NOT written to page frontmatter — the alias_index (index.json) is the resolver,
    # built from the parsed items; the page md would only be dead weight (open_page strips it).
    out = ["---", "source: PDF", f"page: {name}", f"tag: {tag}", f"section: {label}"]
    out += [
        "---",
        "",
        f"# {name}",
        "",
        f"> 출처: FDD&Valuation Report Executive Summary — {label} (`{tag}`). "
        "**PDF 요약 수치이며 엑셀 원천과 정의·기준이 다를 수 있습니다**(예: 영업수익 Total은 "
        "배당금 포함; 보고서 기준일은 Jun-24).",
        "",
        "## What this is",
        "",
        (s.get("summary") or "_(요약 없음)_"),
        "",
        "## Key figures (PDF 보고서 수치)",
        "",
        "| 항목 | 기간 | value |",
        "|---|---|---|",
    ]
    figures = s.get("figures") or []
    for f in figures:
        out.append(f"| {f.get('label','')} | {f.get('period','') or ''} | {f.get('value','')} [{tag}] |")
    # Full vision markdown verbatim — every cell, [그래프] block, and [다이어그램] edge-list the
    # structurer didn't lift (matrices, dense tables, org/structure diagrams). The marker line
    # puts `[tag]` on the page so the retriever can cite any of these values.
    if body and body.strip():
        out += ["", "## 원문 (vision 원문 — 모든 표·그래프·다이어그램)", "",
                f"> 아래 원문의 모든 수치·관계 출처: `[{tag}]` (리포트 페이지 원문).", "",
                body.strip(), ""]
    out += ["## Links", ""]
    if xref:
        out.append("- 엑셀 원천 (교차검증 대상): " + ", ".join(f"[[{x}]]" for x in xref))
    out.append("- PDF 요약 — 동일 항목의 **엑셀 원천 페이지와 교차검증** 대상 (단위·기준일 차이 주의).")
    return "\n".join(out) + "\n"


def build_pages(
    pdf_path: str, pages_dir: Path, structurer=structure_section, index: dict | None = None, doc: str | None = None
) -> tuple[dict, dict, dict]:
    """Build PDF wiki pages and the index pieces to merge into an existing wiki index.

    ``doc`` namespaces the pages to one source report (e.g. ``"CAESAR"``). When several FDD
    PDFs are ingested into one wiki, their per-PDF ``FDD{n}`` numbering and labels collide
    (two decks both have an ``FDD6 — Valuation Summary``); prefixing the page name and ToC
    group with ``[{doc}]`` keeps names unique **and** makes each page's deal identity explicit.
    ``doc=None`` (single-PDF build) preserves the original ``FDD{n} — {label}`` naming.

    Returns ``(pages_entries, alias_additions, tree_section)`` and writes ``<name>.md`` into
    ``pages_dir``. One page per PDF page; each tagged ``FDD<n>`` for provenance. When ``index``
    (the Excel-side wiki index) is given, each page also gets deterministic ``xref`` links to
    its Excel source pages (see :func:`_xrefs`) — written into the entry and the page's Links
    section, so the agent can hop from an FDD claim to its source ledger.
    """
    from concurrent.futures import ThreadPoolExecutor

    pages_dir.mkdir(parents=True, exist_ok=True)
    sections = pdf_to_sections(pdf_path)

    with ThreadPoolExecutor(max_workers=6) as ex:  # one LLM call per section, bounded
        structured = list(ex.map(lambda ls: (ls[0], ls[1], structurer(ls[0], ls[1])), sections))

    entries: dict[str, dict] = {}
    aliases: dict[str, list] = {}
    tree: dict[str, dict] = {SECTION: {}}
    for i, (label, text, s) in enumerate(structured, 1):  # number by section position (stable)
        figs = s.get("figures") or []
        raw_tables = _extract_tables(text)            # the page's full grids, verbatim
        s_aliases = s.get("aliases") or []
        page_aliases = [a for a in s_aliases if a]
        page_aliases += [t for t in _table_terms(raw_tables) if _norm(t) not in
                         {_norm(a) for a in page_aliases}]   # + header/row-label terms
        # Only a genuinely empty page (no figures, no aliases, no tables — a cover/divider) is
        # dropped. Dense-table pages the structurer couldn't parse (Key Financial Information,
        # GPC peer table) used to vanish here; now they're kept via their raw tables/terms.
        if not figs and not page_aliases and not raw_tables:
            continue
        # When the breadcrumb regex missed and the label fell back to "페이지 N", use the LLM's
        # structured title instead — an uninformative "페이지 N" page name/ToC entry is unroutable
        # (the router can't tell it's the Corporate Structure page from "페이지 2").
        s_title = s.get("title")
        if re.fullmatch(r"페이지 \d+", label) and s_title:
            label = re.sub(r"\s+", " ", s_title).strip()
        tag = f"FDD{i}"
        # A label can carry '/' (e.g. a "(2/2)" continuation marker); '/' in a page name
        # breaks the filesystem write and the page-key↔filename match. Sanitize like the
        # Excel side (dump_md/parse_llm use ``.replace('/', '_')``) so name, file stem, index
        # key and [[wikilinks]] all stay consistent. ``doc`` (multi-PDF) prefixes the name +
        # ToC group so two decks' identically-labelled pages don't collide / overwrite.
        prefix = f"[{doc}] " if doc else ""
        name = f"FDD{i} — {prefix}{label}".replace("/", "_")
        group = f"{prefix}{label}"
        labels = [f.get("label") for f in figs if f.get("label")]
        s_summary = s.get("summary") or ""
        entry = {
            "sheet": name,
            "title": s_title or label,
            "desc": s_summary.split(". ")[0][:120] or label,
            "section": SECTION,
            "group": group,
            # The Stella deck's reporting window; unknown/mixed across decks in a multi-PDF
            # build, so don't stamp it on other deals' pages.
            "period": None if doc else "Dec-20–Jun-24",
            "n_items": len(figs),
            "has_page": True,
            "aliases": page_aliases,
            "items": [{"label": lb, "ko": None, "cell": tag, "role": "pdf"} for lb in labels],
            "depends_on": [],
            "feeds_into": [],
            "source": "PDF",
        }
        entry["xref"] = _xrefs(entry, index) if index else []
        (pages_dir / f"{name}.md").write_text(
            _page_md(name, tag, label, s, entry["xref"], text), encoding="utf-8")
        entries[name] = entry
        for term in page_aliases + labels:
            aliases.setdefault(_norm(term), []).append({"page": name, "cell": tag, "term": term})
        tree[SECTION].setdefault(group, []).append(name)

    return entries, aliases, tree


def _fdd_num(name: str) -> int:
    m = re.match(r"FDD(\d+)", name)
    return int(m.group(1)) if m else 0


def build_document(doc: str, entries: dict, curated: dict | None = None) -> dict:
    """Build the per-PDF **document node** — the two-layer index for one source deck.

    Upper layer: a detailed description of the whole report (what company/deal, which sections,
    what chart/matrix types, key figures) — so the router can pick the right *document* from a
    question that only names the project. Lower layer: a table of contents of the deck's pages
    (FDD#, title, one-line summary) — to then pick the right *page*.

    The upper layer is **curated > LLM > default**: a ``curated`` dict (one deck's block from
    ``decks.yaml``) pins ``title``/``description`` by hand; whichever field it omits is filled by
    the LLM, grounded in the deck's own page titles/summaries/figure labels (no new facts). When
    both are pinned the LLM call is skipped entirely — the document node is then fully
    deterministic. The lower-layer ToC is always derived from the pages.
    """
    curated = curated or {}
    curated_title = curated.get("title")
    curated_desc = curated.get("description")

    items = sorted(entries.items(), key=lambda kv: _fdd_num(kv[0]))
    toc = [{"page": n, "title": e.get("title") or n,
            "summary": e.get("desc") or ""}
           for n, e in items]
    title = curated_title or f"{doc} 보고서"
    description = curated_desc or ""

    if not (curated_title and curated_desc):  # something still LLM-filled
        digest = "\n".join(
            f"- {e.get('title') or n}: {e.get('desc') or ''}"
            + (f"  [항목: {', '.join((it.get('label') or '') for it in (e.get('items') or [])[:8])}]"
               if e.get("items") else "")
            for n, e in items)
        try:
            raw = cached_chat(
                [{"role": "system", "content": _DOC_SYSTEM},
                 {"role": "user", "content": f"보고서(프로젝트): {doc}\n\n페이지 목록:\n{digest}\n\nJSON:"}],
                cache_dir=pdf_structure_cache(),
                max_tokens=1200,
                timeout=300,
            )
            obj = _json_span(raw, "{", "}")
            if isinstance(obj, dict):
                title = curated.get("title") or obj.get("title") or title
                description = curated.get("description") or obj.get("description") or ""
        except Exception:  # noqa: BLE001 — a deck still gets a ToC even if the description fails
            pass
    return {"doc": doc, "title": title, "n_pages": len(toc),
            "description": description, "toc": toc}


def _load_decks() -> dict:
    """Load the curated first-layer index (``decks.yaml``) → ``{doc: {title?, description?}}``.

    Absent file → ``{}`` (pure-LLM build, unchanged). Tolerant of an empty or malformed file."""
    import yaml

    from ..config import wiki_decks_yaml

    path = wiki_decks_yaml()
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:  # noqa: BLE001 — a bad curated file must not break the build
        return {}
    return {k: v for k, v in data.items() if isinstance(v, dict)} if isinstance(data, dict) else {}


def strip_pdf(index: dict) -> dict:
    """Remove all PDF artifacts from a wiki index (pages, their alias entries, the PDF tree
    section, the per-deck document nodes) so a rebuild replaces cleanly."""
    pdf = {n for n, e in index["pages"].items() if e.get("source") == "PDF"}
    for n in pdf:
        del index["pages"][n]
    index["tree"].pop(SECTION, None)
    index.pop("documents", None)
    ai = index["alias_index"]
    for key in list(ai):
        kept = [h for h in ai[key] if h["page"] not in pdf]
        if kept:
            ai[key] = kept
        else:
            del ai[key]
    return index


def merge_into_index(index: dict, entries: dict, alias_add: dict, tree_add: dict) -> dict:
    """Merge PDF pieces into a loaded wiki index dict (in place) and return it."""
    index["pages"].update(entries)
    ai = index["alias_index"]
    for key, bucket in alias_add.items():
        existing = ai.setdefault(key, [])
        existing_pairs = {(h["page"], h["cell"]) for h in existing}
        existing.extend(b for b in bucket if (b["page"], b["cell"]) not in existing_pairs)
    for section, groups in tree_add.items():
        dst = index["tree"].setdefault(section, {})
        for g, names in groups.items():
            dst.setdefault(g, []).extend(nm for nm in names if nm not in dst.get(g, []))
    return index


if __name__ == "__main__":
    # Stage 5 of run_pipeline.sh: ingest every PDF report under data/raw/ and merge its pages
    # into the index the Excel pipeline already built (stage 4). Self-skips cleanly when there
    # is no PDF, so the stage is always safe to run. Mirrors eval.stella_crosscheck.build_pdf.
    import json
    import sys

    from ..config import wiki_pdf_dir
    from .index import OUT_JSON, OUT_MD, PAGES_DIR, render_md

    pdf_dir = wiki_pdf_dir()
    pdfs = [str(p) for p in sorted(pdf_dir.glob("*.pdf"))]
    if len(sys.argv) > 1:  # explicit path(s) override the glob
        pdfs = sys.argv[1:]
    if not pdfs:
        print(f"pdf_pages: no {pdf_dir}/*.pdf — skipping PDF ingest.")
        sys.exit(0)
    if not OUT_JSON.exists():
        sys.exit(f"pdf_pages: {OUT_JSON} not found — run the index stage (4) first.")

    for stale in PAGES_DIR.glob("FDD*.md"):  # clean slate so a rebuild replaces
        stale.unlink()
    index = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    index = strip_pdf(index)  # drop any prior PDF entries first
    # Namespace pages by source report only when ingesting several PDFs — a single-PDF build
    # keeps the original FDD{n} — {label} names. ``CAESAR_pages.pdf`` -> doc ``CAESAR``.
    multi = len(pdfs) > 1
    decks = _load_decks()  # curated first layer (decks.yaml); {} = pure-LLM, unchanged
    if decks:
        print(f"pdf_pages: curated deck overrides for {sorted(decks)}")
    documents: dict = {}
    for pdf in pdfs:
        doc = re.sub(r"[_-]?pages$", "", Path(pdf).stem, flags=re.I)
        name_doc = doc if multi else None      # namespace page NAMES only in the multi-deck case
        print(f"pdf_pages: ingest {pdf}  (doc={doc})")
        entries, alias_add, tree_add = build_pages(pdf, PAGES_DIR, index=index, doc=name_doc)
        merge_into_index(index, entries, alias_add, tree_add)
        # two-layer node: deck description (curated > LLM > default) + ToC
        documents[doc] = build_document(doc, entries, curated=decks.get(doc))
        print(f"   built {len(entries)} PDF page(s) + document node '{documents[doc]['title']}'")
    index["documents"] = documents

    # re-dedup: the FDD merge above adds page aliases (incl. scaffolding like Key Issue/Category)
    # AFTER build_index's dedup, so clean the merged result again before persisting.
    from ..config import alias_stopwords
    from .dedup import dedup_alias_index
    dedup_alias_index(index, tuple(alias_stopwords()))

    # directed PDF→Excel cross-refs (derives_from / cited_by) — both page sets now coexist
    from ..config import cross_ref_llm_judge
    from .cross_refs import build_cross_refs
    judge = None
    if cross_ref_llm_judge():
        from .cross_refs import make_llm_judge
        judge = make_llm_judge()
    cx = build_cross_refs(index, judge=judge)
    print(f"pdf_pages: cross_refs PDF→Excel — {cx['edges']} edge(s), "
          f"{cx['pdf_with_links']} FDD page(s) linked, {cx['excel_cited']} Excel page(s) cited")

    OUT_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(index), encoding="utf-8")
    print(f"pdf_pages: merged -> {OUT_JSON}  (pages={len(index['pages'])}, " f"aliases={len(index['alias_index'])})")
