"""Wiki lint pass — deterministic, offline. Builds tiny synthetic wikis in tmp_path and
plants one fault per check, since a freshly-built real wiki is (correctly) clean.

Mirrors the artifact shape lint reads: ``<wiki>/index.json`` (pages / alias_index /
sheet_dag) + ``<wiki>/pages/*.md`` + ``<wiki>/INDEX.md``.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.stella_kb.wiki import lint as L


def _page(section="S", group="G", kind="k", case=None, title="T", body="body") -> str:
    fm = ["---", f"section: {section}", f"group: {group}", f"kind: {kind}"]
    if case:
        fm.append(f"case: {case}")
    fm += ["---", "", f"# {title}", "", "## What this is", body, ""]
    return "\n".join(fm) + "\n"


def _facts(rows: list[tuple[str, str, str]]) -> str:
    """A `value [cell]` facts table for the contradiction check: rows of (term, period, val)."""
    out = ["## Line items", "", "| Item | 2024 |", "|---|---|"]
    for i, (term, _period, val) in enumerate(rows):
        out.append(f"| {term} | {val} [C{i}] |")  # pages render every fact as `value [cell]`
    return "\n".join(out) + "\n"


def _build(tmp: Path, pages: dict[str, str], index: dict, index_md: str = "") -> Path:
    base = tmp / "wiki"
    (base / "pages").mkdir(parents=True)
    for name, md in pages.items():
        (base / "pages" / f"{name}.md").write_text(md, encoding="utf-8")
    (base / "index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    (base / "INDEX.md").write_text(index_md, encoding="utf-8")
    return base


def _checks(report: dict) -> set[str]:
    return {f["check"] for f in report["findings"]}


# --------------------------------------------------------------------------- happy path

def test_clean_wiki_has_no_findings(tmp_path):
    pages = {"A": _page() + "\n[[B]]\n", "B": _page()}
    index = {"pages": {"A": {"section": "S", "group": "G", "kind": "k"},
                       "B": {"section": "S", "group": "G", "kind": "k"}},
             "alias_index": {"alpha": [{"page": "A", "cell": "A1", "term": "alpha"}]},
             "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert report["ok"] and not report["findings"]


# --------------------------------------------------------------------------- links

def test_broken_link_is_an_error(tmp_path):
    pages = {"A": _page() + "\nsee [[Nope]]\n"}
    index = {"pages": {"A": {}}, "alias_index": {}, "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert not report["ok"]
    assert "broken_link" in _checks(report)


def test_dangling_link_to_real_sheet_is_info_not_error(tmp_path):
    # 'Engine' has no page but is a real sheet (appears in the DAG) -> dangling, not broken
    pages = {"A": _page() + "\nflows to [[Engine]]\n"}
    index = {"pages": {"A": {}}, "alias_index": {},
             "sheet_dag": {"A": {"depends_on": [], "feeds_into": ["Engine"]}}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert report["ok"]  # info-only, no error
    assert "dangling_sheet_link" in _checks(report)
    assert "broken_link" not in _checks(report)


def test_hash_in_page_name_is_not_an_anchor(tmp_path):
    # '[[Foo #1]]' points at a real page named 'Foo #1' — the '#1' must not be split off
    pages = {"Foo #1": _page(), "A": _page() + "\nsee [[Foo #1]]\n"}
    index = {"pages": {"Foo #1": {}, "A": {}}, "alias_index": {}, "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert "broken_link" not in _checks(report)


def test_wikilink_inside_code_span_is_ignored(tmp_path):
    pages = {"A": _page(body="open the `[[page]]` then follow links")}
    index = {"pages": {"A": {}}, "alias_index": {}, "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert "broken_link" not in _checks(report)


# --------------------------------------------------------------------------- pages / aliases

def test_missing_page_and_orphan_alias_are_errors(tmp_path):
    pages = {"A": _page()}  # 'B' is indexed + aliased but has no file
    index = {"pages": {"A": {}, "B": {}},
             "alias_index": {"beta": [{"page": "B", "cell": "B2", "term": "beta"}]},
             "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert not report["ok"]
    checks = _checks(report)
    assert "missing_page" in checks and "orphan_alias" in checks


def test_orphan_page_is_a_warning(tmp_path):
    pages = {"A": _page(), "Z": _page()}  # 'Z' on disk but not in the index
    index = {"pages": {"A": {}}, "alias_index": {}, "sheet_dag": {}}
    base = _build(tmp_path, pages, index)
    report = L.lint(base)
    assert report["ok"]  # warn only
    assert "orphan_page" in _checks(report)


# --------------------------------------------------------------------------- contradiction

def test_contradiction_is_opt_in_and_dual_case_aware(tmp_path):
    # same term+period, same slot, different value -> a clash; different `case` -> exempt
    pages = {
        "P1": _page(section="X", group="g", kind="m", title="P1") + _facts([("ev", "2024", "100")]),
        "P2": _page(section="X", group="g", kind="m", title="P2") + _facts([("ev", "2024", "200")]),
        "MGT": _page(section="X", group="g", kind="m", case="MGT", title="MGT") + _facts([("nav", "2024", "10")]),
        "DTT": _page(section="X", group="g", kind="m", case="DTT", title="DTT") + _facts([("nav", "2024", "20")]),
    }
    index = {"pages": {n: {"section": "X", "group": "g", "kind": "m",
                           "case": ("MGT" if n == "MGT" else "DTT" if n == "DTT" else None)}
                       for n in pages},
             "alias_index": {}, "sheet_dag": {}}
    base = _build(tmp_path, pages, index)

    assert "contradiction" not in _checks(L.lint(base))               # off by default
    report = L.lint(base, contradictions=True)
    clashes = [f for f in report["findings"] if f["check"] == "contradiction"]
    assert any(f["term"] == "ev" for f in clashes)                    # P1 vs P2 flagged
    assert not any(f["term"] == "nav" for f in clashes)               # MGT vs DTT exempt
    assert report["ok"]                                               # contradictions are warns


# --------------------------------------------------------------------------- --fix

def test_apply_fixes_prunes_orphan_alias_and_missing_page(tmp_path):
    pages = {"A": _page()}
    index = {"pages": {"A": {}, "B": {}},
             "alias_index": {"beta": [{"page": "B", "cell": "B2", "term": "beta"},
                                      {"page": "A", "cell": "A1", "term": "beta"}]},
             "sheet_dag": {}}
    base = _build(tmp_path, pages, index)

    fixed = L.apply_fixes(base)
    assert fixed["aliases_pruned"] == 1 and fixed["pages_pruned"] == 1

    after = json.loads((base / "index.json").read_text(encoding="utf-8"))
    assert "B" not in after["pages"]                                  # missing page entry dropped
    assert all(h["page"] == "A" for h in after["alias_index"]["beta"])  # orphan alias hit dropped
    assert L.lint(base)["ok"]                                          # clean after fix
