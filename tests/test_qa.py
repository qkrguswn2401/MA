"""Query-compounding: persisting a grounded agent answer onto its wiki page.

Deterministic + offline. Covers the sidecar store, the grounding gate, idempotent section
rendering (so `compile` can re-render on every rebuild without duplicating), and the agent's
`persist_answer` write path including the ungrounded/missing-page refusals.
"""

from __future__ import annotations

from pathlib import Path

from src.stella_kb.wiki import qa
from apps.agent.retrieval.tools import persist_answer

EV = [{"page": "DCF", "cell": "K59", "term": "EV", "value": "120,696"},
      {"page": "DCF", "cell": "K60", "term": "equity", "value": "206,131"},
      {"page": "AUM", "cell": "B12", "term": "aum", "value": "100"}]


# --------------------------------------------------------------------------- store + gate

def test_append_and_load_roundtrip(tmp_path):
    e = qa.new_entry("what drives EV?", "FCFF로 결정됩니다 [K59]", EV)
    qa.append_qa(tmp_path, "DCF", e)
    qa.append_qa(tmp_path, "DCF", qa.new_entry("두번째?", "답 [K60]", EV))
    loaded = qa.load_qa(tmp_path, "DCF")
    assert len(loaded) == 2 and loaded[0]["question"] == "what drives EV?"


def test_is_grounded_requires_answer_and_a_cell():
    assert qa.is_grounded(qa.new_entry("q", "a [K59]", EV))
    assert not qa.is_grounded(qa.new_entry("q", "a", []))           # no evidence
    assert not qa.is_grounded(qa.new_entry("q", "", EV))            # no answer
    assert not qa.is_grounded(qa.new_entry("q", "(no answer)", EV))  # sentinel non-answer


def test_target_page_is_the_most_cited():
    assert qa.target_page(EV) == "DCF"                              # 2 DCF cells vs 1 AUM


# --------------------------------------------------------------------------- rendering

def test_upsert_inserts_replaces_and_removes(tmp_path):
    page = "---\nsheet: DCF\n---\n\n# DCF\n\n## Links\n- **Depends on:** —\n"
    e1 = [qa.new_entry("q1", "a1 [K59]", EV)]
    once = qa.upsert_qa_section(page, e1, target_page="DCF")
    assert once.count(qa.SECTION) == 1 and "q1" in once and "`K59`" in once
    assert "AUM!B12" in once                                        # off-page cite keeps its page

    # idempotent re-render (what compile does each rebuild) — no duplicate section
    twice = qa.upsert_qa_section(once, e1, target_page="DCF")
    assert twice.count(qa.SECTION) == 1

    # adding an entry grows the same section, doesn't add a second
    grown = qa.upsert_qa_section(once, e1 + [qa.new_entry("q2", "a2 [K60]", EV)], "DCF")
    assert grown.count(qa.SECTION) == 1 and "q2" in grown

    # clearing the sidecar removes the section, preserving the page body
    cleared = qa.upsert_qa_section(grown, [], "DCF")
    assert qa.SECTION not in cleared and "## Links" in cleared


# --------------------------------------------------------------------------- write path

def _wiki(tmp_path: Path) -> Path:
    base = tmp_path / "wiki"
    (base / "pages").mkdir(parents=True)
    (base / "pages" / "DCF.md").write_text("---\nsheet: DCF\n---\n\n# DCF\n\n## Links\n- —\n",
                                           encoding="utf-8")
    return base


def test_persist_answer_compounds_onto_the_page(tmp_path):
    base = _wiki(tmp_path)
    out = persist_answer("what drives EV?", "FCFF가 EV를 결정합니다 [K59]", EV, wiki_dir=base)
    assert out["ok"] and out["page"] == "DCF" and out["n_qa"] == 1

    page_md = (base / "pages" / "DCF.md").read_text(encoding="utf-8")
    assert qa.SECTION in page_md and "what drives EV?" in page_md and "`K59`" in page_md
    assert qa.load_qa(base, "DCF")                                  # sidecar is the source of truth


def test_persist_answer_refuses_ungrounded(tmp_path):
    base = _wiki(tmp_path)
    out = persist_answer("q", "근거 없는 답", [], wiki_dir=base)        # no evidence
    assert not out["ok"] and "ungrounded" in out["reason"]
    assert qa.SECTION not in (base / "pages" / "DCF.md").read_text(encoding="utf-8")


def test_persist_answer_refuses_missing_target_page(tmp_path):
    base = _wiki(tmp_path)
    ev = [{"page": "Ghost", "cell": "Z9", "term": "x", "value": "1"}]
    out = persist_answer("q", "답 [Z9]", ev, wiki_dir=base)
    assert not out["ok"] and "not found" in out["reason"]
