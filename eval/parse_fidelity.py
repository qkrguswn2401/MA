"""Measure **PDF parsing fidelity** in isolation — does the vision parser transcribe the
golden values onto the page at all?

This sits *below* the agent eval (``qa_eval.py``): it answers "did the value make it out of
the PDF into the parsed markdown?", independent of retrieval / reasoning / the LLM judge. So
it's the clean signal for a *parser* change (DPI, the describe prompt) — and it's deterministic
and offline (reads the vision **cache**; no agent, no judge, no fresh LLM calls unless the
vision cache misses).

Method: for each ``qa.jsonl`` item, pull the numeric tokens out of its ``answer_basis`` +
``ground_truth`` (the values a correct answer cites), drop obvious non-printed ones (RGB color
triples), and check whether each appears in the deck's parsed text (comma-normalized). Reports
overall + per-deck + per-visual_type fidelity and lists the misses.

    python -m eval.parse_fidelity

⚠️ A "miss" is a *candidate*, not proof of a parser bug — the golden text also contains values
that are **derived** (computed gaps/ratios) or **Excel cross-check** figures the slide never
prints. Triage the miss list: a genuine miss is a value the slide shows but the parser dropped
(e.g. a leading-digit truncation).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "test_data" / "v0.2" / "ground_truth" / "qa.jsonl"
DECKS = {
    "STELLA": ROOT / "test_data" / "v0.2" / "STELLA_pages.pdf",
    "CAESAR": ROOT / "test_data" / "v0.2" / "CAESAR_pages.pdf",
    "LIFE": ROOT / "test_data" / "v0.2" / "LIFE_pages.pdf",
}

_NOCOMMA = lambda s: s.replace(",", "")
_TOK = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def golden_tokens(item: dict) -> set[str]:
    """Numeric tokens a correct answer cites — comma-stripped. Keeps only printed-figure-like
    values (has a comma/decimal, or >=4 digits); drops RGB color triples that appear in some
    structure-diagram goldens (``RGB≈187,235,255``)."""
    t = f"{item.get('answer_basis', '')} {item.get('ground_truth', '')}"
    t = re.sub(r"RGB[^)\n]*\)?", " ", t)            # drop "RGB≈187,235,255"
    t = re.sub(r"\d+,\d+,\d+(?=\s*\))", " ", t)      # drop bare "(187,235,255)" triples
    out: set[str] = set()
    for m in _TOK.finditer(t):
        v = _NOCOMMA(m.group())
        if ("." in m.group() or "," in m.group() or len(v) >= 4):
            out.add(v)
    return out


def deck_text() -> dict[str, str]:
    """{deck: parsed markdown of the whole deck} via the vision parser (cache-backed)."""
    from src.stella_kb.parsers.pdf import describe_pdf

    out = {}
    for deck, pdf in DECKS.items():
        pages, _ = describe_pdf(str(pdf))
        out[deck] = _NOCOMMA("\n".join(sp.text for sp in pages))
    return out


def run() -> dict:
    qa = [json.loads(ln) for ln in QUESTIONS.read_text(encoding="utf-8").splitlines() if ln.strip()]
    text = deck_text()
    by_deck: dict[str, list[int]] = defaultdict(lambda: [0, 0])     # [hit, total]
    by_vt: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    misses: list[dict] = []
    for q in qa:
        gold = golden_tokens(q)
        blob = text[q["doc"]]
        notfound = sorted(v for v in gold if v not in blob)
        by_deck[q["doc"]][0] += len(gold) - len(notfound); by_deck[q["doc"]][1] += len(gold)
        by_vt[q["visual_type"]][0] += len(gold) - len(notfound); by_vt[q["visual_type"]][1] += len(gold)
        if notfound:
            misses.append({"id": q["id"], "doc": q["doc"], "visual_type": q["visual_type"],
                           "missing": notfound})
    return {"by_deck": by_deck, "by_vt": by_vt, "misses": misses}


def _pct(hit_total: list[int]) -> str:
    h, n = hit_total
    return f"{h}/{n} = {h / n:.0%}" if n else "0/0"


if __name__ == "__main__":
    r = run()
    print("=== PDF parser fidelity — golden value present in parsed markdown? ===\n")
    tot = [0, 0]
    print("by deck:")
    for d in ("STELLA", "CAESAR", "LIFE"):
        print(f"  {d:8} {_pct(r['by_deck'][d])}")
        tot[0] += r["by_deck"][d][0]; tot[1] += r["by_deck"][d][1]
    print(f"  {'ALL':8} {_pct(tot)}\n")
    print("by visual_type:")
    for vt in sorted(r["by_vt"]):
        print(f"  {vt:18} {_pct(r['by_vt'][vt])}")
    print(f"\n{len(r['misses'])} item(s) with miss candidates (triage: derived / Excel-xref / genuine):")
    for m in r["misses"]:
        print(f"  {m['id']:6} [{m['doc']}/{m['visual_type']}]  {m['missing']}")
