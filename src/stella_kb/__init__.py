"""Stella KB — build a property-graph knowledge base from the Project Stella valuation model.

Pipeline:
    extract.py  Excel workbook -> cell-level dependency DAG (the native DEPENDS_ON edges)
    graph.py    cell DAG + cached values -> semantic property graph (Entity/Fund/Metric/...)

See CLAUDE.md for the target node/edge schema.
"""

from pathlib import Path

# repo root is two levels up from this file: src/stella_kb/__init__.py -> MA/
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
# The raw source workbook (inputs/ledgers/statements/exhibits + macro); the computed
# Fin.Model engine sheets (DCF, AUM Projection, ...) are not present in this file.
# This is the canonical source for the **wiki** paradigm.
WORKBOOK = str(
    DATA_DIR / "v0.1" / "raw" / "Project Stella_Valuation Model_251103_vShared(Updated)_raw.xlsx"
)

# The full 63-sheet model, which DOES contain the Fin.Model engine sheets. The **graph**
# paradigm (extract/semantic/metrics) and curated engine-only wiki pages read from here;
# `_raw` is a strict subset, so this workbook covers everything `WORKBOOK` does too.
FULL_WORKBOOK = str(
    DATA_DIR / "v0.1" / "raw" / "Project Stella_Valuation Model_251103_vShared(Updated).xlsx"
)
