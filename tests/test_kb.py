"""KB-build helpers: sheet classification and the per-fund carry anchors.

The carry-value tests pin the curated cells against the workbook so a model revision that
shifts a column or row trips a test instead of silently corrupting an answer. Carry now
lives in the **graph** paradigm (`graph/metrics.py`), not the wiki — the engine sheet it
reads (`성과보수, 배당금`) is absent from the `_raw` wiki workbook.
"""

from __future__ import annotations

import networkx as nx
import pytest

from src.stella_kb.graph.metrics import CARRY_FUNDS, CARRY_SHEET, attach_metrics
from src.stella_kb.wiki.index import classify


# --- classify: sheet name → section/group via tokens ----------------------------------


@pytest.mark.parametrize("name, section_contains", [
    ("성과보수, 배당금", "Fin.Model"),
    ("EIU(KR)", "Macro"),
    ("DCF 장표 #1_MGT", "Exhibits"),
    ("제8호_비용", "Biz Plan"),
    ("4.1BS", "BSPL"),
])
def test_classify_section(name, section_contains):
    assert section_contains in classify(name)["section"]


def test_classify_case_from_suffix():
    assert classify("DCF 장표 #1_MGT")["case"] == "MGT"
    assert classify("DCF 장표 #2_DTT")["case"] == "DTT"


# --- carry anchors: the curated per-fund block table ----------------------------------


def test_six_fund_blocks():
    assert [f["alias"] for f in CARRY_FUNDS] == ["제2호", "옐로씨", "제5호", "제7호", "제7-1호", "제8호"]


def test_carry_block_layout():
    # 제2호 sits in value column E; 제7호/제7-1호 share the combined Biz Plan fund node.
    f2 = CARRY_FUNDS[0]
    assert f2["alias"] == "제2호" and f2["val"] == "E" and f2["node"] == "제2호"
    nodes = {f["alias"]: f["node"] for f in CARRY_FUNDS}
    assert nodes["제7호"] == nodes["제7-1호"] == "7호&7-1호"


# --- carry anchors attach the right values + edges (guards model drift) ----------------


def test_carry_anchors_against_workbook(full_workbook):
    """Build the metric layer; the two funds that clear the hurdle are pinned to their cells.

    Active case is DTT (on the node); MGT is kept on ``value_mgt``. The below-hurdle funds
    carry 0, and every fund's carry must DRIVE the aggregate ``performance_fee``.
    """
    g = nx.DiGraph()
    attach_metrics(g, str(full_workbook))

    c7 = g.nodes["Metric:fund_carry:제7호"]
    assert round(c7["value"]) == 135635          # AU6 (DTT, active)
    assert round(c7["value_mgt"]) == 391912      # AU4 (MGT)
    assert g.nodes["Metric:fund_carry:제8호"]["value"] == 20048

    for slug in ("제2호", "옐로씨", "제5호", "제71호"):   # below-hurdle funds carry 0 (제7-1호 -> 제71호)
        assert g.nodes[f"Metric:fund_carry:{slug}"]["value"] == 0

    # each fund's carry feeds the aggregate performance fee, and is defined in the carry sheet
    drivers = {u for u, v, d in g.in_edges("Metric:performance_fee", data=True)
               if d["type"] == "DRIVES" and u.startswith("Metric:fund_carry:")}
    assert len(drivers) == len(CARRY_FUNDS)
    assert any(d["sheet"] == CARRY_SHEET for _, d in g.nodes(data=True) if d.get("type") == "Cell")
