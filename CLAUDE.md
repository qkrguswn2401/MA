# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project goal

**Estimate the value of Centroid for a merger & acquisition (M&A).** (`MA` = M&A.) The
Excel file is the existing valuation model; this project builds a **knowledge base as a
property graph** from it so an agent can answer M&A valuation questions — what drives the
DCF, how a fee/AUM assumption flows to enterprise value, where each number comes from —
and stress-test the deal case on demand.

Approach is **hybrid** (decided, not yet implemented):
1. **Extract** structure from the workbook — most importantly the **formula dependency
   graph**, which is a native knowledge graph already present in the file (each formula
   cell points at its precedent cells).
2. **Lift** raw cells into semantic nodes/edges (entities, funds, financial metrics,
   periods) — a *property graph*, not a formal RDF/OWL ontology.
3. **Query** on demand: an agent traverses/searches the graph at question time rather
   than pre-compiling answers.

Two external repos are the design references — read them before designing pipeline
stages:
- **OpenKB** (`github.com/VectifyAI/OpenKB`) — explicit-compilation paradigm: LLM turns
  sources into persistent concept/entity pages linked by `[[wikilinks]]`, no vector DB.
  Borrow: the entity/concept node taxonomy and the "compile once, cross-reference"
  philosophy. We diverge by using a real property graph instead of Markdown pages.
- **DCI-Agent-Lite** (`github.com/DCI-Agent/DCI-Agent-Lite`) — direct-corpus-interaction
  paradigm: agent searches raw data with `rg`/`grep`, no pre-built index, "on-demand
  graph traversal." Borrow: the query-time agent loop (search → inspect →
  cross-verify → synthesize) and context-compression levels for long runs.

### Reference comparison & what we adopt (verified Jun 2026)

**Neither is a usable Python library** — both are CLI-only (`openkb`, `dci-agent-lite`;
their `__init__`/entry points expose no stable API). So we do **not** depend on either as
a package — we reimplement the *patterns* in `src/stella_kb`. (If we ever want their
behaviour wholesale, the integration path is subprocessing the CLI and reading its output
files, not importing.)

| | OpenKB | DCI-Agent-Lite | This project |
|---|---|---|---|
| Index | LLM-compiled Markdown wiki + PageIndex | none (`rg`/`find`/`sed`) | networkx **property graph** |
| Retrieval | concept reads + tree reasoning, vectorless | agent greps raw `.txt` corpus | graph traversal over the formula DAG |
| Build time | up front (`add`) | zero | up front (`extract.py`/`graph.py`) |
| Stack | LiteLLM, OpenAI Agents SDK, Click, watchdog | Pi (Node) agent, ripgrep, `uv` | openpyxl, networkx |

Patterns we adopt:
- **From OpenKB — whitelist-guarded linking.** Every compile prompt is handed the closed
  set of valid `[[wikilinks]]` targets and forbidden from inventing others (code-side
  backlinks + `lint --fix` clean strays). When we add the LLM pass for cell→`Metric`
  labelling, feed it the **closed set of existing node ids** so it can only attach to
  nodes that exist — no hallucinated edges. Also borrow its hard concept-vs-entity split
  and cross-document salience count for ranking nodes.
- **From DCI — query-time loop + tiered context management.** The eventual query/agent
  layer should search→inspect→cross-verify→synthesize over the graph, and use staged
  context compression (truncate → compact → summarize) once workbook+graph context grows.

## Layout & commands

```
data/                     # the workbook lives here (the "raw"/corpus input)
src/stella_kb/
  __init__.py             # WORKBOOK path constant (resolves data/ regardless of cwd)
  llm.py                  # OpenAI-compatible client (local vLLM); whitelist-guarded term->Metric  (shared)
  prompts/                # prompt templates, one .txt per use  (shared)
  graph/                  # property-graph KB paradigm
    extract.py            # workbook -> cell-level formula dependency DAG (DEPENDS_ON edges)
    semantic.py           # cell DAG -> semantic property graph (Section/Sheet/Fund/Entity)  (was graph.py)
    metrics.py            # curated cell->Metric anchors (Metric/Period + DRIVES/HAS_VALUE/...)
    query.py              # query layer: resolve -> graph traversal -> cited NL answer
  wiki/                   # vectorless wiki KB paradigm  (dump_md -> parse_llm -> compile -> index)
    dump_md.py            # workbook sheet -> Markdown grid (pipeline stage 1; data/md/)
    dump_sheet.py         # dump any sheet's cells (value + formula) for analysis
    parse_llm.py          # LLM parse pass: grid -> grounded structural schema (data/parsed/)
    compile.py            # compile wiki pages from parsed schema (data/wiki/pages/)  (was wiki.py)
    index.py              # build INDEX.md + index.json routing table
apps/agent/               # query agent (separate from the build pipeline above)
  core.py                 # public API: run / ask / stream_run
  io/tools.py             # deterministic wiki access: load_index, lookup, open_page, trace_links (provenance DAG hop)
  graph/                  # LangGraph: state.py (AgentState), nodes.py, build.py (build_app)
  api/                    # FastAPI: server.py (/ask, /ask/stream SSE) + schema/ (pydantic)
  prompts/                # agent prompt(s), Korean-steered
frontend/                 # web UI for the agent (NOT under apps/; node toolchain, gitignored build)
  src/                    # React + TS chat app (Vite): App.tsx, api.ts (SSE client), components/
  vite.config.ts          # dev server :5173, proxies /ask /ask/stream /health -> backend :8000
  web/index.html          # zero-build single-file HTML fallback; FastAPI serves it at / and /ui
scripts/                  # shell launchers only (.sh): run_pipeline.sh, run_server.sh
docs/workbook_analysis.md # per-sheet M&A analysis of all 63 sheets (+ sheet-name taxonomy)
requirements.txt          # openpyxl, pandas, networkx, langgraph, fastapi, uvicorn
.venv/                    # Python 3.11 venv
```

```bash
source .venv/bin/activate                 # or call .venv/bin/python directly
pip install -r requirements.txt           # one-time
python -m src.stella_kb.graph.extract     # parse formulas -> ~13.7k cells, ~74k edges
python -m src.stella_kb.graph.metrics     # cell->Metric layer alone -> 102 metrics, 14 periods
python -m src.stella_kb.graph.semantic    # full semantic graph -> data/stella_graph.json
python -m src.stella_kb.graph.query       # ask questions: resolve -> traverse -> cited answer
```

```bash
# web UI (two processes): FastAPI backend + Vite frontend
scripts/run_server.sh                     # backend on :8000 (also serves the HTML fallback at /ui)
cd frontend && npm install && npm run dev # React app on :5173, proxies the API  (npm run build for prod)
```

Tests live in `tests/` (`pytest` from the repo root). The default run is **deterministic +
offline** — `trace_links`/`lookup`/`parse_action`/reducer wiring/carry anchors, etc.; live-LLM
end-to-end smoke tests are marked `@pytest.mark.llm` and **skipped unless `--run-llm`** (they
hit the guest vLLM — slow, non-deterministic). Fixtures skip cleanly when build artifacts
(`index.json`, the full workbook) are absent, so a fresh checkout still runs green.

```bash
pytest                 # deterministic suite (~2s, no network)
pytest --run-llm       # also run the live-vLLM end-to-end smoke tests
```

Both `__main__` entry points also have a smoke-print; run them from the repo root (`MA/`) so
`src.` resolves. The reference repos use `uv` — translate their `uv run`/`uv add` to
`python`/`pip install`.

## Data source: the workbook

A private-equity / asset-manager **valuation model** for **Centroid Investment Partners
(센트로이드인베스트먼트파트너스)** and its GP entity **Centroid Management
(센트로이드매니지먼트)**. **63 sheets**, live formulas, mixed Korean/English labels.

Sheets are grouped by **divider tabs whose names end in `>>`**. The four layers, in
dependency order (data flows left→right; an output sheet is rarely the right place to
read a source value):

- ` Biz Plan>>` and `BSPL>>` — **inputs/actuals** (upstream).
  - `BSPL` sub-divides by entity: `>>4.1…` = Centroid Investment Partners
    (`BS`, `PL`, `PL_FY24(A)`), `>>4.2…` = Centroid Management.
  - `Biz Plan` holds **per-fund** detail: one group per fund — `차이나1호` (China Fund 1),
    `제2호`/`제3호`/`제5호`/`제8호`, `옐로씨` (Yellow Sea), `7호&7-1호` — each split into
    `_비용` (costs), `_거래내역` (transactions), `_관리보수` (mgmt fee). `IRR` aggregates.
- `Fin.Model>>` — the **valuation engine**.
  - `AUM Projection` → `관리수수료`/`관리보수` (management fees) and `성과보수, 배당금`
    (performance fees / carry / dividends) are the **revenue drivers**.
  - `Operating Revenue`, `Operating Expense`, `임직원 수`/`인력` (headcount),
    `CapEx & DA`, `NWC`, `Net debt, NOA`, `Tax` build the cash flow.
  - `DCF` is the valuation output (`DCF 장표 #1_MGT` = management case,
    `DCF 장표 #2_DTT` = Deloitte case). `EIU(KR)`/`EIU(US)` hold macro assumptions.
- `PPT >>` — **downstream exhibits** (Football Chart, Bridge, `… 장표 #N`). Numbers come
  from the model layer; never the source of truth.

## Target property-graph model

Derived from the layers above — use as the extraction target schema:

- **Node types**: `Entity` (the two Centroid companies) · `Fund` (the Biz Plan funds) ·
  `Metric`/`LineItem` (AUM, management fee, performance fee, OpEx, CapEx, NWC, tax, DCF
  value, headcount) · `Assumption` (EIU macro, discount rate) · `Period` (fiscal year /
  projection year) · `Cell` (`Sheet!Ref`, the raw grain) · `Sheet`.
- **Edge types**: `DEPENDS_ON` (formula precedent → cell that uses it — the native edge) ·
  `BELONGS_TO` (Fund → Entity) · `DRIVES` (AUM → fees) · `HAS_VALUE` (Metric → Period) ·
  `DEFINED_IN` (Metric → Sheet/Cell) · `ASSUMPTION_OF` (Assumption → Metric).

The `DEPENDS_ON` edges are extracted, not authored: parse each formula string for its
precedent references and build a cell-level DAG, then collapse cells into the semantic
nodes above. This DAG is the backbone the agent traverses.

## Reading the workbook — already implemented, watch these caveats

`extract.py` does the formula reading (two passes: `data_only=False` for formula strings
→ edges, `data_only=True` for cached values → node attrs) and `parse_precedents()`
handles the tricky parts. Reuse it rather than re-opening the workbook. The caveats it
encodes — keep them in mind for any new extraction:

- **Cached values are `None`** for cells Excel never recalculated. openpyxl does not
  recalculate; for fresh node values, recalc in Excel/LibreOffice first.
- **Cross-sheet refs** (`='AUM Projection'!B12`), **ranges** (`A1:C9`, expanded to
  cells), `$` absolute markers, and **Korean sheet names** all appear in formulas and
  are parsed by `parse_precedents`.
- Functions/constants (`SUM(...)`, literals) are not cells — only `Sheet!REF` tokens
  become edges.

## What is built vs. still open

- **Built**: `extract.py` (cell DAG), `graph.py` (rule-based semantic lift), and
  `metrics.py` (cell→`Metric` lift). The schema is largely realised — `DEPENDS_ON`,
  `PART_OF`, `BELONGS_TO`, `DEFINED_IN`, `HAS_VALUE`, `DRIVES`, `ASSUMPTION_OF` all exist;
  Section/Sheet/Fund/Entity **and now Metric/Period** nodes exist. `metrics.py` is a
  **curated anchor table** (`METRICS`) — 36 metrics keyed to verified cells, with the
  per-sheet `fiscal_year_axis` resolver handling each sheet's column offset, and a closed
  `METRIC_IDS` whitelist guarding the cross-metric edges (OpenKB pattern). The DCF
  valuation chain is fully traversable: `aum_cumulative → … → fcff → … → enterprise_value
  → equity_value`, with `wacc`/`pgr`/`hurdle_rate`/`carry_rate` as `ASSUMPTION_OF` edges.
  Per-fund fee anchors (`관리수수료` rows 8-19) add `fund_fee_rate`/`fund_committed_capital`/
  `fund_mgmt_fee` per fund (12 funds), each `BELONGS_TO` its `Fund:` node and `DRIVES` the
  aggregate `management_fee`. **Per-fund carry anchors** (`성과보수, 배당금`, `CARRY_FUNDS` in
  `metrics.py`) add `fund_carry`/`fund_distribution` + the Exit assumptions
  (`fund_exit_ebitda`/`fund_exit_multiple`/`fund_hurdle`) per fund (6 funds with a carry
  block: 제2호·옐로씨·제5호·제7호·제7-1호·제8호); carry carries the active **DTT** on the node
  and **MGT** on `value_mgt`, `DRIVES` the aggregate `performance_fee`, and `BELONGS_TO` its
  `Fund:` node (제7호/제7-1호 → the combined `7호&7-1호`). Only 제7호 (DTT 135,635 / MGT
  391,912) and 제8호 (20,048) clear the hurdle; the rest are 0. **Export to disk** is wired:
  `python -m src.stella_kb.graph.semantic` writes `data/stella_graph.json` (node-link JSON;
  `export()` also does GraphML). Metric layer ≈ **102 metrics**.
- **Query layer (v2, multi-hop) built**: `query.py` does resolve → traverse → synthesize.
  `resolve_all()` maps a question to the **set** of focal Metric ids via `llm.resolve_metrics`
  (whitelist-guarded, order-preserving, capped; falls back to single `resolve()`), so a
  comparison fans out to several metrics ("관리보수와 성과보수 비교" → both series). Each metric's
  evidence is gathered deterministically and is itself multi-hop (`drivers` walks the
  DRIVES/ASSUMPTION_OF chain to depth 6); `series`/`source_cells`/`evidence` carry source
  cells; the LLM only writes one joint prose answer over all blocks and must cite cells.
  Answers KO and EN. Loads `data/stella_graph.json`.
- **Not yet built**: the `_MGT`/`_DTT` case as parallel metric values across the *whole*
  model (per-fund carry already carries both via `value`/`value_mgt`, but the DCF/revenue
  series still read only the active DTT case — so "compare MGT vs DTT equity value" can't yet
  be answered from the graph even though the fan-out resolves it). `classify_sheets` (and
  the `metrics.py` anchors) are hand-curated and brittle to renames — an LLM labelling pass
  (the OpenKB approach, seeded by the sheet-name taxonomy in `docs/workbook_analysis.md`)
  can extend coverage without touching graph construction. `metrics.py` values come from
  openpyxl's cached results, so the **cached-value caveat applies** — recalc for fresh
  numbers. The `data/stella_graph.json` export is a regenerable build artifact (don't
  commit it; commit `src/`).

## Retrieval strategy: vectorless by default

Default is **vectorless** — like both reference repos, but for stronger reasons here: the
data is structured (the formula DAG gives exact precedent→dependent edges), numbers and
cell refs embed poorly, and M&A valuation needs **deterministic, complete, auditable
provenance** ("EV ← `DCF!K59` ← `AUM Projection!B12`") that top-k vector recall can't
guarantee. The corpus is tiny (~14k cells), so a vector DB is pure overhead. Primary
retrieval = graph traversal over the dependency graph; answers cite cell paths.

The one real gap is **vocabulary mismatch** — mixed KO/EN labels (`관리수수료` ↔
"management fee" ↔ "mgmt fee", `성과보수` ↔ carry). Pure lexical/structural lookup misses
synonyms. Close it with the **cheapest auditable thing first**:
1. a curated/LLM **alias dictionary** over the few-hundred distinct labels (closed
   vocabulary → fits the OpenKB whitelist pattern; deterministic at query time);
2. only if insufficient, **embeddings over the label set alone** — used to resolve a
   query term to a node, **never to fetch evidence**.

Rule of thumb: vectors (if used at all) map *words → nodes*; the graph maps *nodes →
answers*. Keep evidence retrieval on the graph.

## Local LLM endpoint (shared)

`src/stella_kb/llm.py` is a stdlib-only OpenAI-compatible client. Defaults point at a
**shared local vLLM server** (override with env `STELLA_LLM_URL` / `STELLA_LLM_MODEL`):

- URL `http://localhost:33333/v1` (the server runs on this host — use localhost)
- Model `gemma-4-31B-it` (Gemma instruct, TP=2 on GPUs 6–7, 262k ctx)
- Served by another user (`donghan906`'s `Coinv`) — **guest resource**: keep load light, don't
  assume uptime. Sanity-check: `curl -s localhost:33333/v1/models`.
- The agent fans out independent sub-questions (LangGraph `Send`) and per-page retrieval
  concurrently; a semaphore caps in-flight requests to this server at `STELLA_FANOUT`
  (default 4) so the guest vLLM isn't overloaded. vLLM continuous-batches what lands at once.

Use the LLM only for *words → nodes* (`resolve_metric`, whitelist-guarded against
`METRIC_IDS`) and final NL synthesis — never to fetch evidence (that stays graph traversal).

## Git note

This directory is untracked in the surrounding `/data/hjpark10` git repo (git root is
the parent). Keep the binary `.xlsx` under `data/`; diffs of it aren't meaningful (the
`_251103_`/`_vShared` filename suffixes are the version markers). Commit the `src/` code,
not `.venv/` or `data/`.
