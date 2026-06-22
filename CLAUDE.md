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

**`data/` is versioned** — each corpus version is self-contained under `data/<version>/`;
gitignored (regenerable) **except** the per-version curation (`decks.yaml`/`routes.yaml`), which is
committed via a `.gitignore` exception. Code resolves paths through `config.py` (env > config.yaml >
default), so a version is built/served/evaluated by pointing the env at its dir, not editing code.

```
data/                       # versioned build artifacts + corpora (gitignored; curation yamls committed)
  v0.1/ { raw md parsed wiki derived  + decks.yaml routes.yaml }   # canonical 63-sheet Centroid model — DEFAULT
  v0.2/ { raw md parsed wiki  + decks.yaml routes.yaml }           # multi-deck test: Excel ledgers + CAESAR/LIFE/STELLA FDD
  eval/ { stella_crosscheck/, v0.2/{final,multi} }   # eval outputs
  graph/ { stella_graph.json, stella_semantic.json } # graph-paradigm artifacts
  logs/
src/stella_kb/
  __init__.py             # ROOT/DATA_DIR + WORKBOOK/FULL_WORKBOOK (canonical = data/v0.1/raw/)
  config.py               # central config (env > config.yaml > default); wiki/agent/eval PATH accessors
  llm.py                  # OpenAI-compatible client (local vLLM); whitelist-guarded term->Metric
  prompts/                # build-pipeline prompts (pdf_page_system, pdf_doc_system, ...)
  graph/                  # property-graph KB paradigm (extract / semantic / metrics / lift / query / viz)
  wiki/                   # vectorless wiki paradigm (dump_md -> parse_llm -> compile -> index -> pdf_pages -> lint)
    pdf_pages.py          # PDF ingest: vision describe -> structured figures + RAW grids/diagrams on page;
                          #   per-deck two-layer "document" node (description + ToC); keeps no-figure pages
    lint.py               # maintenance pass (deterministic, offline): broken [[link]] / orphan alias / missing
                          #   + orphan page / stale route; --fix prunes index.json drift; --contradictions opt-in
    qa.py                 # query-compounding: persist a grounded agent answer to <wiki>/qa/<page>.jsonl
                          #   (source of truth) + render it into the page's "## Q&A (compounded)" section;
                          #   compile re-renders it each rebuild so it survives. Gate: answer must cite ≥1 cell
  parsers/pdf/            # vision PDF parser (describe/vision/tables/router); emits tables + [그래프]/[다이어그램]
apps/agent/               # query agent (separate from the build pipeline)
  core.py                 # public API / facade: run / ask / answer(router) / stream_run — takes a dataset `store`;
                          #   dispatches to the agents/ backends. run/answer(save=True) compounds onto the page
  datasets.py             # dataset (wiki VERSION) registry + cached WikiStore  (id -> wiki dir)
  agents/                 # the agent backends (core dispatches here)
    supervisor.py         #   supervisor StateGraph: routes/merges wiki+dart worker nodes; streaming fast-path
    dart.py               #   DART tool-calling backend (public-company questions)
    wiki/                 #   wiki LangGraph: state/nodes/build  (wiki_dir threaded through state, not a global)
  retrieval/tools.py      # deterministic wiki access (lookup/open_page/query_ledger/trace_links); per-request wiki_dir
  api/                    # FastAPI: /ask (GET) · /ask/stream (GET SSE) · /datasets · /health  + schema/ (pydantic)
frontend/                 # React+Vite chat UI; components/DatasetPicker selects wiki version; proxies API -> :5001
eval/
  stella_crosscheck.py    # v0.1 tier-based PDF×Excel cross-check (20 Q)
  qa_eval.py              # v0.2 vision-QA, rubric-based judge (54 Q over the FDD decks)
  ragas_eval.py
config.yaml               # central config incl. `agent.datasets` version registry
scripts/                  # run_pipeline.sh · run_server.sh · run_eval.sh · run_qa_eval.sh · serve_*.sh
docs/workbook_analysis.md # per-sheet M&A analysis of all 63 sheets (+ sheet-name taxonomy)
.venv/                    # Python 3.11 venv
```

```bash
source .venv/bin/activate                 # or call .venv/bin/python directly
pip install -r requirements.txt           # one-time
python -m src.stella_kb.graph.extract     # parse formulas -> ~13.7k cells, ~74k edges
python -m src.stella_kb.graph.semantic    # full semantic graph -> data/graph/stella_graph.json
python -m src.stella_kb.graph.query       # ask questions: resolve -> traverse -> cited answer
```

```bash
# Build a corpus version into its own tree (default writes data/v0.1). For a new version,
# point the env at its dir — no code edits (run_pipeline.sh inherits the exported env):
MNA_WIKI_WORKBOOK=<x.xlsx> MNA_WIKI_DATA=data/v0.2 MNA_WIKI_PDF_DIR=test_data/v0.2 scripts/run_pipeline.sh
# then register it in config.yaml `agent.datasets` so the API/UI can select it.
# Rebuilds are INCREMENTAL + deterministic: the parse/compile/PDF LLM calls are content-addressed
# on disk (.cache/wiki_parse, .cache/wiki_prose, .cache/pdf_structure via llm.cached_chat), so an
# unchanged sheet/deck is a cache hit (no LLM call, identical output) and only edited sources
# re-roll. Stage [6/6] lints the built wiki (broken links / orphans). To force a fresh rebuild of
# a stage, clear its .cache/ dir first.

# web UI (two processes): FastAPI backend + Vite frontend
scripts/run_server.sh                     # backend on :5001 (HTML fallback at /ui); /datasets lists versions
cd frontend && npm install && npm run dev # React app on :5173 (DatasetPicker), proxies the API
#   query a specific version:  GET /ask?question=...&dataset=v0.2   (POST removed; both endpoints are GET+Query)

# evaluate a built wiki against a ground-truth set (rubric/tier judge):
scripts/run_qa_eval.sh                    # v0.2 vision-QA (54 Q)  -> data/eval/v0.2
scripts/run_eval.sh                       # v0.1 cross-check (20 Q) against a chosen wiki
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

## Datasets & versioning

A **dataset** = one built wiki for a corpus version, self-contained under `data/<version>/wiki`.
The agent serves any registered dataset; selection is **per request**, concurrency-safe.

- **Registry** — `config.yaml` `agent.datasets` maps a safe id → wiki dir (`v0.1: data/v0.1/wiki`,
  `v0.2: data/v0.2/wiki`); `apps/agent/datasets.py` resolves + caches a `WikiStore` (index +
  INDEX.md, keyed by mtime). `default` falls back to `agent_wiki_dir()` (= `data/v0.1/wiki`).
- **API** — `GET /ask?question=…&dataset=v0.2` and `GET /ask/stream?…&dataset=…` (both endpoints
  are GET with `Query()` params — `/ask` POST was removed; SSE is `EventSource`-driven). `GET
  /datasets` lists registered + built ids. Unknown id → 422, registered-but-unbuilt → 503. The
  React `DatasetPicker` populates from `/datasets`.
- **Concurrency-safe by construction** — the chosen `wiki_dir` is threaded through
  `AgentState → Send payload → solve_node → open_page/query_ledger` (a per-request arg), never a
  process global, so two requests can target different versions at once.
- **Add a version**: build into `data/<v>/` (`MNA_WIKI_DATA=data/<v> … run_pipeline.sh`), then add
  one line to `config.yaml` `agent.datasets`. No code changes.

The two corpora today: **v0.1** = the canonical 63-sheet Centroid model (Excel + the Stella FDD
ExecSummary). **v0.2** = a multi-deck vision test set — the same Centroid ledgers plus three FDD
decks (CAESAR/LIFE/STELLA), ingested as namespaced `FDD<n> — [DECK] …` pages with a per-deck
two-layer index (deck description + ToC). PDF ingest (`pdf_pages.py`) now also carries the **raw
vision grids** (matrices/dense tables) and **`[다이어그램]` edge-lists** (box→box, %/amount,
legend) onto each page, and keeps a page even when the structurer found no figures.

### Evaluation

Two ground-truth sets under `test_data/`, judged by the shared vLLM, written to `data/eval/`:
- **`eval/stella_crosscheck.py`** (v0.1) — 20 tier-1/2/3 PDF×Excel cross-check questions.
- **`eval/qa_eval.py`** (v0.2) — 54 **vision-only** questions over the FDD decks
  (`test_data/v0.2/ground_truth/qa.jsonl`), scored **rubric-based** (1.0/0.5/0.0) with breakdowns
  by doc / capability (C1–C5) / visual_type. Run via `scripts/run_qa_eval.sh`.

⚠️ **The eval is noisy.** The shared vLLM is non-deterministic even at temperature 0 (continuous
batching), and a wiki rebuild re-runs `structure_section` per page — so single-run score deltas
under ~±0.1 are not signal. Compare **means over several runs** (the multi-run dir), and hold the
built pages fixed when A/B-testing an agent-only change.

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
  391,912) and 제8호 (20,048) clear the hurdle; the rest are 0. **Dual-case** is wired for the
  DCF-summary scalars (`DUAL_CASE_MGT` in `metrics.py`): equity/enterprise/operating value,
  PV of projection & terminal, NOA, net cash(debt), WACC, PGR, valuation date each carry
  `value`=active **DTT** (live `DCF` cell, kept on the formula DAG) **and** `value_mgt`=**MGT**
  read from the frozen `DCF 장표 #1_MGT` exhibit (identical layout to the DTT exhibit), with
  `cell_mgt` for provenance. So "compare MGT vs DTT equity value" answers from one metric
  (206,131 vs 120,696). **Export to disk** is wired: `python -m src.stella_kb.graph.semantic`
  writes `data/graph/stella_graph.json` (node-link JSON; `export()` also does GraphML). Metric layer
  ≈ **102 metrics**.
- **Query layer (v2, multi-hop) built**: `query.py` does resolve → traverse → synthesize.
  `resolve_all()` maps a question to the **set** of focal Metric ids via `llm.resolve_metrics`
  (whitelist-guarded, order-preserving, capped; falls back to single `resolve()`), so a
  comparison fans out to several metrics ("관리보수와 성과보수 비교" → both series). Each metric's
  evidence is gathered deterministically and is itself multi-hop (`drivers` walks the
  DRIVES/ASSUMPTION_OF chain to depth 6); `series`/`source_cells`/`evidence` carry source
  cells; the LLM only writes one joint prose answer over all blocks and must cite cells.
  Answers KO and EN. Loads `data/graph/stella_graph.json`.
- **Not yet built**: dual-case for the **per-year series** (the DCF cashflow rows EBIT/EBITDA/
  FCFF and the revenue series still read only the active DTT case — only the DCF-summary
  *scalars* and per-fund carry carry both cases so far; the two exhibits' projection windows
  differ — MGT 5.5yrs from 2024 2H, DTT 5yrs from 2025 — so a per-year dual axis needs care).
  `classify_sheets` (and
  the `metrics.py` anchors) are hand-curated and brittle to renames — an LLM labelling pass
  (the OpenKB approach, seeded by the sheet-name taxonomy in `docs/workbook_analysis.md`)
  can extend coverage without touching graph construction. `metrics.py` values come from
  openpyxl's cached results, so the **cached-value caveat applies** — recalc for fresh
  numbers. The `data/graph/stella_graph.json` export is a regenerable build artifact (don't
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

## Local LLM endpoint

`src/stella_kb/llm.py` is a stdlib-only OpenAI-compatible client. Defaults point at our
**own gemma-4 vLLM on `:8001`** (override with env `STELLA_LLM_URL` / `STELLA_LLM_MODEL`):

- URL `http://123.37.5.219:8001/v1`
- Model `gemma-4-31B-it` (Gemma instruct, TP=2, 262k ctx) — launched by
  `scripts/serve_gemma.sh` with `--enable-auto-tool-choice --tool-call-parser gemma4`.
- **One endpoint for everything**: the wiki build/agent (this client) *and* the DART
  tool-calling agent both use `123.37.5.219:8001`. (The old guest vLLM on `:33333`, served
  by another user, is no longer used — `:8001` is ours, so tool-calling works and uptime is
  ours.) Sanity-check: `curl -s 123.37.5.219:8001/v1/models`.
- The agent fans out independent sub-questions (LangGraph `Send`) and per-page retrieval
  concurrently; a semaphore caps in-flight requests at `STELLA_FANOUT` (default 4). vLLM
  continuous-batches what lands at once.

Use the LLM only for *words → nodes* (`resolve_metric`, whitelist-guarded against
`METRIC_IDS`) and final NL synthesis — never to fetch evidence (that stays graph traversal).

## Git note

This directory is untracked in the surrounding `/data/hjpark10` git repo (git root is
the parent). Keep the binary `.xlsx` under `data/`; diffs of it aren't meaningful (the
`_251103_`/`_vShared` filename suffixes are the version markers). Commit the `src/` code,
not `.venv/` or `data/`.
