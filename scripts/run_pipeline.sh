#!/usr/bin/env bash
#
# Rebuild the Project Stella vectorless KB end-to-end, from the workbook to the index.
#
#   data/raw/*_raw.xlsx
#     → data/md/*.md            (1) grid dumps          [mechanical]
#     → data/parsed/*.json      (2) LLM parse pass      [needs vLLM; cached -> incremental]
#     → data/wiki/pages/*.md    (3) wiki compile        [needs vLLM; prose cached -> incremental]
#     → data/wiki/INDEX.md      (4) index / ToC         [mechanical]
#       data/wiki/index.json
#   data/raw/*.pdf
#     → data/wiki/pages/FDD*.md (5) PDF ingest + merge  [slow, needs vLLM; skipped if no PDF]
#     → (report)                (6) lint built wiki      [mechanical; broken links / orphans]
#
# Stages 2/3/5 cache their LLM calls (.cache/wiki_parse, .cache/wiki_prose, .cache/pdf_structure),
# keyed by content — so an unchanged source is a cache hit (free, deterministic) and only edited
# sheets/decks re-roll. Force a full fresh rebuild by clearing the relevant .cache/ dir first.
#
# Usage (from anywhere):
#     ./run_pipeline.sh            full rebuild
#     ./run_pipeline.sh --no-llm   skip the three LLM stages (reuse existing parsed JSON;
#                                  scaffold-only pages; no PDF re-ingest), then rebuild index
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root (script lives in scripts/)
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

NO_LLM=0
[ "${1:-}" = "--no-llm" ] && NO_LLM=1

if [ "$NO_LLM" -eq 0 ]; then
  echo "==> [0] checking vLLM endpoint (123.37.5.219:8001) ..."
  if curl -sf --max-time 8 123.37.5.219:8001/v1/models >/dev/null; then
    echo "    vLLM is up"
  else
    echo "    !! vLLM not reachable — the parse/wiki stages need it."
    echo "       Re-run with --no-llm to rebuild structure + index only."
    exit 1
  fi
fi

echo "==> [1/6] dump sheets to markdown  -> data/md/"
"$PY" -m src.stella_kb.wiki.dump_md --all

if [ "$NO_LLM" -eq 0 ]; then
  echo "==> [2/6] LLM parse pass  -> data/parsed/   [slow]"
  "$PY" -m src.stella_kb.wiki.parse_llm --all

  echo "==> [3/6] compile wiki pages  -> data/wiki/pages/   [slow]"
  "$PY" -m src.stella_kb.wiki.compile --all
else
  echo "==> [2/6] LLM parse pass  -> skipped (--no-llm; reusing data/parsed/)"
  echo "==> [3/6] compile wiki pages  -> data/wiki/pages/   (scaffold only)"
  "$PY" -m src.stella_kb.wiki.compile --all --no-llm
fi

echo "==> [4/6] build index / ToC  -> data/wiki/INDEX.md, index.json"
"$PY" -m src.stella_kb.wiki.index

if [ "$NO_LLM" -eq 0 ]; then
  echo "==> [5/6] PDF ingest + merge into index  -> data/wiki/pages/FDD*.md   [slow]"
  "$PY" -m src.stella_kb.wiki.pdf_pages          # self-skips if no data/raw/*.pdf
else
  echo "==> [5/6] PDF ingest  -> skipped (--no-llm; existing FDD pages left as-is)"
fi

WIKI_DIR="${MNA_WIKI_DATA:-data/v0.1}/wiki"
echo "==> [6/6] lint the built wiki  ($WIKI_DIR)"
"$PY" -m src.stella_kb.wiki.lint "$WIKI_DIR" || \
  echo "    !! lint found error-severity issues (see above) — build left in place for inspection"

echo "==> done. entry point: $WIKI_DIR/INDEX.md"
