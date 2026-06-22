#!/usr/bin/env bash
#
# Evaluate the wiki agent against a PREBUILT wiki using the 20-question PDF×Excel
# cross-check set (tier-aware LLM judge). Reuses eval/stella_crosscheck.py — this just
# points it at a target wiki + question set via env and runs eval → judge (no rebuild).
#
# Defaults target the v0.2 wiki (Excel ledgers + the 3 FDD decks, incl. STELLA). The
# question set is the Stella cross-check golden (its Excel is byte-identical to the v0.2
# input, so every excel_anchor still holds); with the STELLA FDD now IN the corpus, the
# T2/T3 cross-check items that were "hard without the PDF" become answerable.
#
# Usage (from anywhere):
#     scripts/run_eval.sh                       # eval + judge the v0.2 wiki -> data/eval_v0.2
#     scripts/run_eval.sh judge                 # re-judge existing answers only
#     EVAL_WIKI=data/wiki scripts/run_eval.sh   # score a different prebuilt wiki
#     EVAL_SOURCE=auto EVAL_OUT_DIR=data/eval/v0.2_auto scripts/run_eval.sh
#                                               # exercise the SUPERVISOR (route + compose/
#                                               #   passthrough), not just the wiki backend —
#                                               #   write elsewhere so you can A/B vs the wiki run
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root, regardless of caller's cwd
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

export EVAL_WIKI="${EVAL_WIKI:-data/v0.2/wiki}"
export EVAL_OUT_DIR="${EVAL_OUT_DIR:-data/eval/v0.2_crosscheck}"
export EVAL_QUESTIONS="${EVAL_QUESTIONS:-test_data/v0.1/rag_test_dataset/stella_case/ground_truth/cross_check_questions.jsonl}"
export EVAL_SOURCE="${EVAL_SOURCE:-wiki}"   # wiki = backend directly · auto = supervisor graph
CMDS="${*:-eval judge}"

echo "==> eval target wiki: $EVAL_WIKI"
if [ ! -f "$EVAL_WIKI/index.json" ]; then
  echo "    !! $EVAL_WIKI/index.json missing — build the wiki first, e.g.:"
  echo "       MNA_WIKI_WORKBOOK=data/v0.2/raw/input.xlsx MNA_WIKI_DATA=data/v0.2 \\"
  echo "       MNA_WIKI_PDF_DIR=test_data/v0.2 scripts/run_pipeline.sh"
  exit 1
fi
echo "    $(ls "$EVAL_WIKI"/pages/*.md 2>/dev/null | wc -l | tr -d ' ') pages"
if [ ! -f "$EVAL_QUESTIONS" ]; then
  echo "    !! question set not found: $EVAL_QUESTIONS"
  exit 1
fi
echo "==> questions: $EVAL_QUESTIONS ($(grep -c . "$EVAL_QUESTIONS" | tr -d ' ') Q)"
echo "==> output:    $EVAL_OUT_DIR  (answers.json · scores.json · report.md)"
echo "==> backend:   $EVAL_SOURCE  (wiki = direct · auto = supervisor route+compose)"

echo "==> checking vLLM endpoint (123.37.5.219:8001) ..."
if curl -sf --max-time 8 123.37.5.219:8001/v1/models >/dev/null; then
  echo "    vLLM is up"
else
  echo "    !! vLLM not reachable — the agent run and the judge both need it."
  exit 1
fi

echo "==> running: stella_crosscheck $CMDS"
exec "$PY" -m eval.stella_crosscheck $CMDS
