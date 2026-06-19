#!/usr/bin/env bash
#
# Drive the handoff-tool supervisor (apps.agent.supervisor) on one question from the CLI.
#
# The supervisor hands a tool-calling gemma-4 (:8001) two tools — consult_centroid_wiki and
# consult_dart — lets it pick (or call both), then composes the final Korean answer. So it
# needs: the wiki built (data/wiki/index.json), the vLLM up, and — for DART questions — the
# DART_MCP_TOKEN to reach the shared DART MCP server (else the dart tool 503s; wiki unaffected).
#
# Usage (from anywhere):
#     scripts/run_supervisor.sh "센트로이드 기업가치는 얼마인가요?"
#     scripts/run_supervisor.sh "삼성전자 2023년 매출액?"
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root, regardless of caller's cwd
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python

# DART backend needs a bearer token to reach the shared DART MCP server (SSE :8002); load it
# from mcps/dart-mcp/.env (gitignored) like run_server.sh. Absent → wiki tool still works.
DART_ENV="mcps/dart-mcp/.env"
if [ -z "${DART_MCP_TOKEN:-}" ] && [ -f "$DART_ENV" ]; then
  DART_MCP_TOKEN="$(grep -E '^DART_MCP_TOKEN=' "$DART_ENV" | head -1 | cut -d= -f2-)"
  export DART_MCP_TOKEN
fi
[ -n "${DART_MCP_TOKEN:-}" ] && echo "    DART_MCP_TOKEN loaded (dart tool enabled)" \
                             || echo "    DART_MCP_TOKEN unset (dart tool will error; wiki unaffected)"

# Resolve the default dataset's wiki dir (config-driven, e.g. data/v0.1/wiki) and check it's built.
if ! "$PY" -c "import sys; from apps.agent import datasets; sys.exit(0 if datasets.get_store(None).exists() else 1)"; then
  echo "    !! default wiki not built — build it first: scripts/run_pipeline.sh"
  exit 1
fi

exec "$PY" -m apps.agent.supervisor "$@"
