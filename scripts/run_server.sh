#!/usr/bin/env bash
#
# Launch the Project Stella wiki-agent HTTP API (FastAPI/uvicorn).
#
# Preflight: the wiki must be built (data/wiki/index.json) and the local vLLM must be up
# (the agent calls it on every /ask). Then serves apps.agent.api.server:app.
#
# Usage (from anywhere):
#     agent/run_server.sh                 # serve on 0.0.0.0:8000
#     PORT=9001 agent/run_server.sh       # custom port
#     HOST=127.0.0.1 agent/run_server.sh  # custom bind
#     RELOAD=1 agent/run_server.sh        # dev auto-reload
#
set -euo pipefail

cd "$(dirname "$0")/.."                    # repo root, regardless of caller's cwd
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5001}"

# DART backend needs a bearer token to reach the shared DART MCP server (SSE :8002).
# The token lives in mcps/dart-mcp/.env (gitignored, never in source); load it here so
# auto/dart-routed questions can authenticate. Absent token → wiki still works; dart 503s.
DART_ENV="mcps/dart-mcp/.env"
if [ -z "${DART_MCP_TOKEN:-}" ] && [ -f "$DART_ENV" ]; then
  DART_MCP_TOKEN="$(grep -E '^DART_MCP_TOKEN=' "$DART_ENV" | head -1 | cut -d= -f2-)"
  export DART_MCP_TOKEN
fi
[ -n "${DART_MCP_TOKEN:-}" ] && echo "    DART_MCP_TOKEN loaded (dart backend enabled)" \
                             || echo "    DART_MCP_TOKEN unset (dart backend will 503; wiki unaffected)"

echo "==> checking wiki artifacts ..."
if [ ! -f data/wiki/index.json ]; then
  echo "    !! data/wiki/index.json missing — build the wiki first: ./run_pipeline.sh"
  exit 1
fi
echo "    data/wiki present ($(ls data/wiki/pages/*.md 2>/dev/null | wc -l | tr -d ' ') pages)"

echo "==> checking vLLM endpoint (123.37.5.219:8001) ..."
if curl -sf --max-time 8 123.37.5.219:8001/v1/models >/dev/null; then
  echo "    vLLM is up"
else
  echo "    !! vLLM not reachable — /ask will return 503 until it is up."
fi

RELOAD_FLAG=""
[ "${RELOAD:-0}" = "1" ] && RELOAD_FLAG="--reload"

echo "==> serving apps.agent.api.server:app on http://${HOST}:${PORT}  (docs: /docs)"
exec "$PY" -m uvicorn apps.agent.api.server:app --host "$HOST" --port "$PORT" $RELOAD_FLAG
