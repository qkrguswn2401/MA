"""FastAPI server exposing the wiki query agent over HTTP.

Thin wrapper around ``apps.agent.core.run`` — the wiki index is loaded once at startup
and reused across requests; each ``/ask`` runs the LangGraph agent and returns the cited
Korean answer plus the routing trace (which page it opened and why). Endpoints run in
FastAPI's threadpool (the agent's LLM calls are blocking urllib), so the event loop is
never stalled.

Run (from repo root, venv active; needs data/wiki/ and the local vLLM — see llm.py):
    .venv/bin/uvicorn apps.agent.api.server:app --host 0.0.0.0 --port 8000
    # interactive docs at http://localhost:8000/docs

    curl -s localhost:8000/health
    curl -s localhost:8000/ask -H 'Content-Type: application/json' \
         -d '{"question": "기업가치는 얼마인가요?"}' | python -m json.tool
"""

from __future__ import annotations

import json
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.stella_kb.llm import BASE_URL, MODEL

from ..core import run, stream_run
from ..io import INDEX_JSON, PAGES_DIR, load_index
from .schema import AskRequest, AskResponse

# the static chat frontend (single-file HTML fallback; React app lives in frontend/)
# server.py = apps/agent/api/server.py -> parents[3] = repo root
WEB_DIR = Path(__file__).resolve().parents[3] / "frontend" / "web"

# index is loaded once and stashed here for every request to reuse
_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not INDEX_JSON.exists():
        raise RuntimeError(f"{INDEX_JSON} missing — build the wiki first (run_pipeline.sh)")
    _STATE["index"] = load_index()
    yield
    _STATE.clear()


app = FastAPI(
    title="Project Stella — Wiki Query Agent",
    description="Answer Centroid M&A valuation questions by navigating the vectorless wiki.",
    version="1.0.0",
    lifespan=lifespan,
)


def _vllm_up() -> bool:
    try:
        with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 — health probe, any failure means "down"
        return False


@app.get("/health")
def health() -> dict:
    """Liveness + dependency check: wiki artifacts present and the vLLM reachable."""
    n_pages = len(list(PAGES_DIR.glob("*.md"))) if PAGES_DIR.exists() else 0
    llm_ok = _vllm_up()
    return {
        "status": "ok" if (n_pages and llm_ok) else "degraded",
        "wiki_pages": n_pages,
        "index_loaded": "index" in _STATE,
        "llm": {"url": BASE_URL, "model": MODEL, "reachable": llm_ok},
    }


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest) -> AskResponse:
    """Answer one question by navigating the wiki; returns the answer + routing trace."""
    if not req.question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if not _vllm_up():
        raise HTTPException(status_code=503, detail=f"LLM endpoint {BASE_URL} unreachable")
    result = run(req.question, max_steps=req.max_steps, index=_STATE.get("index"))
    return AskResponse(
        question=req.question,
        answer=result["answer"],
        steps=result["steps"],
        trace=result["trace"] if req.include_trace else None,
    )


@app.get("/ask/stream")
def ask_stream(
    question: str = Query(..., description="KO/EN question about the valuation."),
    max_steps: int = Query(3, ge=1, le=20),
) -> StreamingResponse:
    """Stream the agent's routing live as Server-Sent Events.

    Emits one ``step`` event per agent decision (which page it opens and why), a final
    ``answer`` event, then ``done``. Consume with an EventSource (browser) or
    ``curl -N localhost:8000/ask/stream?question=...``.
    """
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if not _vllm_up():
        raise HTTPException(status_code=503, detail=f"LLM endpoint {BASE_URL} unreachable")

    def gen():
        try:
            for ev in stream_run(question, max_steps=max_steps, index=_STATE.get("index")):
                etype = ev.pop("type")
                yield f"event: {etype}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as e:  # noqa: BLE001 — surface failures to the SSE client
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    # sync generator → Starlette iterates it in a threadpool (the agent's calls are blocking)
    return StreamingResponse(
        gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    """Serve the chat frontend (single-file HTML app talking to /ask/stream)."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/info", include_in_schema=False)
def info() -> dict:
    return {
        "service": "stella-wiki-agent",
        "ui": "/",
        "docs": "/docs",
        "endpoints": ["/health", "/ask (POST)", "/ask/stream (GET, SSE)"],
    }


# serve any other static assets (kept after routes so it never shadows /ask etc.)
app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
