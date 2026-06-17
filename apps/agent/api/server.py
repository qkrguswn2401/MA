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
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.stella_kb.llm import BASE_URL, MODEL

from .. import datasets
from ..core import answer, stream_run
from .schema import AskResponse

# the static chat frontend (single-file HTML fallback; React app lives in frontend/)
# server.py = apps/agent/api/server.py -> parents[3] = repo root
WEB_DIR = Path(__file__).resolve().parents[3] / "frontend" / "web"


def _resolve_store(dataset: str | None) -> datasets.WikiStore:
    """Map a requested dataset id to its WikiStore, or raise the right HTTP error.

    Unknown id → 422 (list the registered ids); known but not built → 503. Stores (and their
    loaded indices) are cached, so this is cheap per request and lets concurrent requests
    target different datasets safely (the dir is threaded through the agent, not a global)."""
    try:
        store = datasets.get_store(dataset)
    except KeyError:
        raise HTTPException(
            status_code=422,
            detail=f"unknown dataset {dataset!r}; available: {datasets.available()}")
    if not store.exists():
        raise HTTPException(
            status_code=503,
            detail=f"dataset {store.dataset!r} not built: {store.index_json} missing")
    return store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Validate the default dataset is built; per-request datasets are resolved + cached lazily.
    default = datasets.get_store(None)
    if not default.exists():
        raise RuntimeError(
            f"{default.index_json} missing — build the wiki first (run_pipeline.sh)")
    default.index  # warm the default index cache
    yield


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
    default = datasets.get_store(None)
    pages_dir = default.wiki_dir / "pages"
    n_pages = len(list(pages_dir.glob("*.md"))) if pages_dir.exists() else 0
    llm_ok = _vllm_up()
    return {
        "status": "ok" if (n_pages and llm_ok) else "degraded",
        "default_dataset": default.dataset,
        "wiki_pages": n_pages,
        "datasets": _dataset_status(),
        "llm": {"url": BASE_URL, "model": MODEL, "reachable": llm_ok},
    }


def _dataset_status() -> dict:
    """``{id: built?}`` over every registered dataset — for /health and /datasets."""
    return {ds: datasets.WikiStore(ds).exists() for ds in datasets.available()}


@app.get("/datasets")
def list_datasets() -> dict:
    """Registered wiki datasets (versions) selectable via the ``dataset`` payload param,
    and whether each is built. Pass one of the ``built`` ids as ``dataset`` to /ask."""
    return {"default": datasets.DEFAULT, "datasets": _dataset_status()}


@app.get("/ask", response_model=AskResponse)
def ask_endpoint(
    question: str = Query(..., description="KO/EN question about the Centroid valuation or a public company."),
    max_steps: int = Query(3, ge=1, le=20, description="Per-branch read budget (initial read + retries)."),
    source: Literal["auto", "wiki", "dart"] = Query(
        "auto", description="Backend: auto-route, 'wiki' (Centroid KB), or 'dart' (public co. via DART)."),
    include_trace: bool = Query(True, description="Return the routing trace."),
    dataset: str | None = Query(
        None, description="Wiki dataset/version id (e.g. 'v0.2'); omit for the default. See GET /datasets."),
) -> AskResponse:
    """Answer one question, routing to the wiki (Centroid) or DART (public co.) backend.

    Inputs are query parameters (same shape as ``/ask/stream``). ``source`` selects the backend:
    ``auto`` (route via the LLM), ``wiki``, or ``dart``. Returns the answer, which backend served
    it, the dataset queried, and the routing trace."""
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    # auto/wiki need the guest vLLM (routing + wiki retrieval run on it); dart uses its own
    # endpoints (the tool LLM on :8001 + the DART MCP server) and degrades to an error answer.
    if source != "dart" and not _vllm_up():
        raise HTTPException(status_code=503, detail=f"LLM endpoint {BASE_URL} unreachable")
    # The dart backend has no wiki, so only resolve a dataset for wiki/auto requests.
    store = None if source == "dart" else _resolve_store(dataset)
    result = answer(question, source=source, max_steps=max_steps, store=store)
    return AskResponse(
        question=question,
        answer=result["answer"],
        steps=result["steps"],
        source=result["source"],
        dataset=(store.dataset if store is not None else None),
        trace=result["trace"] if include_trace else None,
    )


@app.get("/ask/stream")
def ask_stream(
    question: str = Query(..., description="KO/EN question about the valuation."),
    max_steps: int = Query(3, ge=1, le=20, description="Per-branch read budget (initial read + retries)."),
    dataset: str | None = Query(None, description="Wiki dataset/version id (e.g. 'v0.2'); "
                                                  "omit for the default. See GET /datasets."),
) -> StreamingResponse:
    """Stream the agent's routing live as Server-Sent Events.

    Inputs are query parameters (the browser drives this with ``EventSource``, which is GET-only
    and can't send a body). Emits one ``step`` event per agent decision (which page it opens and
    why), a final ``answer`` event, then ``done``. Consume with an EventSource (browser) or
    ``curl -N localhost:8000/ask/stream?question=...``.
    """
    if not question.strip():
        raise HTTPException(status_code=422, detail="question must not be empty")
    if not _vllm_up():
        raise HTTPException(status_code=503, detail=f"LLM endpoint {BASE_URL} unreachable")
    store = _resolve_store(dataset)

    def gen():
        try:
            for ev in stream_run(question, max_steps=max_steps, store=store):
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
        "endpoints": ["/health", "/datasets", "/ask (GET)", "/ask/stream (GET, SSE)"],
    }


# serve any other static assets (kept after routes so it never shadows /ask etc.)
app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")
