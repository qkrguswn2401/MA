"""Separate tool-calling DART agent — answers public-company questions via the DART MCP server.

Unlike the wiki agent (deterministic retrieval over the Centroid wiki, JSON-per-turn because
the *guest* vLLM has no tool-calling), this is a native tool-calling loop: a tools-capable
model (gemma-4 served WITH ``--tool-call-parser gemma4``) decides which DART tool to call and
with what arguments, via LangChain ``create_agent`` over the MCP tools. It connects to the
already-running DART MCP server over **SSE** (the containerized instance), authenticating with
a bearer token — i.e. it consumes the exact service we share with others.

Config (env; defaults target the local services on this box):
    STELLA_TOOL_LLM_URL    tool-calling LLM base URL   (default http://123.37.5.219:8001/v1)
    STELLA_TOOL_LLM_MODEL  served model name           (default gemma-4-31B-it)
    DART_MCP_URL           DART MCP SSE endpoint        (default http://127.0.0.1:8002/sse)
    DART_MCP_TOKEN         bearer token for that server (REQUIRED — no default in source)
"""

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path

from src.stella_kb import config

try:                                  # load repo-root .env so secrets stay out of source,
    from dotenv import load_dotenv    # mirroring the server side (mcps/dart-mcp/dart.py).
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:                   # dotenv optional — fall back to a real exported env var.
    pass

# NB: the gemma-4 tool-calling container isn't reachable on 127.0.0.1 from this box
# (a docker loopback quirk) — use the host IP that works, matching test_mcps.py.
TOOL_LLM_URL = config.tool_llm_url()
TOOL_LLM_MODEL = config.tool_llm_model()
DART_MCP_URL = config.dart_mcp_url()
DART_MCP_TOKEN = os.environ.get("DART_MCP_TOKEN", "")  # required; no secret baked into source


# gemma-4 leaks its channel control tokens into message content, e.g.
#   "<|channel>thought\n<channel|>오늘 날짜는 ..."
# Strip the opener (with its channel name) and the closer so only prose remains.
_CHANNEL_OPEN = re.compile(r"<\|channel>\w*")


def _clean(text: str) -> str:
    return _CHANNEL_OPEN.sub("", text or "").replace("<channel|>", "").strip()


def _short_args(args) -> str:
    if not isinstance(args, dict):
        return str(args)[:80]
    return ", ".join(f"{k}={v}" for k, v in args.items())[:120]


def _trace_from(messages: list) -> list[dict]:
    """Render the agent's messages as the same trace shape the wiki agent emits
    ({step, agent, action, arg, thought}), so the API/UI can show DART tool calls."""
    trace: list[dict] = []
    step = 0
    for m in messages:
        for tc in (getattr(m, "tool_calls", None) or []):
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
            trace.append({"step": step, "agent": "dart", "action": "call",
                          "arg": f"{name}({_short_args(args)})", "thought": ""})
            step += 1
        if m.__class__.__name__ == "ToolMessage":
            content = m.content if isinstance(m.content, str) else str(m.content)
            trace.append({"step": step, "agent": "dart", "action": "result",
                          "arg": f"{getattr(m, 'name', 'tool')}: {content[:120]}", "thought": ""})
            step += 1
    return trace


async def _arun(question: str) -> dict:
    # imported lazily so importing this module (e.g. for config) doesn't require the
    # langchain/MCP stack unless the DART agent is actually used.
    from langchain.agents import create_agent
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_openai import ChatOpenAI

    if not DART_MCP_TOKEN:
        raise RuntimeError("DART_MCP_TOKEN is unset — export the bearer token for the DART MCP server.")

    client = MultiServerMCPClient({
        "dart": {
            "transport": "sse",
            "url": DART_MCP_URL,
            "headers": {"Authorization": f"Bearer {DART_MCP_TOKEN}"},
        }
    })
    tools = await client.get_tools()
    llm = ChatOpenAI(model=TOOL_LLM_MODEL, base_url=TOOL_LLM_URL, api_key="EMPTY", temperature=0)
    agent = create_agent(model=llm, tools=tools)

    result = await agent.ainvoke({"messages": [("user", question)]})
    messages = result.get("messages", [])
    last = messages[-1].content if messages else ""
    answer = _clean(last if isinstance(last, str) else str(last)) or "(빈 답변)"
    trace = _trace_from(messages)
    steps = sum(1 for e in trace if e["action"] == "call")
    return {"answer": answer, "trace": trace, "steps": steps}


def run_dart(question: str) -> dict:
    """Answer a public-company question with the DART tool-calling agent.

    Returns ``{answer, trace, steps}`` (same shape as ``core.run``). Network/LLM failures
    are caught and reported in the answer rather than raised, so a caller/router degrades
    gracefully when the tool LLM (:8001) or the DART server (:8002) is down."""
    try:
        return asyncio.run(_arun(question))
    except Exception as e:  # noqa: BLE001 — surface dependency failures as an answer
        return {"answer": f"(DART 에이전트 오류: {type(e).__name__}: {e})", "trace": [], "steps": 0}


def ask_dart(question: str) -> str:
    """Convenience wrapper returning just the answer string."""
    return run_dart(question)["answer"]


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "삼성전자 공시 알려줘"
    print(f"tool LLM: {TOOL_LLM_URL} ({TOOL_LLM_MODEL})  |  DART: {DART_MCP_URL}\n")
    out = run_dart(q)
    for e in out["trace"]:
        print(f"  [{e['agent']}] {e['action']}: {e['arg']}")
    print("\n" + out["answer"])
