"""Thin OpenAI-compatible client for the local vLLM endpoint, plus the one LLM use the
retrieval strategy actually sanctions: **words -> nodes** (resolve a KO/EN query term to a
Metric id), never evidence retrieval — see CLAUDE.md "Retrieval strategy".

Config via env (defaults point at the gemma-4 vLLM on :8001 — the same server the DART
tool agent uses; see scripts/serve_gemma.sh):
    STELLA_LLM_URL    base URL, default http://123.37.5.219:8001/v1
    STELLA_LLM_MODEL  served model name, default gemma-4-31B-it

The server runs on another box (123.37.5.219), so the default points at that host, not
``localhost``. Stdlib only (urllib) — no new dependency. The endpoint is OpenAI-compatible,
so swapping in a hosted API is just two env vars.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from . import config
from .graph.metrics import METRICS, METRIC_IDS
from .prompts import load as load_prompt

BASE_URL = config.llm_url()
MODEL = config.llm_model()


def chat(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
         timeout: float = 60.0) -> str:
    """One chat-completions round trip; returns the assistant text."""
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]


def chat_stream(messages: list[dict], temperature: float = 0.0, max_tokens: int = 512,
                timeout: float = 120.0):
    """Streaming :func:`chat` — yields assistant content deltas as the model emits them.

    Uses the OpenAI-compatible ``stream: true`` SSE protocol: the server sends ``data: {json}``
    lines, each with a ``choices[0].delta.content`` fragment, terminated by ``data: [DONE]``.
    Stdlib only (urllib reads the chunked response line by line). The buffered :func:`chat` stays
    the default; this is for the one place streaming pays off — the final user-facing answer
    (the apps.agent synthesizer), so it appears token by token instead of all at once.
    """
    payload = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw_line in r:
            line = raw_line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[len(b"data:"):].strip()
            if data == b"[DONE]":
                break
            try:
                obj = json.loads(data)
            except (ValueError, json.JSONDecodeError):
                continue  # skip keep-alive / non-JSON lines
            delta = (obj.get("choices") or [{}])[0].get("delta", {}).get("content")
            if delta:
                yield delta


def cached_chat(messages: list[dict], *, cache_dir: str, temperature: float = 0.0,
                max_tokens: int = 512, timeout: float = 60.0) -> str:
    """Disk-cached :func:`chat` — same return, but reproducible across reruns for the same
    (model, messages, params). Use for **build-time** LLM calls (e.g. PDF structuring) so a
    wiki rebuild doesn't re-roll the corpus and perturb downstream eval results. The shared
    vLLM is non-deterministic even at temperature 0 (continuous batching), so this is what
    makes a rebuild idempotent. Do NOT use for the agent/judge — those stay stochastic by
    design (averaged over runs). Only successful responses are cached; a corrupt entry recomputes.

    See also: ``parsers.pdf.vision.get_or_compute`` — a parallel cache for vision (multimodal)
    calls; stores under ``"markdown"`` and takes a ``compute`` callable rather than calling
    ``chat`` directly. Different cache dir and payload shape; kept separate intentionally.
    """
    key = hashlib.sha256(
        json.dumps([MODEL, messages, temperature, max_tokens], ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:32]
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        except Exception:  # noqa: BLE001 — ignore a corrupt cache entry and recompute
            pass
    result = chat(messages, temperature=temperature, max_tokens=max_tokens, timeout=timeout)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"content": result}, ensure_ascii=False), encoding="utf-8")
    return result


def _json_span(raw: str, open_ch: str, close_ch: str):
    """Pull the first JSON value delimited by ``open_ch``..``close_ch`` out of a model reply.

    Tolerates ```json fences and surrounding chatter (the model sometimes wraps or prefaces
    its answer). Returns the parsed object, or ``None`` if nothing parseable is found — the
    one place both resolvers share their messy-output handling.
    """
    s = raw.strip()
    if "```" in s:                       # strip ```json fences if the model adds them
        s = s.split("```")[1].lstrip("json").strip() if s.count("```") >= 2 else s.strip("`")
    start, end = s.find(open_ch), s.rfind(close_ch)
    if start < 0 or end < start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except (ValueError, json.JSONDecodeError):
        return None


def _catalog() -> str:
    """Closed-vocabulary catalog handed to the model: id + EN/KO labels + aliases."""
    lines = []
    for m in METRICS:
        names = [m.label_en] + ([m.label_ko] if m.label_ko else []) + list(m.aliases)
        lines.append(f"- {m.id}: {' | '.join(names)}")
    return "\n".join(lines)


def resolve_metric(term: str, timeout: float = 60.0) -> dict:
    """Map a free-text term (Korean or English) to a Metric id from the closed set.

    Whitelist-guarded (OpenKB pattern): the model is given only the existing ids and must
    return one of them or ``null`` — a returned id outside ``METRIC_IDS`` is rejected to
    ``null`` rather than trusted, so no hallucinated node can leak through.
    """
    system = load_prompt("resolve_metric_system")
    user = f"Catalog:\n{_catalog()}\n\nTerm: {term!r}\nJSON:"
    raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
               max_tokens=80, timeout=timeout)
    obj = _json_span(raw, "{", "}")
    if not isinstance(obj, dict):
        return {"id": None, "confidence": 0.0, "raw": raw}
    if obj.get("id") not in METRIC_IDS:  # guard: reject anything off-whitelist
        obj["id"] = None
    return obj


def resolve_metrics(question: str, max_metrics: int = 4, timeout: float = 60.0) -> list[str]:
    """Map a question to the **set** of Metric ids it asks about (the multi-hop fan-out).

    Comparative / cross-metric questions ("compare the management and performance fees")
    resolve to several ids; single-metric questions to one. Same whitelist guard as
    ``resolve_metric`` — every returned id must be in ``METRIC_IDS`` — applied per element,
    so a hallucinated id is dropped rather than trusted. Order-preserving, deduplicated,
    capped at ``max_metrics``.
    """
    system = load_prompt("resolve_metrics_system")
    user = f"Catalog:\n{_catalog()}\n\nQuestion: {question!r}\nJSON array:"
    raw = chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
               max_tokens=120, timeout=timeout)
    arr = _json_span(raw, "[", "]")
    if not isinstance(arr, list):
        return []
    out: list[str] = []
    for x in arr:
        mid = x.get("id") if isinstance(x, dict) else x  # tolerate [{"id":..}] or ["id"]
        if mid in METRIC_IDS and mid not in out:         # whitelist guard, per element
            out.append(mid)
        if len(out) >= max_metrics:
            break
    return out


if __name__ == "__main__":
    print(f"endpoint: {BASE_URL}  model: {MODEL}\n")
    for term in ["관리수수료", "carry", "discount rate", "성과보수", "EV", "누적 AUM",
                 "퇴직급여충당부채", "그냥 아무 말"]:
        r = resolve_metric(term)
        print(f"  {term:16s} -> {str(r['id']):26s} (conf {r.get('confidence')})")
    print("\nmulti-metric fan-out:")
    for q in ["compare the management fee and the performance fee",
              "EBITDA와 FCFF 추이", "what is the equity value?"]:
        print(f"  {q:48s} -> {resolve_metrics(q)}")
