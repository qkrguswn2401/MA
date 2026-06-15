"""Vision LLM 클라이언트 — 페이지 이미지 → markdown (gemma-4 멀티모달 vLLM).

벤더 코드의 ``core.llm.backends.vllm_openai.invoke_vision`` 자리를 대체하는 얇은 shim.
프로젝트의 ``src/stella_kb/llm.py`` 와 동일 규약: stdlib urllib, OpenAI 호환,
env ``STELLA_LLM_URL`` / ``STELLA_LLM_MODEL`` (default gemma-4-31B-it @ :8001).

Gemma 는 별도 ``system`` role 을 지원하지 않으므로(chat template) system 텍스트를
user 메시지 첫 text 세그먼트로 접어 넣는다. 이미지는 base64 data URL 로 전송.
JSON/tool 강제 없이 페이지 markdown 을 직접 받아 반환 — 로컬 31B 모델에서 JSON
강제는 취약하고, 출력이 어차피 markdown 한 덩어리라 파싱이 불필요.

``get_or_compute`` 는 (model, system, user) 해시 기반 디스크 캐시 — 같은 페이지를
재실행해도 LLM 을 다시 부르지 않는다(벤더 ``core.llm.cache`` 의 최소 대체).
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import urllib.request
from collections.abc import Callable
from pathlib import Path

from ...config import llm_model, llm_url, pdf_vision_cache

BASE_URL = llm_url()
MODEL = llm_model()

_CACHE_DIR = Path(pdf_vision_cache())

log = logging.getLogger("stella_kb.parsers.pdf.vision")


def _b64_data_url(image_path: str | Path) -> str:
    raw = Path(image_path).read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def invoke_vision(
    *, system: str, prompt: str, image_path: str | Path,
    model: str | None = None, max_tokens: int = 8000, timeout: float = 240.0,
) -> str:
    """이미지 1장 + 프롬프트 → 모델 응답 텍스트(페이지 markdown).

    실패(HTTP/빈 응답) 시 RuntimeError — 호출 측이 재시도/폴백을 결정한다.
    """
    model = model or MODEL
    content = [
        {"type": "text", "text": f"{system}\n\n{prompt}"},
        {"type": "image_url", "image_url": {"url": _b64_data_url(image_path)}},
    ]
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/chat/completions", data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
    except Exception as e:  # noqa: BLE001 — 네트워크/엔드포인트 실패를 통일 타입으로
        raise RuntimeError(f"vision endpoint 호출 실패: {e}") from e
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"vision 응답 파싱 실패: {data}") from e
    if not text or not text.strip():
        raise RuntimeError("vision 응답이 비어 있음")
    return text.strip()


def _cache_key(model: str, system: str, user: str) -> str:
    h = hashlib.sha256()
    for part in (model, system, user):
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def get_or_compute(
    *, model: str, system: str, user: str, compute: Callable[[], str],
) -> str:
    """디스크 캐시 wrapper. 캐시 히트면 즉시 반환, 미스면 compute() 결과를 적재.

    성공 결과만 저장한다(compute 가 raise 하면 캐시에 남지 않음).
    """
    key = _cache_key(model, system, user)
    path = _CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))["markdown"]
        except Exception:  # noqa: BLE001 — 손상 캐시는 무시하고 재계산
            log.warning("손상된 캐시 무시: %s", path)
    result = compute()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"markdown": result}, ensure_ascii=False), encoding="utf-8")
    return result
