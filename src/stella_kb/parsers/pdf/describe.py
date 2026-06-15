"""PDF describe 파서 — 페이지 이미지를 gemma-4 멀티모달 vLLM 으로 읽어 markdown 화.

벤더 ``core.parsers.pdf_describe`` 의 **per-page vision (vllm)** 경로만 남긴 슬림 버전.
gateway(anthropic) / agent_sdk(claude -p) 분기와 그에 딸린 ``core.llm.backends`` 의존을
전부 제거했다 — 이 프로젝트는 사내 gemma vLLM(:8001) 하나만 쓴다.

페이지마다:
  1. pymupdf 로 PNG 렌더(dpi 220) → 모델 입력 이미지
  2. pymupdf 텍스트(``text.parse_pdf``)를 reference 로 같이 전달(철자·숫자 authoritative)
  3. invoke_vision → 페이지 markdown (실패 시 3회 재시도 후 pymupdf 텍스트로 degrade)
  4. 표는 description-only + 검색행으로 교체하고 PdfTablePayload 로 분리(값 임베딩 X)

vision 호출은 결과를 디스크 캐시(``vision.get_or_compute``)에 적재 — 재실행 시 무과금.
페이지는 ThreadPool 로 병렬(``MNA_PDF_DESCRIBE_CONCURRENCY``, default 4).
"""
from __future__ import annotations

import hashlib
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # pymupdf

from ...config import pdf_describe_concurrency, pdf_page_png_cache
from . import vision
from .state import SourceAbbrev, SourcePage
from .tables import PdfTablePayload, extract_tables_from_markdown
from .text import parse_pdf

log = logging.getLogger("stella_kb.parsers.pdf.describe")

DEFAULT_DPI = 220
_PAGE_CACHE_DIR = Path(pdf_page_png_cache())
_VISION_RETRIES = 3  # vision flake (빈 응답 등) 재시도 — backoff 2s/5s

_SYSTEM = (
    "You are a precise parser for Korean financial deal materials "
    "(IM, CDD, FM, Legal, 회의록, 프레젠테이션). "
    "Extract EVERY piece of information from each page faithfully. "
    "Preserve original text, all table cells, chart numbers and labels. "
    "Do NOT translate Korean. Do NOT summarize away detail. Do NOT invent."
)

_PROMPT_TMPL = (
    "이 페이지(이미지)를 빠짐없이 구조화 markdown 으로 변환하세요.\n"
    "규칙:\n"
    "- 텍스트 원문 보존. 표는 모든 행·열을 markdown 표(| col | col |)로.\n"
    "- 차트/그래프 수치·축·레이블·추세를 [그래프 N] 블록으로.\n"
    "- 흐름도·조직도 구조 살려서 표현. 다단 컬럼은 위→아래, 왼→오 순서.\n"
    "- 한국어는 한국어 그대로. 요약·창작 금지. markdown 본문만 출력.\n\n"
    "아래 reference 텍스트(pymupdf 추출)는 철자·숫자의 authoritative 근거이니 "
    "이미지와 교차검증하되, 레이아웃·표·그래프는 이미지를 우선하세요.\n"
    "=== REFERENCE TEXT (pymupdf) ===\n{ref}\n=== END ==="
)


@dataclass
class DescribeMetrics:
    """describe 호출 집계."""

    page_count: int = 0
    latency_ms: int = 0
    table_payloads: list = field(default_factory=list)
    fallback_pages: list[int] = field(default_factory=list)  # vision 실패 → pymupdf degrade (1-based)


def _render_page_png(path: Path, page_num: int, *, file_sha: str, dpi: int = DEFAULT_DPI) -> Path:
    """PDF 1페이지 → PNG (캐시). page_num 1-based."""
    out = _PAGE_CACHE_DIR / file_sha / f"p{page_num:04d}_dpi{dpi}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open(str(path)) as doc:
        doc[page_num - 1].get_pixmap(dpi=dpi).save(str(out.resolve()))
    return out


def describe_pdf(
    path: Path | str,
    abbrev: SourceAbbrev = "ETC",
    *,
    concurrency: int | None = None,
    dpi: int = DEFAULT_DPI,
    model: str | None = None,
    max_pages: int | None = None,
) -> tuple[list[SourcePage], DescribeMetrics]:
    """PDF → 페이지별 markdown ``SourcePage`` + ``DescribeMetrics``.

    Args:
        path: PDF 경로.
        abbrev: 자료 약칭 태그.
        concurrency: 동시 vision 호출 수 (default env ``MNA_PDF_DESCRIBE_CONCURRENCY`` or 4).
        dpi: 페이지 PNG 렌더 해상도.
        model: 모델 override (default env ``STELLA_LLM_MODEL``).
        max_pages: 앞에서부터 N 페이지만 처리(샘플/스모크용). None 이면 전체.
    """
    path = Path(path)
    t0 = time.perf_counter()
    with fitz.open(str(path)) as doc:
        total = doc.page_count
    if total == 0:
        return [], DescribeMetrics(page_count=0)
    if max_pages is not None:
        total = min(total, max_pages)
    if concurrency is None:
        concurrency = max(1, pdf_describe_concurrency())
    resolved_model = model or vision.MODEL

    file_sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    ref = {p.page: p.text for p in parse_pdf(path, abbrev)}  # 철자·숫자 reference
    log.info("pdf describe[vllm] · %s · %d페이지 · model=%s · conc=%d",
             path.name, total, resolved_model, concurrency)

    def _describe_page(page_num: int) -> tuple[int, str, bool]:
        png = _render_page_png(path, page_num, file_sha=file_sha, dpi=dpi)
        ref_text = ref.get(page_num, "(reference unavailable)")
        prompt = _PROMPT_TMPL.format(ref=ref_text)
        png_sha = hashlib.sha256(png.read_bytes()).hexdigest()[:16]
        cache_user = prompt + f"\n[png:{png_sha}]"

        def _compute() -> str:
            backoffs = [2.0, 5.0]
            for attempt in range(1, _VISION_RETRIES + 1):
                try:
                    return vision.invoke_vision(
                        system=_SYSTEM, prompt=prompt, image_path=str(png),
                        model=resolved_model,
                    )
                except RuntimeError as e:
                    if attempt >= _VISION_RETRIES:
                        raise
                    log.warning("vision 실패 page=%d attempt=%d/%d 재시도: %s",
                                page_num, attempt, _VISION_RETRIES, e)
                    time.sleep(backoffs[attempt - 1])
            raise RuntimeError("unreachable")

        try:
            md = vision.get_or_compute(
                model=resolved_model, system=_SYSTEM, user=cache_user, compute=_compute)
        except RuntimeError as e:
            log.error("vision 최종 실패 page=%d — pymupdf 텍스트 폴백: %s", page_num, e)
            fallback = ref.get(page_num) or f"(vision describe failed: page {page_num})"
            if fallback == "(reference unavailable)":
                fallback = f"(vision describe failed: page {page_num})"
            return page_num, fallback, True
        return page_num, md, False

    page_nums = list(range(1, total + 1))
    if concurrency > 1 and total > 1:
        with ThreadPoolExecutor(max_workers=min(concurrency, total)) as ex:
            pairs = list(ex.map(_describe_page, page_nums))
    else:
        pairs = [_describe_page(n) for n in page_nums]
    pairs.sort(key=lambda t: t[0])

    # 조립 — 표를 description+검색행으로 교체하고 payload 분리 (값 임베딩 X).
    all_payloads: list[PdfTablePayload] = []
    pages: list[SourcePage] = []
    fallback_pages: list[int] = []
    for n, md, is_fallback in pairs:
        if is_fallback:
            fallback_pages.append(n)
        cleaned_md, page_payloads = extract_tables_from_markdown(
            md, page=n, table_offset=len(all_payloads))
        for tp in page_payloads:
            tp.abbrev = abbrev
            tp.file = str(path)
        all_payloads.extend(page_payloads)
        pages.append(SourcePage(abbrev=abbrev, file=path, page=n, text=cleaned_md))

    metrics = DescribeMetrics(
        page_count=total,
        latency_ms=int((time.perf_counter() - t0) * 1000),
        table_payloads=all_payloads,
        fallback_pages=fallback_pages,
    )
    log.info("pdf_describe: pages=%d tables=%d fallback=%s latency_ms=%d file=%s",
             metrics.page_count, len(all_payloads), fallback_pages or "-",
             metrics.latency_ms, path.name)
    return pages, metrics
