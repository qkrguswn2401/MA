"""RAGAS scoring of the stella cross-check answers — async, task-tuned DiscreteMetrics.

The retrieval/grounding axis the tier-aware judge (``stella_crosscheck.judge``) doesn't
measure. Consumes ``data/eval_stella/answers.json`` (which records ``retrieved_contexts`` —
the wiki agent's evidence cells) and scores each answer with the local gemma vLLM.

Why custom DiscreteMetrics instead of stock RAGAS metrics: this agent does **deterministic
arithmetic** over cited cells (CLAUDE.md: compute, don't transcribe), so a correct answer like
"EBITDA = 영업이익 5,734.16 + 감가상각 56.17 = 5,790.33" has a *sum* that's in no single
evidence cell. Stock ``Faithfulness`` (literal entailment) and ``FactualCorrectness`` (numeric
claim-NLI on a 31B judge) score such answers ~0 — false negatives. We replace them with two
gold/arithmetic-aware ``DiscreteMetric``s, and keep stock ``ContextRecall`` (which doesn't
depend on arithmetic and gave a real retrieval signal):

  - **grounded_faithfulness** (custom) — every figure is in, or arithmetic-derivable from, the
    cited cells → grounded / partial / ungrounded.
  - **answer_correctness** (custom) — gold-anchored compare-don't-recompute → correct / partial
    / incorrect (the tier judge's philosophy as a metric).
  - **context_recall** (stock) — did retrieval surface the cells the gold answer needs?
    (conservative: penalizes derived gold figures not literally in a cell)
  - **retrieval_sufficiency** (custom) — the arithmetic-fair version: are the cells needed to
    *derive* the gold answer present? credits totals/%/ratios when their operands were retrieved.

Async (RAGAS modern API): one ``AsyncOpenAI`` client → ``llm_factory`` instructor LLM. Every
(record × metric) ``ascore`` is a coroutine, run concurrently under a semaphore so gemma isn't
saturated. DiscreteMetrics take ``llm`` at ``ascore``; collection metrics take it at __init__.

    .venv-ragas/bin/python -m eval.ragas_eval

Env (shared with the agent): ``STELLA_LLM_URL`` (default http://123.37.5.219:8001/v1),
``STELLA_LLM_MODEL`` (default gemma-4-31B-it), ``RAGAS_CONCURRENCY`` (default 6).
"""
from __future__ import annotations

import asyncio
import csv
import json
import math
from pathlib import Path

from src.stella_kb import config

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "data" / "eval_stella"
ANSWERS_JSON = EVAL_DIR / "answers.json"
RAGAS_CSV = EVAL_DIR / "ragas_scores.csv"
RAGAS_MD = EVAL_DIR / "ragas_report.md"

BASE_URL = config.llm_url()
MODEL = config.llm_model()
CONCURRENCY = config.ragas_concurrency()

METRIC_COLS = ["grounded_faithfulness", "answer_correctness", "context_recall",
               "retrieval_sufficiency"]
NAN = float("nan")
# discrete label -> numeric (unknown -> NaN). 'partial' shared across the custom metrics.
_LABEL = {"grounded": 1.0, "correct": 1.0, "sufficient": 1.0, "partial": 0.5,
          "ungrounded": 0.0, "incorrect": 0.0, "insufficient": 0.0}

# --- task-tuned prompts (Korean, per repo rule). {vars} are filled by ascore kwargs. --------

_GROUNDED_PROMPT = (
    "다음은 M&A 밸류에이션 질의에 대한 에이전트 답변과, 그 답변이 인용한 근거 셀(Excel 셀=값)이다.\n"
    "답변에 등장하는 핵심 수치 각각이 (1) 근거 셀에 직접 존재하거나 (2) 근거 셀 값들로부터 "
    "사칙연산(합·차·비율 등)으로 도출 가능한지 판정하라. 계산으로 도출된 값도 근거 셀이 "
    "피연산자를 제공하면 '근거 있음'으로 본다. 산수를 직접 검산하지는 말고 도출 가능성만 보라.\n\n"
    "- grounded: 핵심 수치가 모두 근거 셀에 있거나 근거 셀로부터 도출 가능.\n"
    "- partial: 일부만 그렇다.\n"
    "- ungrounded: 핵심 수치가 근거 셀로 뒷받침되지 않는다(환각 위험).\n\n"
    "[답변]\n{response}\n\n[근거 셀]\n{contexts}\n\n"
    "grounded / partial / ungrounded 중 하나로만 판정하라."
)

_CORRECT_PROMPT = (
    "M&A 밸류에이션 질문에 대한 골든(정답) 답과 에이전트 답을 비교해 채점하라.\n"
    "산수를 새로 하지 말고 골든 답을 기준으로 비교만 하라. 반올림 차이는 일치로 본다.\n\n"
    "- correct: 핵심 수치와 근거가 골든 답과 일치(±반올림).\n"
    "- partial: 일부만 일치하거나 핵심 근거 일부 누락.\n"
    "- incorrect: 틀리거나 무관.\n\n"
    "[질문]\n{question}\n\n[골든 답]\n{reference}\n(Excel 앵커: {anchor} · 산식: {method})\n\n"
    "[에이전트 답]\n{response}\n\n"
    "correct / partial / incorrect 중 하나로만 판정하라."
)

# retrieval_sufficiency = the arithmetic-fair version of context_recall. Stock ContextRecall
# checks literal attribution of gold claims to the cells, so it penalizes DERIVED gold figures
# (합계/%/배수) that aren't in any single cell even when the operand cells WERE retrieved. This
# asks the gold-deriving question instead.
_SUFFICIENCY_PROMPT = (
    "M&A 밸류에이션 질문의 골든(정답) 답을 도출하는 데 필요한 정보가, 검색된 근거 셀에 들어 "
    "있는지 판정하라.\n"
    "핵심: 골든 답의 합계·비중(%)·배수·증감 같은 '파생 수치'는 근거 셀에 그대로 없어도, 그 "
    "수치를 계산할 피연산자 셀이 검색되어 있으면 '충분'으로 본다(직접 검산은 하지 말고 도출 "
    "가능성만 보라). 골든 답이 '제공 데이터로 확인 불가/데이터 부족'을 핵심으로 하면, 검색된 "
    "근거에도 그 수치가 없어 결론과 일치할 때 '충분'으로 본다.\n\n"
    "- sufficient: 골든 답을 도출(또는 그 '확인 불가' 결론을 지지)하는 데 필요한 셀이 모두 검색됨.\n"
    "- partial: 일부만 검색됨.\n"
    "- insufficient: 필요한 핵심 셀이 검색되지 않음.\n\n"
    "[질문]\n{question}\n\n[골든 답]\n{reference}\n\n[검색된 근거 셀]\n{contexts}\n\n"
    "sufficient / partial / insufficient 중 하나로만 판정하라."
)


def _build():
    """One instructor LLM over gemma; two custom DiscreteMetrics + stock ContextRecall."""
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics import DiscreteMetric
    from ragas.metrics.collections import ContextRecall

    llm = llm_factory(MODEL, client=AsyncOpenAI(base_url=BASE_URL, api_key="x"))
    grounded = DiscreteMetric(name="grounded_faithfulness",
                              allowed_values=["grounded", "partial", "ungrounded"],
                              prompt=_GROUNDED_PROMPT)
    correct = DiscreteMetric(name="answer_correctness",
                             allowed_values=["correct", "partial", "incorrect"],
                             prompt=_CORRECT_PROMPT)
    sufficiency = DiscreteMetric(name="retrieval_sufficiency",
                                 allowed_values=["sufficient", "partial", "insufficient"],
                                 prompt=_SUFFICIENCY_PROMPT)
    return llm, grounded, correct, ContextRecall(llm=llm), sufficiency


async def _discrete(sem, metric, llm, **kw) -> tuple[float, str]:
    """Run a DiscreteMetric.ascore under the semaphore -> (numeric, reason). NaN on flake."""
    async with sem:
        try:
            r = await metric.ascore(llm=llm, **kw)
            return _LABEL.get(str(r.value).strip().lower(), NAN), str(getattr(r, "reason", "") or "")
        except Exception:  # noqa: BLE001 — one gemma/instructor flake -> NaN, not abort
            return NAN, ""


async def _collection(sem, metric, **kw) -> float:
    async with sem:
        try:
            return float((await metric.ascore(**kw)).value)
        except Exception:  # noqa: BLE001
            return NAN


async def _score_record(r: dict, M, sem: asyncio.Semaphore) -> dict:
    llm, grounded, correct, ctx_metric, sufficiency = M
    q = r.get("question", "")
    resp = r.get("agent_answer", "") or ""
    ref = r.get("answer", "") or ""
    ctx = list(r.get("retrieved_contexts") or [])
    has = bool(ctx)

    async def _grounded():
        if not has:
            return NAN, ""
        return await _discrete(sem, grounded, llm, response=resp, contexts="\n".join(ctx))

    async def _correct():
        return await _discrete(sem, correct, llm, question=q, reference=ref,
                               anchor=r.get("excel_anchor", ""), method=r.get("method", ""),
                               response=resp)

    async def _recall():
        if not has:
            return NAN
        return await _collection(sem, ctx_metric, user_input=q, retrieved_contexts=ctx,
                                 reference=ref)

    async def _sufficiency():
        return await _discrete(sem, sufficiency, llm, question=q, reference=ref,
                               contexts="\n".join(ctx) if ctx else "(검색된 근거 없음)")

    (gv, gr), (cv, cr), rv, (sv, sr) = await asyncio.gather(
        _grounded(), _correct(), _recall(), _sufficiency())
    row = {"id": r.get("id", "?"), "tier": r.get("tier"),
           "grounded_faithfulness": gv, "answer_correctness": cv, "context_recall": rv,
           "retrieval_sufficiency": sv,
           "_grounded_reason": gr, "_correct_reason": cr, "_sufficiency_reason": sr}
    print(f"  {row['id']:<5} T{row['tier']}  "
          + "  ".join(f"{c}={_fmt(row[c])}" for c in METRIC_COLS))
    return row


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def _mean(vals) -> float:
    keep = [v for v in vals if v is not None and not math.isnan(v)]
    return sum(keep) / len(keep) if keep else NAN


def _write_report(rows: list[dict], n_ctx: int) -> None:
    with RAGAS_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=["id", "tier", *METRIC_COLS,
                                           "_grounded_reason", "_correct_reason",
                                           "_sufficiency_reason"])
        w.writeheader()
        w.writerows(rows)

    def agg(rs):
        return " | ".join(_fmt(_mean([r[c] for r in rs])) for c in METRIC_COLS)

    def cov(rs):
        return " | ".join(f"{sum(1 for r in rs if not math.isnan(r[c]))}/{len(rs)}"
                          for c in METRIC_COLS)

    lines = ["# Stella cross-check — RAGAS async (custom DiscreteMetrics, gemma judge)", "",
             f"answers: `{ANSWERS_JSON.name}` · {len(rows)} Q · {n_ctx} with retrieved context "
             f"· model `{MODEL}` · concurrency {CONCURRENCY}", "",
             "metrics: **grounded_faithfulness**, **answer_correctness**, "
             "**retrieval_sufficiency** are custom (arithmetic/gold-aware) DiscreteMetrics; "
             "**context_recall** is stock RAGAS (conservative — literal attribution).", "",
             "| scope | n | " + " | ".join(METRIC_COLS) + " |",
             "|---|---|" + "---|" * len(METRIC_COLS),
             f"| **all** | {len(rows)} | {agg(rows)} |"]
    for t in sorted({r["tier"] for r in rows if r["tier"] is not None}):
        sub = [r for r in rows if r["tier"] == t]
        lines.append(f"| T{int(t)} | {len(sub)} | {agg(sub)} |")
    lines += ["", f"_coverage (non-NaN / n): all = {cov(rows)}_", "",
              "## Per-question", "",
              "| Q | tier | " + " | ".join(METRIC_COLS) + " |",
              "|---|---|" + "---|" * len(METRIC_COLS)]
    for r in rows:
        lines.append(f"| {r['id']} | T{int(r['tier'])} | "
                     + " | ".join(_fmt(r[c]) for c in METRIC_COLS) + " |")
    RAGAS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nwrote {RAGAS_CSV}\nwrote {RAGAS_MD}")


async def main_async() -> int:
    if not ANSWERS_JSON.exists():
        print(f"no {ANSWERS_JSON} — run `python -m eval.stella_crosscheck eval` first.")
        return 1
    records = json.loads(ANSWERS_JSON.read_text(encoding="utf-8"))
    n_ctx = sum(1 for r in records if r.get("retrieved_contexts"))
    print(f"loaded {len(records)} answers · {n_ctx} carry retrieved_contexts · "
          f"model={MODEL} · concurrency={CONCURRENCY}\nscoring (async) ...")

    M = _build()
    sem = asyncio.Semaphore(CONCURRENCY)
    rows = await asyncio.gather(*(_score_record(r, M, sem) for r in records))
    order = [r.get("id") for r in records]
    rows = sorted(rows, key=lambda r: order.index(r["id"]))
    _write_report(rows, n_ctx)
    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
