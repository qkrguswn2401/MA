"""Evaluate the wiki agent on the **v0.2 vision-QA** ground-truth set (rubric-based).

This set (``test_data/v0.2/ground_truth/qa.jsonl``, 54 items over the STELLA/CAESAR/LIFE FDD
decks) is *not* the tier-based Excel cross-check (see ``stella_crosscheck.py``). Each record is
a born-digital advisory-deck **visual** question — structure diagrams, charts, matrices — with
a golden ``ground_truth`` and a per-item ``rubric`` instead of a tier. So judging is
**rubric-driven**: an LLM compares the agent's answer to the golden answer under the item's own
rubric and scores 1.0 / 0.5 / 0.0.

The agent runs against a **prebuilt wiki** selected by dataset id (default ``v0.2``) via the
dataset-store API — no global rebinding, no rebuild here.

    python -m eval.qa_eval                 # eval + judge the v0.2 wiki -> data/eval_v0.2_qa
    python -m eval.qa_eval eval            # run answers only
    python -m eval.qa_eval judge           # re-judge existing answers
    EVAL_DATASET=default python -m eval.qa_eval   # score a different built dataset
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.stella_kb import config

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = Path(os.environ.get("EVAL_QA", str(ROOT / "test_data" / "v0.2" / "ground_truth" / "qa.jsonl")))
DATASET = os.environ.get("EVAL_DATASET", "v0.2")  # which built wiki the agent reads
EVAL_DIR = Path(os.environ.get("EVAL_OUT_DIR", str(ROOT / "data" / "eval" / "v0.2")))
ANSWERS_JSON = EVAL_DIR / "answers.json"
SCORES_JSON = EVAL_DIR / "scores.json"
REPORT_MD = EVAL_DIR / "report.md"


def load_questions() -> list[dict]:
    return [json.loads(ln) for ln in QUESTIONS.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --- stage 1: run every question through the agent against the chosen wiki ---------------


def _answer_one(q: dict, app, store) -> dict:
    from apps.agent import core

    try:
        res = core.run(q["question"], max_steps=3, app=app, store=store)
        ans = res["answer"]
        pages = sorted(
            {
                e.get("arg")
                for e in res.get("trace", [])
                if e.get("action") == "route" and e.get("arg") not in (None, "(none)")
            }
        )
    except Exception as e:  # noqa: BLE001 — record the failure, keep going
        ans, pages = f"[ERROR] {type(e).__name__}: {e}", []
    print(f"--- {q['id']} ({q.get('doc')}/{q.get('capability')})  {ans[:70].replace(chr(10), ' ')}")
    return {
        "id": q["id"],
        "doc": q.get("doc"),
        "capability": q.get("capability"),
        "visual_type": q.get("visual_type"),
        "difficulty": q.get("difficulty"),
        "question": q["question"],
        "ground_truth": q.get("ground_truth", ""),
        "rubric": q.get("rubric", ""),
        "agent_answer": ans,
        "pages_opened": pages,
    }


def run_eval(workers: int = 8) -> None:
    """Answer all questions concurrently against ``DATASET``'s prebuilt wiki (store-based)."""
    from apps.agent import datasets
    from apps.agent.backends.wiki import build_app, nodes

    store = datasets.get_store(DATASET)
    if not store.exists():
        sys.exit(
            f"qa_eval: dataset {DATASET!r} not built ({store.index_json} missing). "
            "Build it first (run_pipeline.sh with MNA_WIKI_DATA=...)."
        )
    qs = load_questions()
    workers = min(workers, len(qs))
    nodes.set_fanout(config.eval_fanout(default=workers))  # don't starve the workers
    app = build_app(store.index)  # compile once, reuse across items
    print(
        f"answering {len(qs)} questions · dataset={DATASET} ({len(store.index['pages'])} pages) "
        f"· {workers} workers"
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda q: _answer_one(q, app, store), qs))
    order = {q["id"]: i for i, q in enumerate(qs)}
    results.sort(key=lambda r: order[r["id"]])
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    ANSWERS_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {ANSWERS_JSON}  ({len(results)} answers)")


# --- stage 2: rubric-based LLM judge ----------------------------------------------------

_JUDGE_SYS = (
    "당신은 비전 기반 RAG 답변 채점관이다. 질문, 골든 정답(ground_truth), 채점 기준(rubric), "
    "그리고 에이전트의 답을 받는다. **반드시 rubric을 기준으로** 에이전트 답이 골든 정답과 맞는지 "
    "판정하라. 산수를 새로 하지 말고 골든 정답과 비교만 한다. 수치는 반올림 오차까지 허용한다. "
    "rubric이 '둘 다 맞아야 정답'이면 일부만 맞으면 0.5다. C5(정직성) 유형은 '데이터로 확인 불가'를 "
    "정직하게 답하면 1.0, 없는 값을 지어내면 0.0이다. "
    'JSON만 출력: {"score": 1.0|0.5|0.0, "verdict": "correct|partial|incorrect", "reason": "한 문장"}'
)


def _judge_one(rec: dict) -> dict:
    from src.stella_kb import llm

    user = (
        f"[질문]\n{rec['question']}\n\n"
        f"[골든 정답]\n{rec['ground_truth']}\n\n"
        f"[채점 기준 rubric]\n{rec['rubric']}\n\n"
        f"[에이전트 답]\n{rec['agent_answer']}\n\nJSON:"
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content": _JUDGE_SYS}, {"role": "user", "content": user}], max_tokens=1000, timeout=90
        )
        obj = llm._json_span(raw, "{", "}") or {}
    except Exception as e:  # noqa: BLE001
        obj = {"score": 0.0, "verdict": "error", "reason": f"{type(e).__name__}: {e}"}
    score = obj.get("score")
    if score not in (0.0, 0.5, 1.0):
        score = 0.0
    return {
        "id": rec["id"],
        "doc": rec["doc"],
        "capability": rec["capability"],
        "visual_type": rec["visual_type"],
        "difficulty": rec["difficulty"],
        "score": float(score),
        "verdict": obj.get("verdict", "?"),
        "reason": obj.get("reason", ""),
    }


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _breakdown_table(title: str, key: str, scores: list[dict]) -> list[str]:
    groups: dict[object, list[float]] = defaultdict(list)
    for s in scores:
        groups[s.get(key)].append(s["score"])
    rows = [f"### {title}", "", "| 그룹 | n | mean |", "|---|---|---|"]
    for g in sorted(groups, key=lambda x: str(x)):
        v = groups[g]
        rows.append(f"| {g} | {len(v)} | {_mean(v):.2f} |")
    rows.append("")
    return rows


def judge() -> None:
    recs = json.loads(ANSWERS_JSON.read_text(encoding="utf-8"))
    print(f"judging {len(recs)} answers (rubric-based) ...")
    with ThreadPoolExecutor(max_workers=4) as ex:
        scores = list(ex.map(_judge_one, recs))
    by_id = {s["id"]: s for s in scores}
    SCORES_JSON.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")

    all_scores = [s["score"] for s in scores]
    lines = [
        f"# v0.2 비전-QA 평가 — wiki agent (dataset={DATASET})",
        "",
        f"**전체: {len(scores)}문항 · 평균 {_mean(all_scores):.2f}**",
        "",
    ]
    lines += _breakdown_table("프로젝트(doc)별", "doc", scores)
    lines += _breakdown_table("능력축(capability)별", "capability", scores)
    lines += _breakdown_table("시각유형(visual_type)별", "visual_type", scores)
    lines += _breakdown_table("난이도(difficulty)별", "difficulty", scores)
    lines += ["## 문항별", "", "| ID | doc | capability | score | verdict | reason |", "|---|---|---|---|---|---|"]
    for r in recs:
        s = by_id[r["id"]]
        lines.append(
            f"| {r['id']} | {r['doc']} | {r['capability']} | {s['score']:.1f} | "
            f"{s['verdict']} | {s['reason'].replace('|', '/')} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14]))
    print(f"\nwrote {SCORES_JSON}\nwrote {REPORT_MD}")


if __name__ == "__main__":
    cmds = sys.argv[1:] or ["eval", "judge"]
    print(f"qa_eval: questions={QUESTIONS.name} dataset={DATASET} out={EVAL_DIR} cmds={cmds}")
    for cmd in cmds:
        if cmd in ("eval", "all"):
            run_eval()
        if cmd in ("judge", "all"):
            judge()
