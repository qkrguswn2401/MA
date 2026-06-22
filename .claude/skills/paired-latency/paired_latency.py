"""Paired wall-clock latency A/B for the wiki agent — same question, two arms, back-to-back.

Usage:
  PYTHONPATH=$PWD EVAL_DATASET=v0.2 python paired_latency.py \
      --qids /tmp/diff_qids.json --off 'MNA_AGENT_ROUTES=/tmp/routes_off.yaml' --on '-'

Each --off/--on value is a single "VAR=value" env pin applied for that arm (or "-" for the
default config). For every question we time core.run under OFF then ON, back-to-back, so vLLM
load drift cancels in the per-question delta. Reports median per-question speedup. The graph is
compiled once and reused; pick a --qids set where the change actually fires (timing untouched
questions only adds noise).
"""
import argparse
import json
import os
import time
from statistics import mean, median


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--qids", required=True, help="JSON list of question ids to time")
    p.add_argument("--off", required=True, help='env pin for arm A, e.g. "MNA_X=1" or "-"')
    p.add_argument("--on", required=True, help='env pin for arm B, or "-" for default')
    p.add_argument("--dataset", default=os.environ.get("EVAL_DATASET", "v0.2"))
    p.add_argument("--qa", default="test_data/v0.2/ground_truth/qa.jsonl")
    return p.parse_args()


def set_pin(pin, prev):
    """Apply 'VAR=value' (or '-' to clear) and return the var name so we can restore."""
    if prev:
        os.environ.pop(prev, None)
    if pin == "-":
        return None
    var, _, val = pin.partition("=")
    os.environ[var] = val
    return var


def main():
    a = parse_args()
    os.environ["EVAL_DATASET"] = a.dataset
    from apps.agent import core, datasets
    from apps.agent.backends.wiki import build_app

    qids = json.load(open(a.qids))
    qmap = {q["id"]: q for q in
            (json.loads(l) for l in open(a.qa) if l.strip())}
    store = datasets.get_store(a.dataset)
    app = build_app(store.index)

    def timed(question):
        t = time.perf_counter()
        core.run(question, max_steps=3, app=app, store=store)
        return time.perf_counter() - t

    off_t, on_t, prev = [], [], None
    for qid in qids:
        q = qmap[qid]["question"]
        prev = set_pin(a.off, prev); d_off = timed(q)
        prev = set_pin(a.on, prev);  d_on = timed(q)
        off_t.append(d_off); on_t.append(d_on)
        print(f"  {qid:8s}  OFF {d_off:5.1f}s   ON {d_on:5.1f}s   Δ {d_on-d_off:+5.1f}s", flush=True)
    set_pin("-", prev)

    if off_t:
        mo, mn = median(off_t), median(on_t)
        print("\n" + "=" * 52)
        print(f"n={len(off_t)} questions, serial paired")
        print(f"  OFF  median {mo:5.1f}s   mean {mean(off_t):5.1f}s")
        print(f"  ON   median {mn:5.1f}s   mean {mean(on_t):5.1f}s")
        print(f"  per-question speedup: median {mo-mn:+.1f}s ({100*(mo-mn)/mo:+.0f}%)")


if __name__ == "__main__":
    main()
