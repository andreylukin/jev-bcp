"""Replay the final stage over a finished dev run, on ALL its questions (a change that fixes wrong answers can
also break right ones). Every arm starts from the agent's own answer and the docs it read.

  base     one best window per read doc, strict check sees notes + passages        (the stage as it ran)
  top12    the 12 best windows of all windows of the read docs
  nonotes  top12, strict check sees the passages only
  search   nonotes + a rejected answer triggers new queries and passages before the redo

    uv run python -m bcp.replay --run out/modal_dev_passages --model deepseek/deepseek-v4-flash-0731
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev
from bcp.corpus import Corpus
from bcp.data import load_all
from bcp.dense import Dense
from bcp.judge import judge

ARMS = ("base", "top12", "nonotes", "search")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/modal_dev_passages")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    cases = {c.qid: c for c in load_all()}
    rows = [r for r in json.load(open(Path(a.run) / "results.json"))["rows"] if "error" not in r]
    corpus, dense = Corpus(), Dense()

    with httpx.Client(timeout=300) as client:

        def one(r):
            q = cases[r["qid"]].query
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0)
            ask = lambda m: agent.llm(client, m, stats)
            parts = [(f"{d}#{i}", w) for d in r["read"] for i, w in enumerate(corpus.windows(d))]
            s, _, _ = jev.score_batched(client, q, parts)
            pool = {w: s[k] for k, w in parts}
            best = {}
            for k, w in parts:
                d = k.split("#")[0]
                if d not in best or s[k] > pool[best[d]]:
                    best[d] = w
            out = {"qid": r["qid"], "live": bool(r["correct"]), "gold_read": r["gold_read"]}
            for arm in ARMS:
                try:
                    seen = {d: 0.0 for d in r["read"]}
                    ans, notes, scores = final.finalize(
                        client, ask, q, str(r["llm_answer"] or ""), str(r["notes"]),
                        {best[d]: pool[best[d]] for d in best} if arm == "base" else pool,
                        use_notes=arm in ("base", "top12"),
                        search=agent.requery(q, corpus, dense, True, client, ask, seen, stats) if arm == "search" else None)
                    out[arm] = {"answer": ans, "attempts": len(scores), "score": max(scores), "correct": bool(judge(client, q, ans, cases[r["qid"]].answer, notes)["correct"])}
                except Exception as e:
                    out[arm] = {"answer": None, "attempts": 0, "score": 0.0, "correct": False, "error": repr(e)[:200]}
            print(f"{r['qid']:>5} live {'OK' if out['live'] else '--'}  " + "  ".join(f"{arm} {'OK' if out[arm]['correct'] else '--'}" for arm in ARMS), flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, rows))
    Path("out/replay").mkdir(parents=True, exist_ok=True)
    Path("out/replay/results.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{n} questions; live run {sum(x['live'] for x in res) / n:.3f}")
    for arm in ARMS:
        c = [x[arm]["correct"] for x in res]
        print(f"{arm:8} {sum(c) / n:.3f}   vs base: fixed {sum(x[arm]['correct'] and not x['base']['correct'] for x in res)} broke {sum(x['base']['correct'] and not x[arm]['correct'] for x in res)}"
              f"   attempts {sum(x[arm]['attempts'] for x in res) / n:.2f}  errors {sum('error' in x[arm] for x in res)}")


if __name__ == "__main__":
    main()
