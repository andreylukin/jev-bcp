"""Which documents should get every window graded? Jev's doc score today comes from ONE term-overlap window,
and only ~66% of gold docs reach its top 10 (recall.py). Grading every window of every hit costs ~6x the tokens.

  pool     BM25 top-100 + dense top-100 for the question text, plus the gold docs (this measures ranking, not search)
  overlap  score = Jev on the term-overlap window                       (today)
  cascade  overlap for all; the top K by overlap get every window graded, score = max over windows
  full     every window of every doc (K = all)

    uv run python -m bcp.rank --n 100
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import jev
from bcp.corpus import Corpus
from bcp.data import split
from bcp.dense import Dense

KS = (8, 16, 32, 64)
TOPS = (8, 12, 20)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dense = Corpus(), Dense()

    with httpx.Client(timeout=300) as client:

        def one(case):
            pool = list(dict.fromkeys(corpus.search(case.query, 100) + dense.search(case.query, 100) + list(case.gold)))
            overlap, _, t_overlap = jev.score_batched(client, case.query, [(d, corpus.window(d, case.query)) for d in pool])
            parts = [(f"{d}#{i}", w) for d in pool for i, w in enumerate(corpus.windows(d))]
            s, _, t_full = jev.score_batched(client, case.query, parts)
            best = {d: max(v for k, v in s.items() if k.split("#")[0] == d) for d in pool}
            chars = {d: sum(len(w) for w in corpus.windows(d)) for d in pool}
            by_overlap = sorted(pool, key=lambda d: -overlap[d])
            gold = set(case.gold)
            out = {"qid": case.qid, "pool": len(pool), "tokens": {"overlap": t_overlap, "full": t_overlap + t_full}, "recall": {}}
            arms = {"overlap": overlap, "full": best}
            for k in KS:
                arms[f"cascade{k}"] = {**overlap, **{d: best[d] for d in by_overlap[:k]}}
                out["tokens"][f"cascade{k}"] = t_overlap + t_full * sum(chars[d] for d in by_overlap[:k]) / sum(chars.values())
            for arm, sc in arms.items():
                ranked = sorted(pool, key=lambda d: -sc[d])
                out["recall"][arm] = {str(t): len(gold & set(ranked[:t])) / len(gold) for t in TOPS}
            print(f"{case.qid:>5} pool {len(pool)}  top8 overlap {out['recall']['overlap']['8']:.2f} cascade32 {out['recall']['cascade32']['8']:.2f} full {out['recall']['full']['8']:.2f}", flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, dev))
    Path("out/rank").mkdir(parents=True, exist_ok=True)
    Path("out/rank/results.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{n} questions, pool {sum(r['pool'] for r in res) / n:.0f} docs")
    for arm in res[0]["recall"]:
        print(f"{arm:10} gold recall " + "  ".join(f"top{t} {sum(r['recall'][arm][str(t)] for r in res) / n:.3f}" for t in TOPS)
              + f"   Jev tokens/q {sum(r['tokens'][arm] for r in res) / n:,.0f}")


if __name__ == "__main__":
    main()
