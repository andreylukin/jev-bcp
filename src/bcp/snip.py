"""Can Jev do the reading triage, so the LLM reads less text from more documents? Labels only: no LLM, no judge.

  windows   the 8 best-graded docs, one 4000-char window each (~32k chars)              (today's reader input)
  snips     the 40 best-graded docs' windows cut into ~400-char excerpts; Jev grades every excerpt;
            the reader gets the best excerpts up to the same 32k chars
  snips/4   the same at 8k chars

Metric: is the gold answer string in the reader's input, and what share of gold docs contribute any text to it.
Questions whose answer string appears in no pool document are left out of the first metric (the judge accepts
paraphrases; a substring test cannot). Same pool as rank.py, so its doc grades come from the Jev cache.

    uv run python -m bcp.snip --n 50
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import jev
from bcp.corpus import Corpus, excerpts
from bcp.data import split
from bcp.dense import Dense

DOCS = 40
BUDGETS = {"snips": 32_000, "snips/4": 8_000}


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dense = Corpus(), Dense()

    with httpx.Client(timeout=300) as client:

        def one(case):
            pool = list(dict.fromkeys(corpus.search(case.query, 100) + dense.search(case.query, 100) + list(case.gold)))
            win = {d: corpus.window(d, case.query) for d in pool}
            score, _, _ = jev.score_batched(client, case.query, list(win.items()))
            ranked = sorted(pool, key=lambda d: -score[d])
            parts = [(f"{d}#{i}", e) for d in ranked[:DOCS] for i, e in enumerate(excerpts(win[d]))]
            s, _, tokens = jev.score_batched(client, case.query, parts)
            text = dict(parts)
            inputs = {"windows": [(d, win[d]) for d in ranked[:8]]}
            for arm, budget in BUDGETS.items():
                picked, size = [], 0
                for k in sorted(s, key=lambda k: -s[k]):
                    if size + len(text[k]) > budget:
                        break
                    picked.append((k.split("#")[0], text[k]))
                    size += len(text[k])
                inputs[arm] = picked
            ans, gold = norm(case.answer), set(case.gold)
            out = {"qid": case.qid, "answerable": any(ans in norm(corpus.text[d]) for d in pool), "tokens": tokens}
            for arm, items in inputs.items():
                out[arm] = {"has_answer": any(ans in norm(t) for _, t in items), "gold_docs": len(gold & {d for d, _ in items}) / len(gold),
                            "chars": sum(len(t) for _, t in items), "docs": len({d for d, _ in items})}
            print(f"{case.qid:>5} answerable={out['answerable']!s:5} " + "  ".join(f"{arm} ans={out[arm]['has_answer']!s:5} gold={out[arm]['gold_docs']:.2f}" for arm in inputs), flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, dev))
    Path("out/snip").mkdir(parents=True, exist_ok=True)
    Path("out/snip/results.json").write_text(json.dumps(res, indent=1))
    n, able = len(res), [r for r in res if r["answerable"]]
    print(f"{n} questions, {len(able)} with the answer string somewhere in the pool; excerpt grading {sum(r['tokens'] for r in res) / n:,.0f} Jev tokens/q")
    for arm in ("windows", *BUDGETS):
        print(f"{arm:8} answer string in reader input {sum(r[arm]['has_answer'] for r in able) / len(able):.3f}   gold docs represented {sum(r[arm]['gold_docs'] for r in res) / n:.3f}"
              f"   {sum(r[arm]['chars'] for r in res) / n:,.0f} chars from {sum(r[arm]['docs'] for r in res) / n:.0f} docs")


if __name__ == "__main__":
    main()
