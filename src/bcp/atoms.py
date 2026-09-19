"""Does grading documents fact by fact fix the ranking? A page from the middle of a clue chain supports ONE fact
of the question, and Jev grading it against the whole question ranks it low (rank.py: 75% of gold docs in the
top 8, and no amount of window grading moves that).

  compile  the LLM splits the question into atoms: self-contained facts, intermediate entities described, not named
  grade    one Jev request per doc: state = the doc's window (billed once), questions = the whole-question noul
           plus one noul per atom
  arms     whole   the whole-question noul                                   (today)
           max     best atom
           cover   each atom's best doc first (every fact represented), then by max

Same pool as rank.py: BM25 top-100 + dense top-100 for the question text, plus the gold docs.

    uv run python -m bcp.atoms --model deepseek/deepseek-v4-flash-0731 --n 50
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, jev
from bcp.corpus import Corpus
from bcp.data import split
from bcp.dense import Dense

TOPS = (8, 12, 20)
COMPILE = """Split the research question into the separate facts it relies on. Each fact must stand on its own: describe
intermediate people, places and works by what the question says about them, never as "the answer" or "this person".
A web page that supports just one of these facts should be recognisable from that fact alone. 3 to 7 facts.
Reply with JSON: {"atoms": ["...", "..."]}"""
ATOM = "Does the document state this fact, or specific information that establishes part of it?\n\nFact: "


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dense = Corpus(), Dense()
    jev.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=300) as client:

        def grade(question, atoms, text):
            body = {"model": jev.JEV_MODEL, "state": {"question": question, "document": text},
                    "questions": {"whole": {"type": "noul", "instructions": jev.INSTRUCTIONS, "criteria": jev.CRITERIA},
                                  **{f"a{i}": {"type": "noul", "instructions": ATOM + x} for i, x in enumerate(atoms)}}}
            path = jev._cache_path(body)
            if path.exists():
                data = json.loads(path.read_text())
            else:
                data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
                path.write_text(json.dumps(data))
            return {k: v["noul"] for k, v in data["answers"].items()}, data["usage"]["input_tokens"]

        def one(case):
            stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            atoms = [str(x) for x in agent.llm(client, [{"role": "system", "content": COMPILE}, {"role": "user", "content": case.query}], stats).get("atoms", [])][:7]
            pool = list(dict.fromkeys(corpus.search(case.query, 100) + dense.search(case.query, 100) + list(case.gold)))
            with ThreadPoolExecutor(8) as ex:
                graded = list(ex.map(lambda d: grade(case.query, atoms, corpus.window(d, case.query)), pool))
            s = {d: g for d, (g, _) in zip(pool, graded)}
            best = {d: max([v for k, v in s[d].items() if k != "whole"], default=0.0) for d in pool}
            by_max = sorted(pool, key=lambda d: -best[d])
            cover = list(dict.fromkeys([max(pool, key=lambda d: s[d][f"a{i}"]) for i in range(len(atoms))] + by_max))
            ranked = {"whole": sorted(pool, key=lambda d: -s[d]["whole"]), "max": by_max, "cover": cover}
            gold = set(case.gold)
            out = {"qid": case.qid, "atoms": atoms, "tokens": sum(t for _, t in graded),
                   "recall": {arm: {str(t): len(gold & set(r[:t])) / len(gold) for t in TOPS} for arm, r in ranked.items()}}
            print(f"{case.qid:>5} {len(atoms)} atoms  top8 " + "  ".join(f"{arm} {out['recall'][arm]['8']:.2f}" for arm in ranked), flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, dev))
    Path("out/atoms").mkdir(parents=True, exist_ok=True)
    Path("out/atoms/results.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{n} questions, {sum(len(r['atoms']) for r in res) / n:.1f} atoms, Jev tokens/q {sum(r['tokens'] for r in res) / n:,.0f}")
    for arm in res[0]["recall"]:
        print(f"{arm:6} gold recall " + "  ".join(f"top{t} {sum(r['recall'][arm][str(t)] for r in res) / n:.3f}" for t in TOPS))


if __name__ == "__main__":
    main()
