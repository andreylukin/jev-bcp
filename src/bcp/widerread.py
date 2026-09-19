"""The agent FINDS ~86-91% of gold documents but only 72-80% reach the reader, which sees the 12 best windows
(final.CHARS). Jev's top 8 holds 75% of gold docs, its top 20 holds 87% (rank.py). Flash reads long contexts cheaply,
so: does simply handing the reader MORE of Jev's ranking close the gap? Replay over the saved agent evidence
(out/jevlog_seeds): one read per question per arm, official judge.

    uv run python -m bcp.widerread --model deepseek/deepseek-v4-flash-0731 --n 60
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final
from bcp.corpus import Corpus
from bcp.data import split
from bcp.judge import judge

ARMS = (12, 24, 40, 64)  # windows of ~4000 chars handed to the reader, best Jev grade first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    agent.LLM_DEADLINE = 180
    dev = [c for c in split()[0] if (Path("out/jevlog_seeds") / f"{c.qid}.json").exists()][a.skip : a.skip + a.n]
    corpus = Corpus()

    with httpx.Client(timeout=300) as client:

        def one(case):
            seed = json.loads((Path("out/jevlog_seeds") / f"{case.qid}.json").read_text())
            ranked = sorted(seed["pool"], key=lambda w: -seed["pool"][w])
            docs = list(seed["scores"])
            owner = lambda w: next((d for d in docs if w[:200] in corpus.text[d]), None)
            gold = set(case.gold)
            out = {"qid": case.qid, "agent_ok": seed["agent_ok"], "pool": len(ranked)}
            for k in ARMS:
                top = ranked[:k]
                stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
                try:
                    r = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
                        f"Question: {case.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(top))}], stats)
                    ok = bool(judge(client, case.query, r.get("answer"), case.answer, str(r.get("notes", "")))["correct"])
                except Exception:
                    ok = False
                out[str(k)] = {"correct": ok, "gold": len(gold & {owner(w) for w in top}) / len(gold) if gold else None, "tokens": stats["llm_in"]}
            print(f"{case.qid:>5} agent {'OK' if out['agent_ok'] else '--'}  " + "  ".join(f"top{k} {'OK' if out[str(k)]['correct'] else '--'} gold {out[str(k)]['gold']:.2f}" for k in ARMS), flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, dev))
    Path("out/widerread").mkdir(parents=True, exist_ok=True)
    Path(f"out/widerread/results_{a.skip}.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{n} questions; the agent's own final answer: {sum(r['agent_ok'] for r in res) / n:.3f}")
    for k in ARMS:
        print(f"  one read of Jev's top {k:2}: right {sum(r[str(k)]['correct'] for r in res) / n:.3f}   gold docs in the input {sum(r[str(k)]['gold'] or 0 for r in res) / n:.3f}   {sum(r[str(k)]['tokens'] for r in res) / n:,.0f} input tokens")


if __name__ == "__main__":
    main()
