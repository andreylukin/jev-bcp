"""Two reads disagreeing flags 19% of questions holding 67% of the wrong answers (agree.py), and those wrong answers
are mostly missing evidence: gold retrieved 0.49 vs 0.92 for right answers. So on disagreement, search WIDER
(recall.py's query writer: many diverse queries, Jev grades every hit), then read twice again; the new answer
replaces the old one only if the two new reads agree. Questions where the first two reads agreed are untouched.

    uv run python -m bcp.widen --model deepseek/deepseek-v4-flash-0731
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev, recall
from bcp.agree import SAME
from bcp.corpus import Corpus
from bcp.data import load_all
from bcp.dense import Dense
from bcp.judge import judge

ROUNDS, QUERIES, HITS = 2, 20, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dev_base_v3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    cases = {c.qid: c for c in load_all()}
    rows = {r["qid"]: r for r in json.load(open(Path(a.run) / "results.json"))["rows"] if "error" not in r}
    flagged = [x for x in json.load(open("out/agree/results.json")) if "error" not in x and x["same"] < 0.5]
    corpus, dense = Corpus(), Dense()

    with httpx.Client(timeout=300) as client:

        def top(pool):
            out, size = [], 0
            for w in sorted(pool, key=lambda w: -pool[w][0]):
                if size + len(w) > final.CHARS:
                    break
                out.append(w)
                size += len(w)
            return out

        def one(x):
            r, case = rows[x["qid"]], cases[x["qid"]]
            q, gold = case.query, set(case.gold)
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0)
            ask = lambda m, schema=None: agent.llm(client, m, stats, schema)
            parts = [(f"{d}#{i}", w) for d in r["read"] for i, w in enumerate(corpus.windows(d))]
            s, _, _ = jev.score_batched(client, q, parts)
            pool = {w: (s[k], k.split("#")[0]) for k, w in parts}  # window -> (score, docid)
            seen = {d: 0.0 for d in r["read"]}
            cover = lambda: len(gold & {pool[w][1] for w in top(pool)}) / len(gold)
            out = {"qid": x["qid"], "A_ok": x["A_ok"], "gold_before": cover()}
            try:
                tried: list[str] = []
                for _ in range(ROUNDS):
                    snippets = "\n".join(f"- {w[:500]}" for w in top(pool)[:10])
                    qs = ask([{"role": "system", "content": recall.PROMPT.format(n=QUERIES)}, {"role": "user", "content":
                        f"Question: {q}\n\nEarlier queries: {json.dumps(tried)}\n\nTop snippets so far:\n{snippets}"}], recall.schema(QUERIES)).get("queries", [])
                    qs = [y for y in qs if isinstance(y, str) and y.strip() and y not in tried]
                    tried += qs
                    found = {}
                    for y, dh in zip(qs, dense.search_many(qs, HITS)):
                        for d in corpus.search(y, HITS) + dh:
                            if d not in seen and d not in found:
                                found[d] = y
                    if found:
                        sc, best, graded = agent.grade(client, corpus, q, found, stats, agent.DEEP)
                        seen.update(sc)
                        for d in found:  # a window's doc: its best window, or any deep-graded window of the deep docs
                            pool[best[d]] = (sc[d], d)
                        for d in sorted(found, key=lambda d: -sc[d])[: agent.DEEP]:
                            for w in corpus.windows(d):
                                if w in graded:
                                    pool[w] = (graded[w], d)
                out["gold_after"], out["new_docs"] = cover(), len(seen) - len(r["read"])
                docs = top(pool)
                show = lambda ds: "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(ds))
                c1, c2 = (ask([{"role": "system", "content": final.READER}, {"role": "user", "content": f"Question: {q}\n\nDocuments:\n{show(ds)}"}]) for ds in (docs, docs[::-1]))
                C, C2 = str(c1.get("answer")), str(c2.get("answer"))
                data, _ = jev._post(client, {"model": jev.JEV_MODEL, "state": {"question": q, "answer_a": C, "answer_b": C2},
                                             "questions": {"same": {"type": "noul", "instructions": SAME}}}, os.environ["TYPESAFE_API_KEY"])
                out.update(C=C, same=data["answers"]["same"]["noul"], C_ok=bool(judge(client, q, C, case.answer, str(c1.get("notes", "")))["correct"]), jev_tokens=stats["jev_tokens"], llm_calls=stats["llm_calls"])
            except Exception as e:
                out["error"] = repr(e)[:200]
            print(f"{x['qid']:>5} A {'OK' if x['A_ok'] else '--'}  gold in reader input {out['gold_before']:.2f} -> {out.get('gold_after', -1):.2f}  "
                  f"C {'OK' if out.get('C_ok') else '--'} (new reads agree {out.get('same', -1):.2f})", flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, flagged))
    Path("out/widen").mkdir(parents=True, exist_ok=True)
    Path("out/widen/results.json").write_text(json.dumps(res, indent=1))
    ok = [x for x in res if "error" not in x]
    n = len(ok)
    m = lambda k: sum(x[k] for x in ok) / n
    print(f"{n} flagged questions ({len(res) - n} errors): gold docs in reader input {m('gold_before'):.3f} -> {m('gold_after'):.3f}; {m('new_docs'):.0f} new docs, {m('jev_tokens'):,.0f} Jev tokens, {m('llm_calls'):.1f} LLM calls each")
    print(f"first answer right {sum(x['A_ok'] for x in ok)}/{n}   new answer right {sum(x['C_ok'] for x in ok)}/{n}")
    take = [x for x in ok if x["same"] >= 0.5]
    print(f"policy: take the new answer only when its two reads agree ({len(take)} of {n}): right {sum(x['C_ok'] for x in take) + sum(x['A_ok'] for x in ok if x['same'] < 0.5)}/{n}"
          f"   (new answers taken: {sum(x['C_ok'] for x in take)}/{len(take)} right)")


if __name__ == "__main__":
    main()
