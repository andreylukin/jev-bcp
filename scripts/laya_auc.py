"""Laya in Jev's seat on the labelled harness: evidence docs vs hard negatives, AUC per question (Jev 0.94, BM25 0.50).
Laya's budget is 512 tokens for instructions + question + document (1024 for typed-decisions), so the 4000-char
window Jev grades whole is cut into chunks; doc score = max over chunks."""

import json
import sys
import time

import laya

from bcp import jev, metrics
from bcp.data import load_cases

n, sub, chunk = int(sys.argv[1]), (sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None), int(sys.argv[3]) if len(sys.argv) > 3 else 700
agent = laya.load("convaiinnovations/laya", subfolder=sub) if sub else laya.load("convaiinnovations/laya")
print("device", agent.device, "chunk", chunk, flush=True)
Q = {"q": {"type": "noul", "instructions": jev.INSTRUCTIONS}}
rows = []
for c in load_cases(n):
    t0 = time.perf_counter()
    calls = 0
    scores = {}
    for d in c.docs:
        text = d.text[:4000]
        best = 0.0
        for i in range(0, max(len(text), 1), chunk):
            best = max(best, agent.system_one({"question": c.query, "document": text[i : i + chunk]}, Q)["answers"]["q"]["noul"])
            calls += 1
        scores[d.docid] = best
    y = [d.label for d in c.docs]
    s = [scores[d.docid] for d in c.docs]
    j = jev.score(c, doc_chars=4000)
    rows.append(
        {
            "qid": c.qid,
            "laya": metrics.auc(y, s),
            "jev": metrics.auc(y, [j.scores[d.docid] for d in c.docs]),
            "calls": calls,
            "secs": time.perf_counter() - t0,
            "qtok": len(agent.tok(c.query)["input_ids"]),
        }
    )
    print(f"{c.qid:>5} laya {rows[-1]['laya']:.3f}  jev {rows[-1]['jev']:.3f}  {calls} calls {rows[-1]['secs']:.0f}s  question {rows[-1]['qtok']} tok", flush=True)
m = lambda k: sum(r[k] for r in rows) / len(rows)
print(f"{len(rows)} questions: Laya AUC {m('laya'):.3f}   Jev AUC {m('jev'):.3f}   {m('calls'):.0f} Laya calls/q, {m('secs'):.0f}s/q, question {m('qtok'):.0f} tokens")
json.dump(rows, open(f"laya_auc_{sub or 'english'}_{chunk}.json", "w"))
