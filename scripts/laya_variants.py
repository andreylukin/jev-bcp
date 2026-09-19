"""Which framing does Laya need? Same labelled harness as laya_auc.py; each variant is one way of asking.
Laya's card: trained on short instructions naming state fields in backticks; families include search relevance
(acc 0.63), inference and fact checking (0.88), reading comprehension (0.85)."""

import json
import sys
import time

import httpx
import laya

from bcp import agent as A
from bcp import jev, metrics
from bcp.data import load_cases

n, sub, chunk = int(sys.argv[1]), (sys.argv[2] if sys.argv[2] != "-" else None), int(sys.argv[3])
m = laya.load("convaiinnovations/laya", subfolder=sub) if sub else laya.load("convaiinnovations/laya")
A.LLM_MODEL = "deepseek/deepseek-v4-flash-0731"
COMPILE = """Split the research question into the separate facts it relies on. Each fact must stand on its own: describe
intermediate people, places and works by what the question says about them, never as "the answer" or "this person".
3 to 6 short facts, each under 30 words. Reply with JSON: {"atoms": ["...", "..."]}"""


def noul(state, q):
    return m.system_one(state, {"q": q})["answers"]["q"]


V = {
    "jev_prompt": lambda qu, ch: noul({"question": qu, "document": ch}, {"type": "noul", "instructions": jev.INSTRUCTIONS})["noul"],
    "relevance": lambda qu, ch: noul({"query": qu, "passage": ch}, {"type": "noul", "instructions": "Is `passage` relevant to `query`?"})["noul"],
    "rc": lambda qu, ch: noul(
        {"question": qu, "passage": ch},
        {
            "type": "noul",
            "instructions": "Does `passage` contain information that helps answer `question`?",
            "criteria": {"true": "it states a fact the answer depends on", "false": "it is off topic or only loosely related"},
        },
    )["noul"],
    "choice": lambda qu, ch: noul(
        {"query": qu, "passage": ch},
        {
            "type": "choice",
            "instructions": "How does `passage` relate to `query`?",
            "criteria": {"relevant": "states facts that help answer the query", "not_relevant": "does not help answer the query"},
        },
    )["probabilities"]["relevant"],
}
rows = []
with httpx.Client(timeout=120) as client:
    for c in load_cases(n):
        t0 = time.perf_counter()
        stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
        atoms = [str(x) for x in A.llm(client, [{"role": "system", "content": COMPILE}, {"role": "user", "content": c.query}], stats).get("atoms", [])][:6]
        s = {k: {} for k in (*V, "facts_max", "facts_mean")}
        for d in c.docs:
            chunks = [d.text[:4000][i : i + chunk] for i in range(0, max(len(d.text[:4000]), 1), chunk)]
            for k, f in V.items():
                s[k][d.docid] = max(f(c.query, ch) for ch in chunks)
            per = [max(noul({"claim": a, "document": ch}, {"type": "noul", "instructions": "Does `document` support `claim`?"})["noul"] for ch in chunks) for a in atoms] or [0.0]
            s["facts_max"][d.docid], s["facts_mean"][d.docid] = max(per), sum(per) / len(per)
        y = [d.label for d in c.docs]
        j = jev.score(c, doc_chars=4000)
        r = {"qid": c.qid, "jev": metrics.auc(y, [j.scores[d.docid] for d in c.docs]), **{k: metrics.auc(y, [s[k][d.docid] for d in c.docs]) for k in s}}
        rows.append(r)
        print(f"{c.qid:>5} {time.perf_counter() - t0:4.0f}s  " + "  ".join(f"{k} {v:.2f}" for k, v in r.items() if k != "qid"), flush=True)
print(f"{len(rows)} questions, checkpoint {sub or 'english'}, chunk {chunk}")
for k in rows[0]:
    if k != "qid":
        print(f"  {k:11} AUC {sum(r[k] for r in rows) / len(rows):.3f}")
json.dump(rows, open(f"variants_{sub or 'english'}_{chunk}.json", "w"))
