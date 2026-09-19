"""Check 2: on the questions the full agent got wrong, re-read the best passages from the docs it had read -
once with Astra (thinking high), once with DeepSeek-flash as the control on the identical input."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx

from bcp import agent, jev
from bcp.corpus import Corpus
from bcp.data import load_all
from bcp.final import READER
from bcp.judge import judge

model, effort = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
agent.LLM_MODEL = model
agent.REASONING = {"effort": effort} if effort else {"enabled": False}
cases = {c.qid: c for c in load_all()}
rows = [r for r in json.load(open("out/modal_dev_passages/results.json"))["rows"] if "error" not in r and not r["correct"]]
corpus = Corpus()
with httpx.Client(timeout=300) as client:

    def one(r):
        q = cases[r["qid"]].query
        parts = [(f"{d}#{i}", w) for d in r["read"] for i, w in enumerate(corpus.windows(d))]
        scores, _, _ = jev.score_batched(client, q, parts)
        text = dict(parts)
        top = sorted(scores, key=lambda k: -scores[k])[:12]
        docs = "\n\n".join(f"[doc {k}]\n{text[k]}" for k in top)
        stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
        try:
            p = agent.llm(client, [{"role": "system", "content": READER}, {"role": "user", "content": f"Question: {q}\n\nDocuments:\n{docs}"}], stats)
            v = judge(client, q, p.get("answer"), cases[r["qid"]].answer, str(p.get("notes", "")))["correct"]
        except Exception as e:
            p, v = {"answer": None, "error": repr(e)[:120]}, False
        return {"qid": r["qid"], "gold_read": r["gold_read"], "was": str(r["answer"])[:60], "now": str(p.get("answer"))[:60], "gold": r["gold"][:60], "correct": bool(v), **stats}

    with ThreadPoolExecutor(12) as pool:
        res = list(pool.map(one, rows))
json.dump(res, open(f"out/escalate_{model.replace('/', '_')}.json", "w"), indent=1)
n = len(res)
fixed = sum(x["correct"] for x in res)
print(
    f"{model} {effort or 'off'}: fixed {fixed}/{n} of the agent's wrong answers | among those with every gold doc read: {sum(x['correct'] for x in res if x['gold_read'] == 1)}/{sum(1 for x in res if x['gold_read'] == 1)}"
    f" | cost ~${sum(x['llm_in'] for x in res) * (10e-6 if 'astra' in model else 0.06e-6) + sum(x['llm_out'] for x in res) * (50e-6 if 'astra' in model else 0.12e-6):.2f}"
)
