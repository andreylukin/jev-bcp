"""30 of the agent's 39 wrong dev answers used all 8 search rounds without ever feeling confident. Is the search budget
the limit? Rerun the same questions with 8 rounds (control: run-to-run noise) and with 16."""

import json
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx

from bcp import agent
from bcp.corpus import Corpus
from bcp.data import split
from bcp.dense import Dense

rounds = int(sys.argv[1])
agent.LLM_MODEL = "deepseek/deepseek-v4-flash-0731"
agent.ROUNDS = rounds
agent.DEADLINE = 900
rows = [r for r in json.load(open("out/dev_base_v3/results.json"))["rows"] if "error" not in r and r["rounds"] == 8]
pick = [r["qid"] for r in rows if not r["correct"]] + [r["qid"] for r in rows if r["correct"]][:10]
cases = [c for c in split()[0] if c.qid in set(pick)]
corpus, dense = Corpus(), Dense()
with httpx.Client(timeout=300) as client, ThreadPoolExecutor(8) as ex:
    res = list(ex.map(lambda c: agent.run_one(c, corpus, client, True, dense, "hybrid"), cases))
json.dump(res, open(f"out/rounds_{rounds}.json", "w"))
was = {r["qid"]: bool(r["correct"]) for r in rows}
ok = [r for r in res if "error" not in r]
print(
    f"ROUNDS={rounds}: {len(res)} questions; right before {sum(was[r['qid']] for r in res)}; right now {sum(bool(r['correct']) for r in res)}; errors {len(res) - len(ok)}; mean rounds {sum(r['rounds'] for r in ok) / len(ok):.1f}; gold retrieved {sum(r['gold_retrieved'] or 0 for r in ok) / len(ok):.2f}; gold read {sum(r['gold_read'] or 0 for r in ok) / len(ok):.2f}; secs {sum(r['secs'] for r in ok) / len(ok):.0f}"
)
