"""Free check (no Jev, no LLM): do gold docs surface under reciprocal-rank fusion of the deep rankings for the question
and every unbound claim? A bridging page tends to rank for SEVERAL claims even if it tops none."""

import json
import sys
from collections import defaultdict
from pathlib import Path

from bcp import dense as D
from bcp import jevlog
from bcp.corpus import Corpus
from bcp.data import split

runs = {r["qid"]: r for d in ("out/jevlog_v35", "out/jevlog_v35_fresh") for r in map(json.loads, open(d + "/rows.jsonl")) if "error" not in r and r.get("bindings")}
hard = set("219 244 362 427 523 543 688 713 758 843 872 873 1141 1230".split())
dev = [c for c in split()[0] if c.qid in runs and c.gold][: int(sys.argv[1])]
dev += [c for c in split()[0] if c.qid in hard and c not in dev]
corpus = Corpus()
dn = D.Dense()
K = 200
res = []
for c in dev:
    prog = jevlog.parse(runs[c.qid]["program"])
    unb = {v: d for v, (_, d) in prog.vars.items()}
    qs = [c.query] + [jevlog.claim(t, unb) for _, _, t in prog.facts]
    rrf = defaultdict(float)
    for q, hits in zip(qs, dn.search_many(qs, K)):
        for r, d in enumerate(hits):
            rrf[d] += 1 / (60 + r)
        for r, d in enumerate(corpus.search(q, K)):
            rrf[d] += 1 / (60 + r)
    order = sorted(rrf, key=lambda d: -rrf[d])
    gold = set(c.gold)
    seed = json.loads((Path("out/jevlog_seeds") / f"{c.qid}.json").read_text())
    seed_docs = set(seed["scores"])
    res.append(
        dict(
            qid=c.qid,
            hard=c.qid in hard,
            agent_found=len(gold & seed_docs) / len(gold),
            **{f"rrf{k}": len(gold & set(order[:k])) / len(gold) for k in (20, 40, 100, 400)},
            union20=len(gold & (seed_docs | set(order[:20]))) / len(gold),
        )
    )
for name, rows in (("all", res), ("14 hard", [r for r in res if r["hard"]]), ("rest", [r for r in res if not r["hard"]])):
    m = lambda k: sum(r[k] for r in rows) / len(rows)
    print(
        f"{name:8} n={len(rows):3}  gold found by the agent {m('agent_found'):.3f} | gold in RRF top20 {m('rrf20'):.3f} top40 {m('rrf40'):.3f} top100 {m('rrf100'):.3f} top400 {m('rrf400'):.3f} | agent pool + RRF top20 {m('union20'):.3f}"
    )
