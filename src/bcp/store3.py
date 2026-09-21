"""store2.py: the entity hop lifts gold docs in Jev's top 40 from 63% to 93% against hard negatives (hard core: every
gold doc in the top 40 on 47% -> 73% of questions). Every earlier evidence-delivery gain left accuracy flat
(claimsweep.py), so before paying for a corpus-wide card build: does the FROZEN reader answer better from the hop
pool? Same testbed, cards, queries and Jev grades as store2 (Jev's cache makes the grading free); one read of Jev's
top --top windows per arm, official judge.

    uv run python -m bcp.store3 --model deepseek/deepseek-v4-flash-0731

Result (dev 200, one read of the top 24): text search 65.5% -> with cards + hop 77.5% (+12 [+6.5, +18], fixed 31, broke 7).
BUT that baseline is one search from question-only queries. Inside the 8-round agent (modal_app.py --hop, hop.py) the same
store is a NULL: 80.0 / 82.5% vs 81.4% for the agent without it, at 1.75x the Jev tokens. The agent already hops by naming.
"""


import argparse
import json
import random
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import httpx

from bcp import agent, final, jev
from bcp import dense as denselib
from bcp.corpus import Corpus
from bcp.data import split
from bcp.judge import judge
from bcp.store import embed_docs
from bcp.store2 import negatives

OUT = Path("out/store")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--negatives", type=int, default=20)
    ap.add_argument("--distractors", type=int, default=6000)
    a = ap.parse_args()
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dn = Corpus(), denselib.Dense()
    negs = negatives({c.qid for c in dev}, a.negatives)
    labelled = sorted({d.docid for c in dev for d in c.docs} | {g for c in dev for g in c.gold} | {d for v in negs.values() for d in v})
    ids = labelled + random.Random(1).sample(sorted(set(corpus.docids) - set(labelled)), a.distractors)
    queries = json.loads((OUT / "queries.json").read_text())
    print(f"testbed: {len(ids)} docs ({sum(len(v) for v in negs.values())} hard negatives), {len(dev)} questions", flush=True)
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
    agent.LLM_MODEL = a.model
    agent.LLM_DEADLINE = 180

    with httpx.Client(timeout=300) as client:

        cards = json.loads((OUT / "cards.json").read_text())  # written by store2
        card_text = [" ".join([str(cards[d].get("subject", "")), *map(str, cards[d].get("entities") or []), *map(str, cards[d].get("facts") or [])]) or "empty" for d in ids]
        tok = lambda xs: bm25s.tokenize(xs, stopwords="en", show_progress=False)
        bm_text, bm_card = bm25s.BM25(), bm25s.BM25()
        bm_text.index(tok([corpus.text[d] for d in ids]), show_progress=False)
        bm_card.index(tok(card_text), show_progress=False)
        row = {d: i for i, d in enumerate(dn.docids)}
        vec_text = dn.index[[row[d] for d in ids]]
        vec_card = embed_docs([t[:6000] for t in card_text])
        norm = lambda e: re.sub(r"\W+", " ", str(e).lower()).strip()
        ent_docs = defaultdict(set)
        for d in ids:
            for e in cards[d].get("entities") or []:
                if len(norm(e)) > 3:
                    ent_docs[norm(e)].add(d)
        rare = {e: ds for e, ds in ent_docs.items() if 2 <= len(ds) <= 10}
        print("indexes done", flush=True)

        def one(c):
            qs = queries[c.qid][:20]
            qv = denselib.embed(qs)
            hits = lambda bm, vec: {ids[i] for r in bm.retrieve(tok(qs), k=3, show_progress=False)[0] for i in r} | {ids[i] for r in denselib.top_k(vec, qv, 3) for i in r}
            l0 = hits(bm_text, vec_text)
            l1 = l0 | hits(bm_card, vec_card)
            l2 = l1 | {x for d in l0 for e in cards[d].get("entities") or [] if norm(e) in rare for x in rare[norm(e)]}
            scores, _, tokens = jev.score_batched(client, c.query, [(d, corpus.window(d, c.query)) for d in sorted(l2)])
            gold = set(c.gold)
            out = {"qid": c.qid}
            for name, pool in (("L0", l0), ("L1+L2", l2)):
                top = sorted(pool, key=lambda d: -scores[d])[: a.top]
                try:
                    r = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
                        f"Question: {c.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{corpus.window(d, c.query)}" for i, d in enumerate(top))}], stats)
                    ok = bool(judge(client, c.query, r.get("answer"), c.answer, str(r.get("notes", "")))["correct"])
                except Exception:
                    ok = False
                out[name] = {"correct": ok, "gold": len(gold & set(top)) / len(gold)}
            print(f"{c.qid:>5} " + "  ".join(f"{k} {'OK' if v['correct'] else '--'} gold {v['gold']:.2f}" for k, v in out.items() if isinstance(v, dict)), flush=True)
            return out

        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, [c for c in dev if queries.get(c.qid)]))
    (OUT / "read_results.json").write_text(json.dumps(res, indent=1))
    hard = {x.strip() for x in Path("out/bestofn/hard.txt").read_text().split()}
    for label, rs in (("all", res), ("hard core", [r for r in res if r["qid"] in hard])):
        print(f"{label}: {len(rs)} questions")
        for name in ("L0", "L1+L2"):
            print(f"  {name:6} right {sum(r[name]['correct'] for r in rs) / len(rs):.3f}   gold docs in the input {sum(r[name]['gold'] for r in rs) / len(rs):.3f}")
    both = lambda x, y: sum(r["L1+L2"]["correct"] == x and r["L0"]["correct"] == y for r in res)
    print(f"  hop fixed {both(True, False)}, broke {both(False, True)}")

if __name__ == "__main__":
    main()
