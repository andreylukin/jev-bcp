"""store.py's entity hop reached 92-98% of gold docs, but on a testbed of random distractors, and by pulling in
hundreds of docs. The harder test: add each question's HARD NEGATIVES (topically related pages) and more
distractors, then let Jev rank each pool and ask where the gold docs land - the number that predicts accuracy
(8B reader: 86% right with every gold doc in its input, 50% without).

  pools (3 hits per query per index):  L0 text search | L1 + card search | L1+L2 + entity hop
  metric: gold recall in Jev's top 12 / 20 / 40 of the pool

    uv run python -m bcp.store2 --writer mistralai/ministral-8b-2512 --n 100
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
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from bcp import agent, jev
from bcp import dense as denselib
from bcp.corpus import Corpus
from bcp.data import REPO, decrypt, split
from bcp.store import CARD, embed_docs

OUT = Path("out/store")
TOPS = (12, 20, 40)


def negatives(qids: set[str], n: int) -> dict[str, list[str]]:
    out = {}
    for i in range(6):
        path = hf_hub_download(REPO, f"data/test-{i:05d}-of-00006.parquet", repo_type="dataset")
        for r in pq.ParquetFile(path).read(columns=["query_id", "negative_docs"]).to_pylist():
            if r["query_id"] in qids:
                ids = [decrypt(d["docid"]) for d in r["negative_docs"]]
                out[r["query_id"]] = random.Random(r["query_id"]).sample(ids, min(n, len(ids)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--negatives", type=int, default=20)
    ap.add_argument("--distractors", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=48)
    a = ap.parse_args()
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dn = Corpus(), denselib.Dense()
    negs = negatives({c.qid for c in dev}, a.negatives)
    labelled = sorted({d.docid for c in dev for d in c.docs} | {g for c in dev for g in c.gold} | {d for v in negs.values() for d in v})
    ids = labelled + random.Random(1).sample(sorted(set(corpus.docids) - set(labelled)), a.distractors)
    queries = json.loads((OUT / "queries.json").read_text())
    print(f"testbed: {len(ids)} docs ({sum(len(v) for v in negs.values())} hard negatives), {len(dev)} questions", flush=True)
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
    agent.LLM_MODEL = a.writer

    with httpx.Client(timeout=300) as client:

        def card(d):
            try:
                return agent.llm(client, [{"role": "system", "content": CARD}, {"role": "user", "content": corpus.text[d][:12000]}], stats)
            except RuntimeError:
                return {}

        cards = json.loads((OUT / "cards.json").read_text())  # store.py's cards; only the new docs get written
        todo = [d for d in ids if d not in cards]
        print(f"{len(todo)} cards to write", flush=True)
        with ThreadPoolExecutor(a.threads) as ex:
            for i, (d, c) in enumerate(zip(todo, ex.map(card, todo))):
                cards[d] = c
                if i % 500 == 499:
                    (OUT / "cards.json").write_text(json.dumps(cards))
                    print(f"  {i + 1}/{len(todo)} cards", flush=True)
        (OUT / "cards.json").write_text(json.dumps(cards))

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
            out = {"qid": c.qid, "tokens": tokens}
            for name, pool in (("L0", l0), ("L1", l1), ("L1+L2", l2)):
                ranked = sorted(pool, key=lambda d: -scores[d])
                out[name] = {"pool": len(pool), "in_pool": len(gold & pool) / len(gold), **{str(t): len(gold & set(ranked[:t])) / len(gold) for t in TOPS}}
            print(f"{c.qid:>5} " + "  ".join(f"{k} pool {v['pool']:3} top12 {v['12']:.2f}" for k, v in out.items() if isinstance(v, dict)), flush=True)
            return out

        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, [c for c in dev if queries.get(c.qid)]))
    (OUT / "rank_results.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"\n{n} questions, {len(ids)} docs; Jev tokens/q {sum(r['tokens'] for r in res) / n:,.0f}")
    for name in ("L0", "L1", "L1+L2"):
        m = lambda k: sum(r[name][k] for r in res) / n
        print(f"  {name:6} pool {m('pool'):4.0f}  gold in pool {m('in_pool'):.3f}  | after Jev ranking: " + "  ".join(f"top{t} {m(str(t)):.3f}" for t in TOPS))


if __name__ == "__main__":
    main()
