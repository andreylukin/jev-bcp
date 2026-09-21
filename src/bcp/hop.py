"""The card store (cards.py) as a retriever for the agent: `agent.EXTRA`-shaped, search_many(queries, k).

  card search  BM25 over the cards (subject + entities + facts): reach a page by what it says
  entity hop   pages that share a RARE card entity (on 2..RARE_MAX pages) with the query's top SEEDS text hits per
               index: the bridge hop. At most HOP pages per query, rarest entity first.

store2/store3 measured this on an 11k-doc testbed (gold in Jev's top 40: 63% -> 93%; frozen read 65.5% -> 77.5%).
Here: the same recall question over the FULL corpus, where a "rare" entity admits more pages. No Jev, no LLM:

    uv run python -m bcp.hop            # gold docs in the pool per dev question: text search vs + cards vs + hop

Result (full corpus, one round of 8 queries): every gold doc in the pool on 27.0% of questions -> 35.5% with cards -> 53.0%
with the hop (pool 156 -> 445 docs). In the agent: NULL, -0.1 [-3.5, +3.3] over two dev-200 runs (see store3.py).
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import bm25s

from bcp import cards as cardlib
from bcp.corpus import Corpus

SEEDS = 3
RARE_MAX = 10
HOP = 40
norm = lambda e: re.sub(r"\W+", " ", str(e).lower()).strip()
tok = lambda xs: bm25s.tokenize(xs, stopwords="en", show_progress=False)


class Hop:
    def __init__(self, corpus: Corpus, dense):
        self.corpus, self.dense = corpus, dense
        cards = cardlib.load()
        self.ids = [d for d in corpus.docids if cards.get(d)]
        self.bm = bm25s.BM25()
        self.bm.index(tok([" ".join([str(cards[d].get("subject", "")), *map(str, cards[d].get("entities") or []), *map(str, cards[d].get("facts") or [])]) for d in self.ids]), show_progress=False)
        self.ents = {d: [e for e in map(norm, cards[d].get("entities") or []) if len(e) > 3] for d in self.ids}
        docs = defaultdict(set)
        for d, es in self.ents.items():
            for e in es:
                docs[e].add(d)
        self.rare = {e: ds for e, ds in docs.items() if 2 <= len(ds) <= RARE_MAX}

    def cards(self, queries: list[str], k: int) -> list[list[str]]:
        return [[self.ids[i] for i in row] for row in self.bm.retrieve(tok(queries), k=k, show_progress=False)[0]]

    def hops(self, queries: list[str]) -> list[list[str]]:
        dense = self.dense.search_many(queries, SEEDS) if self.dense else [[] for _ in queries]
        out = []
        for q, dh in zip(queries, dense):
            seeds = self.corpus.search(q, SEEDS) + dh
            es = sorted({e for d in seeds for e in self.ents.get(d, []) if e in self.rare}, key=lambda e: len(self.rare[e]))
            out.append(list(dict.fromkeys(x for e in es for x in sorted(self.rare[e]) if x not in seeds))[:HOP])
        return out

    def search_many(self, queries: list[str], k: int) -> list[list[str]]:
        return [c + h for c, h in zip(self.cards(queries, k), self.hops(queries))]


def main():
    from bcp import dense as denselib
    from bcp.data import split

    corpus, dn = Corpus(), denselib.Dense()
    hop = Hop(corpus, dn)
    print(f"{len(hop.ids)} cards, {len(hop.rare)} rare entities", flush=True)
    queries = json.loads(Path("out/store/queries.json").read_text())
    hard = set(Path("out/bestofn/hard.txt").read_text().split())
    res = []
    for c in [c for c in split()[0] if c.gold and queries.get(c.qid)]:
        qs = queries[c.qid][:8]  # the agent writes 8 per round
        l0 = {d for q, dh in zip(qs, dn.search_many(qs, 15)) for d in corpus.search(q, 15) + dh}
        l1 = l0 | {d for row in hop.cards(qs, 15) for d in row}
        l2 = l1 | {d for row in hop.hops(qs) for d in row}
        gold = set(c.gold)
        res.append({"qid": c.qid, **{n: {"pool": len(p), "gold": len(gold & p) / len(gold), "all": gold <= p} for n, p in (("text", l0), ("+cards", l1), ("+hop", l2))}})
    Path("out/store/hop_recall.json").write_text(json.dumps(res, indent=1))
    for label, rs in (("all", res), ("hard core", [r for r in res if r["qid"] in hard])):
        print(f"{label}: {len(rs)} questions, one round of 8 queries")
        for n in ("text", "+cards", "+hop"):
            m = lambda k: sum(r[n][k] for r in rs) / len(rs)
            print(f"  {n:7} pool {m('pool'):4.0f}  gold docs in pool {m('gold'):.3f}  every gold doc {m('all'):.3f}")


if __name__ == "__main__":
    main()
