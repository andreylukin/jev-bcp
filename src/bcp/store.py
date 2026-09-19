"""Would a knowledge store built offline from the documents alone make missed gold docs reachable? Three layers,
one cheap testbed. Labels are used ONLY to pick the testbed docs and to score; the store never sees a question.

  testbed   every gold + evidence doc of the dev questions, plus --distractors random docs. Search runs inside this
            subset, so hits per query are kept tiny (k=1,3: roughly top-25 and top-75 in the full 100k corpus)
  queries   20 per dev question from the writer prompt (recall.py), written from the question alone
  cards     the SMALL model writes one card per doc: subject, entities, dates/places, standalone facts

  L0  BM25(text) + dense(text)                                  today's search
  L1  L0 + BM25(cards) + dense(cards)                            reach a page by what it says
  L2  L0 + entity hop: docs sharing a rare card entity with L0's hits      the bridge hop
  L3  Jev facets per doc (page type, era, region); the LLM maps each question to the facet values a relevant
      page could have. Reported as selectivity: share of gold docs kept vs share of all docs kept.
      Also: Jev validates a sample of card facts against the source text.

    uv run python -m bcp.store --writer mistralai/ministral-8b-2512 --model deepseek/deepseek-v4-flash-0731
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import bm25s
import httpx
import numpy as np

from bcp import agent, jev, recall
from bcp import dense as denselib
from bcp.corpus import Corpus
from bcp.data import split

KS = (1, 3)
CARD = """Write an index card for this web page so that a researcher can find it. Use only what the page states.
Reply with JSON: {"subject": "who or what the page is mainly about, one line",
 "entities": ["up to 15 proper names that matter on this page: people, places, organisations, works, events"],
 "facts": ["5 to 10 standalone facts, each a full sentence naming its entities, with dates, places and numbers"]}"""
FACETS = {
    "page_type": {"person": "a biography, profile, obituary or interview about one person", "organisation": "a company, institution, club, team or band",
                  "place": "a town, region, building, landmark or natural feature", "work": "a book, film, song, album, game, artwork or TV show",
                  "event": "a match, competition, election, disaster, conflict or other dated event", "research": "a scientific or scholarly paper, thesis or technical report",
                  "other": "a list, directory, product page, forum thread or anything else"},
    "era": {"before_1900": "mainly about things before 1900", "1900_1949": "mainly 1900 to 1949", "1950_1989": "mainly 1950 to 1989",
            "1990_2009": "mainly 1990 to 2009", "2010_on": "mainly 2010 or later", "timeless": "not tied to a period"},
    "region": {"north_america": "USA or Canada", "latin_america": "Mexico, Central or South America, Caribbean", "europe": "Europe including Russia",
               "africa": "Africa", "middle_east": "Middle East and North Africa", "south_asia": "India, Pakistan, Bangladesh, Sri Lanka, Nepal",
               "east_asia": "East and South-East Asia", "oceania": "Australia, New Zealand, Pacific", "none": "global or no particular region"},
}
MAP = """A research question needs several web pages to answer. For each facet, list EVERY value a relevant page could have
(pages about intermediate people, places or works count). When unsure include the value. Facets and values:
""" + json.dumps({k: list(v) for k, v in FACETS.items()}) + """
Reply with JSON: {"page_type": [...], "era": [...], "region": [...]}"""


def cached(path: Path, build):
    if path.exists():
        return json.loads(path.read_text())
    data = build()
    path.write_text(json.dumps(data))
    return data


def embed_docs(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 64):
        for attempt in range(5):
            r = httpx.post(denselib.URL, headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                           json={"model": denselib.MODEL, "input": texts[i : i + 64], "provider": {"only": ["nebius"]}}, timeout=120)
            if r.status_code == 200:
                out += [d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"])]
                break
        else:
            raise RuntimeError(f"embed: {r.status_code} {r.text[:200]}")
    v = np.array(out, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True, help="the small model that writes the cards")
    ap.add_argument("--model", required=True, help="model for the question side (queries, facet mapping)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--distractors", type=int, default=3000)
    ap.add_argument("--out", default="out/store")
    a = ap.parse_args()
    OUT = Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    dev = [c for c in split()[0] if c.gold][: a.n]
    corpus, dn = Corpus(), denselib.Dense()
    labelled = sorted({d.docid for c in dev for d in c.docs} | {g for c in dev for g in c.gold})
    rng = random.Random(0)
    ids = labelled + rng.sample(sorted(set(corpus.docids) - set(labelled)), a.distractors)
    print(f"testbed: {len(ids)} docs ({len(labelled)} gold/evidence), {len(dev)} questions", flush=True)
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)

    with httpx.Client(timeout=300) as client:

        def ask(model, messages, schema=None):
            agent.LLM_MODEL = model  # calls below run one model at a time
            try:
                return agent.llm(client, messages, stats, schema)
            except RuntimeError:
                return {}

        with ThreadPoolExecutor(24) as ex:
            queries = cached(OUT / "queries.json", lambda: dict(zip([c.qid for c in dev], ex.map(lambda c: [
                q for q in ask(a.model, [{"role": "system", "content": recall.PROMPT.format(n=20)}, {"role": "user", "content": f"Question: {c.query}\n\nEarlier queries: []\n\nTop snippets so far:\n(none yet)"}], recall.schema(20)).get("queries", []) if isinstance(q, str)], dev))))
            print("queries done", flush=True)
            cards = cached(OUT / "cards.json", lambda: dict(zip(ids, ex.map(lambda d: ask(a.writer, [{"role": "system", "content": CARD}, {"role": "user", "content": corpus.text[d][:12000]}]), ids))))
            print(f"cards done: {sum(bool(c.get('facts')) for c in cards.values())}/{len(ids)} usable, LLM tokens in {stats['llm_in']:,} out {stats['llm_out']:,}", flush=True)
            fmap = cached(OUT / "facetmap.json", lambda: dict(zip([c.qid for c in dev], ex.map(lambda c: ask(a.model, [{"role": "system", "content": MAP}, {"role": "user", "content": c.query}]), dev))))

        card_text = [" ".join([str(cards[d].get("subject", "")), *map(str, cards[d].get("entities") or []), *map(str, cards[d].get("facts") or [])]) or "empty" for d in ids]
        tok = lambda xs: bm25s.tokenize(xs, stopwords="en", show_progress=False)
        bm_text, bm_card = bm25s.BM25(), bm25s.BM25()
        bm_text.index(tok([corpus.text[d] for d in ids]), show_progress=False)
        bm_card.index(tok(card_text), show_progress=False)
        row = {d: i for i, d in enumerate(dn.docids)}
        vec_text = dn.index[[row[d] for d in ids]]
        if (OUT / "card_vecs.npy").exists():
            vec_card = np.load(OUT / "card_vecs.npy")
        else:
            vec_card = embed_docs([t[:6000] for t in card_text])
            np.save(OUT / "card_vecs.npy", vec_card)
        print("indexes done", flush=True)

        # L2: entity -> docs; an entity is a bridge only if it is rare
        norm = lambda e: re.sub(r"\W+", " ", str(e).lower()).strip()
        ent_docs = defaultdict(set)
        for d in ids:
            for e in cards[d].get("entities") or []:
                if len(norm(e)) > 3:
                    ent_docs[norm(e)].add(d)
        rare = {e: ds for e, ds in ent_docs.items() if 2 <= len(ds) <= 10}

        # L3: Jev facets on each doc's head
        def facet(d):
            body = {"model": jev.JEV_MODEL, "state": {"page": corpus.text[d][:4000]},
                    "questions": {k: {"type": "choice", "instructions": f"Which {k.replace('_', ' ')} fits the page best?", "criteria": v} for k, v in FACETS.items()}}
            data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
            return {k: data["answers"][k]["choice"] for k in FACETS}

        with ThreadPoolExecutor(12) as ex:
            facets = cached(OUT / "facets.json", lambda: dict(zip(ids, ex.map(facet, ids))))
        print("facets done", flush=True)

        def valid(d):  # Jev checks each card fact against the source
            facts = [str(f) for f in (cards[d].get("facts") or [])][:10]
            if not facts:
                return []
            body = {"model": jev.JEV_MODEL, "state": {"page": corpus.text[d][:12000]},
                    "questions": {f"f{i}": {"type": "noul", "instructions": f"Does the page state this fact?\n\nFact: {f}"} for i, f in enumerate(facts)}}
            data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
            return [data["answers"][f"f{i}"]["noul"] for i in range(len(facts))]

        with ThreadPoolExecutor(12) as ex:
            checked = cached(OUT / "valid.json", lambda: list(ex.map(valid, rng.sample(ids, min(300, len(ids))))))

        res = []
        for c in dev:
            qs = queries[c.qid][:20]
            if not qs:
                continue
            qv = denselib.embed(qs)
            gold = set(c.gold)
            out = {"qid": c.qid}
            for k in KS:
                hits = lambda bm, vec: {ids[i] for r in bm.retrieve(tok(qs), k=k, show_progress=False)[0] for i in r} | {ids[i] for r in denselib.top_k(vec, qv, k) for i in r}
                l0 = hits(bm_text, vec_text)
                l1 = l0 | hits(bm_card, vec_card)
                hop = {x for d in l0 for e in cards[d].get("entities") or [] if norm(e) in rare for x in rare[norm(e)]}
                l2 = l0 | hop
                for name, pool in (("L0", l0), ("L1", l1), ("L2", l2), ("L1+L2", l1 | hop)):
                    out[f"{name}@{k}"] = (len(gold & pool) / len(gold), len(pool))
            wide = {ids[i] for r in bm_text.retrieve(tok(qs), k=6, show_progress=False)[0] for i in r} | {ids[i] for r in denselib.top_k(vec_text, qv, 6) for i in r}
            out["L0@6"] = (len(gold & wide) / len(gold), len(wide))  # today's search at a pool as large as the layered ones
            m = fmap.get(c.qid) or {}
            keep = lambda d: all(facets[d][f] in (m.get(f) or list(FACETS[f])) for f in FACETS)
            out["facet_gold"] = sum(keep(d) for d in gold) / len(gold)
            out["facet_all"] = sum(keep(d) for d in ids) / len(ids)
            res.append(out)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"\n{n} questions. gold recall (pool size), searching {len(ids)} docs:")
    for key in [f"{name}@{k}" for k in KS for name in ("L0", "L1", "L2", "L1+L2")] + ["L0@6"]:
        print(f"  {key:8} {sum(r[key][0] for r in res) / n:.3f}  ({sum(r[key][1] for r in res) / n:.0f} docs)")
    print(f"L3 facets: keep {sum(r['facet_gold'] for r in res) / n:.3f} of gold docs while keeping {sum(r['facet_all'] for r in res) / n:.3f} of all docs")
    flat = [v for c in checked for v in c]
    print(f"card facts Jev confirms against the page (p>=0.5): {sum(v >= 0.5 for v in flat) / max(1, len(flat)):.3f} of {len(flat)}; page types: {Counter(f['page_type'] for f in facets.values()).most_common(4)}")


if __name__ == "__main__":
    main()
