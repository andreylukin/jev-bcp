"""Can Jev replace the LLM in entity extraction? Jev cannot name things, so a regex PROPOSES names (capitalised
n-grams over the whole page, not the card writer's first 12k chars and 15-name cap) and Jev keeps the ones the page
is actually about: one request per page, the page in the state, one noul per candidate name.

Measured where the 8B cards were (store2's testbed: every dev gold / evidence doc, 20 hard negatives per question,
6000 distractors): the entity hop built from each entity source, same rarity rule, same seeds (3 text hits per
query per index). No reader, no ranking: gold docs in the pool, and what the pool costs to grade.

    uv run python -m bcp.jevents

Result (11,170 docs, 200 dev questions): gold docs in the hop pool 95.6% from the 8B cards, 93.6% from regex + Jev (pool 25%
larger), 96.9% from both. Jev keeps 36 of 57 candidate names per page at ~10k tokens per page: ~8x the cost of having a
cheap LLM write the cards. It works without a generator; on this corpus it is not worth it.
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

from bcp import dense as denselib
from bcp import jev
from bcp.corpus import Corpus
from bcp.data import split
from bcp.store2 import negatives

OUT = Path("out/store")
CHARS = 40_000  # page text in the state; ~12k tokens
MAX_CANDS = 80
KEEP = 0.5
WORD = r"(?:[A-Z][\w'’\-]*[a-z\d][\w'’\-]*|[A-Z]\.)"  # Capitalised, not ALL CAPS headings; initials
NAME = re.compile(rf"{WORD}(?:\s+(?:(?:of|de|la|le|van|von|der|del|di|da|du|al|bin|ibn|the|and)\s+)?(?:{WORD}|\d{{4}}))*")
LEAD = {"The", "A", "An", "In", "On", "At", "By", "For", "From", "He", "She", "It", "They", "We", "His", "Her", "This", "That", "These", "Those", "After",
        "Before", "During", "When", "While", "As", "But", "And", "If", "There", "However", "Although", "Since", "With", "One", "Its", "Their", "Our", "I"}
INSTR = "Is the name below a specific person, place, organisation, work or event that the document gives facts about?\n\nName: {}"
CRITERIA = {"true": "The document states at least one fact about this specific named person, place, organisation, work or event.",
            "false": "The name is a generic word, a sentence fragment, site navigation or boilerplate, or the document only mentions it in passing."}
norm = lambda e: re.sub(r"\W+", " ", str(e).lower()).strip()


def candidates(text: str) -> list[str]:
    """Names by frequency. A single capitalised word counts only where it is not the start of a sentence."""
    counts = Counter()
    for m in NAME.finditer(text[:CHARS]):
        words = m.group().split()
        while words and words[0] in LEAD:
            words = words[1:]
        if not words or (len(words) == 1 and (len(words[0]) < 4 or text[: m.start()].rstrip()[-1:] in ("", ".", "!", "?", ":", "\n"))):
            continue
        counts[" ".join(words)] += 1
    return [c for c, _ in counts.most_common(MAX_CANDS)]


def extract(client, text: str) -> tuple[dict[str, float], int]:
    cands = candidates(text)
    if not cands:
        return {}, 0
    body = {"model": jev.JEV_MODEL, "state": {"document": text[:CHARS]},
            "questions": {f"c{i}": {"type": "noul", "instructions": INSTR.format(c), "criteria": CRITERIA} for i, c in enumerate(cands)}}
    path = jev._cache_path(body)
    if path.exists():
        data = json.loads(path.read_text())
    else:
        try:
            data, _ = jev._post(client, body, os.environ["TYPESAFE_API_KEY"])
        except RuntimeError as e:
            if "max_tokens_exceeded" not in str(e):
                raise
            return extract(client, text[: len(text[:CHARS]) // 2])
        path.write_text(json.dumps(data))
    return {c: float(data["answers"][f"c{i}"]["noul"]) for i, c in enumerate(cands)}, int(data["usage"]["input_tokens"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="docs to extract (smoke test)")
    a = ap.parse_args()
    dev = [c for c in split()[0] if c.gold]
    corpus, dn = Corpus(), denselib.Dense()
    negs = negatives({c.qid for c in dev}, 20)
    labelled = sorted({d.docid for c in dev for d in c.docs} | {g for c in dev for g in c.gold} | {d for v in negs.values() for d in v})
    ids = labelled + random.Random(1).sample(sorted(set(corpus.docids) - set(labelled)), 6000)  # store2's testbed
    queries = json.loads((OUT / "queries.json").read_text())
    cards = json.loads((OUT / "cards.json").read_text())
    jev.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=300) as client:
        todo = ids[: a.limit or None]
        with ThreadPoolExecutor(16) as ex:
            got = list(ex.map(lambda d: extract(client, corpus.text[d]), todo))
    graded = {d: g for d, (g, _) in zip(todo, got)}
    tokens = sum(t for _, t in got)
    (OUT / "jev_ents.json").write_text(json.dumps(graded))
    kept = [sum(p >= KEEP for p in g.values()) for g in graded.values()]
    print(f"{len(todo)} docs, {tokens / 1e6:.1f}M Jev tokens (${tokens / 1e6 * 0.042:.2f}); candidates/doc {sum(map(len, graded.values())) / len(todo):.0f}, kept {sum(kept) / len(todo):.0f}", flush=True)
    if a.limit:
        d = todo[0]
        print(sorted(graded[d].items(), key=lambda x: -x[1])[:25], "\n8B:", cards.get(d, {}).get("entities"))
        return

    sources = {"8B cards": {d: [norm(e) for e in cards.get(d, {}).get("entities") or []] for d in ids},
               "Jev": {d: [norm(c) for c, p in graded[d].items() if p >= KEEP] for d in ids}}
    sources["8B + Jev"] = {d: sources["8B cards"][d] + sources["Jev"][d] for d in ids}
    tok = lambda xs: bm25s.tokenize(xs, stopwords="en", show_progress=False)
    bm = bm25s.BM25()
    bm.index(tok([corpus.text[d] for d in ids]), show_progress=False)
    row = {d: i for i, d in enumerate(dn.docids)}
    vec = dn.index[[row[d] for d in ids]]
    rare = {}
    for name, ents in sources.items():
        docs = defaultdict(set)
        for d, es in ents.items():
            for e in es:
                if len(e) > 3:
                    docs[e].add(d)
        rare[name] = {e: ds for e, ds in docs.items() if 2 <= len(ds) <= 10}
    hard = set(Path("out/bestofn/hard.txt").read_text().split())
    res = []
    for c in [c for c in dev if queries.get(c.qid)]:
        qs = queries[c.qid][:20]
        l0 = {ids[i] for r in bm.retrieve(tok(qs), k=3, show_progress=False)[0] for i in r} | {ids[i] for r in denselib.top_k(vec, denselib.embed(qs), 3) for i in r}
        gold = set(c.gold)
        out = {"qid": c.qid, "text": {"pool": len(l0), "gold": len(gold & l0) / len(gold), "all": gold <= l0}}
        for name, ents in sources.items():
            pool = l0 | {x for d in l0 for e in ents[d] if e in rare[name] for x in rare[name][e]}
            out[name] = {"pool": len(pool), "gold": len(gold & pool) / len(gold), "all": gold <= pool}
        res.append(out)
    (OUT / "jev_ents_recall.json").write_text(json.dumps(res, indent=1))
    for label, rs in (("all", res), ("hard core", [r for r in res if r["qid"] in hard])):
        print(f"{label}: {len(rs)} questions; pool = text hits + entity hop")
        for n in ("text", *sources):
            m = lambda k: sum(r[n][k] for r in rs) / len(rs)
            print(f"  {n:9} pool {m('pool'):5.0f}  gold docs in pool {m('gold'):.3f}  every gold doc {m('all'):.3f}")


if __name__ == "__main__":
    main()
