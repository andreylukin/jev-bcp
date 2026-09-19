"""Retrieval only: can a small LOCAL model whose only job is to write queries, plus hybrid search and Jev
grading, put every gold document in front of the reader? No reader, no judge - the metric is gold recall.

  round 1  the model sees the question and writes N diverse queries
  search   BM25 + dense for every query; Jev grades every hit (batched)
  round 2+ the model sees the top-graded snippets (names it could not know before) and writes N more

    uv run python -m bcp.recall --ollama gemma4:12b-mlx
    uv run python -m bcp.recall --model mistralai/ministral-8b-2512        # same thing through OpenRouter
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, jev
from bcp.corpus import Corpus
from bcp.data import split
from bcp.dense import Dense

PROMPT = """You write search queries for a hard research question. Your only job is queries - do not answer.
The question describes something indirectly through several constraints. Write {n} DIFFERENT queries:
some short keyword queries (names, places, years, titles), some descriptive sentences, at least one per constraint,
and several that combine two constraints. If snippets are given, use the specific names, titles and dates in them to
write follow-up queries that chase the entities they mention. Do not repeat earlier queries.
Reply with JSON: {{"queries": ["...", "..."]}}"""


def schema(n: int) -> dict:
    """Exactly-n-ish queries: Granite 8B wrote 36 of the 60 asked for when the count was only in the prompt."""
    return {"type": "object", "properties": {"queries": {"type": "array", "items": {"type": "string", "maxLength": 200}, "minItems": n, "maxItems": n}}, "required": ["queries"], "additionalProperties": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--ollama", help="local model name, served by Ollama")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--queries", type=int, default=25)
    ap.add_argument("--hits", type=int, default=30)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--endpoint", help="self-hosted OpenAI-compatible chat URL (modal_llm.py)")
    a = ap.parse_args()
    if a.ollama:
        agent.LLM_URL, agent.LLM_MODEL = "http://localhost:11434/v1/chat/completions", a.ollama
    elif a.endpoint:
        agent.use_endpoint(a.endpoint)
    else:
        agent.LLM_MODEL = a.model
    dev = split()[0][: a.n]
    corpus, dense = Corpus(), Dense()
    tag = (a.ollama or a.model or "slm").replace("/", "_").replace(":", "_")

    with httpx.Client(timeout=300) as client:

        def one(case):
            t0 = time.perf_counter()
            stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            scores: dict[str, float] = {}
            tried: list[str] = []
            tokens = 0
            by_round = []
            gold = set(case.gold)
            for rnd in range(a.rounds):
                top = sorted(scores, key=lambda d: -scores[d])[:10]
                snippets = "\n".join(f"- {corpus.window(d, case.query)[:500]}" for d in top)
                user = f"Question: {case.query}\n\nEarlier queries: {json.dumps(tried[-40:])}\n\nTop snippets so far:\n{snippets or '(none yet)'}"
                try:
                    qs = agent.llm(client, [{"role": "system", "content": PROMPT.format(n=a.queries)}, {"role": "user", "content": user}], stats, schema(a.queries)).get("queries", [])
                except Exception:
                    qs = []
                qs = [q for q in qs if isinstance(q, str) and q.strip() and q not in tried][: a.queries]
                tried += qs
                found = {}
                if qs:
                    for q, dh in zip(qs, dense.search_many(qs, a.hits)):
                        for d in corpus.search(q, a.hits) + dh:
                            if d not in scores and d not in found:
                                found[d] = corpus.window(d, q + " " + case.query)
                if found:
                    s, _, t = jev.score_batched(client, case.query, list(found.items()))
                    scores.update(s)
                    tokens += t
                ranked = sorted(scores, key=lambda d: -scores[d])
                by_round.append({"pool": len(scores), "any": bool(gold & set(scores)), "all": gold <= set(scores),
                                 **{f"top{k}": len(gold & set(ranked[:k])) / len(gold) for k in (10, 20, 50)}})
            print(f"{case.qid:>5} {len(tried):3} queries  pool {len(scores):4}  gold any={by_round[-1]['any']!s:5} all={by_round[-1]['all']!s:5} top10={by_round[-1]['top10']:.2f}", flush=True)
            return {"qid": case.qid, "rounds": by_round, "queries": len(tried), "jev_tokens": tokens, "secs": time.perf_counter() - t0, **stats}

        with ThreadPoolExecutor(a.workers) as pool:
            res = list(pool.map(one, dev))
    Path("out/recall").mkdir(parents=True, exist_ok=True)
    Path(f"out/recall/{tag}.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{tag}: {n} questions, {sum(r['queries'] for r in res) / n:.0f} queries, pool {sum(r['rounds'][-1]['pool'] for r in res) / n:.0f} docs, "
          f"{sum(r['secs'] for r in res) / n:.0f}s, Jev ${sum(r['jev_tokens'] for r in res) / n * 0.042e-6:.3f}/q")
    for i in range(a.rounds):
        m = lambda k: sum(r["rounds"][i][k] for r in res) / n
        print(f"  after round {i + 1}: gold in pool any {m('any'):.3f} all {m('all'):.3f} | gold recall in Jev top10 {m('top10'):.3f} top20 {m('top20'):.3f} top50 {m('top50'):.3f}")


if __name__ == "__main__":
    main()
