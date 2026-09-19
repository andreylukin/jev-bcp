"""The pipeline shaped for a SMALL model: every LLM call is one small job. As the whole agent (queries + notes +
reading in one JSON reply) Ministral 8B read 34% of gold docs and scored ~40%; with writing queries as its ONLY
job it put a gold doc in the pool for 94% of questions (recall.py), and as a reader of good evidence it scores
84-86% (oracle.py). Jev does everything between the two.

  round:  writer -> N diverse queries (schema-constrained)      small job 1
          BM25 + dense; Jev grades every hit and picks windows
          reader -> answers from the best windows so far          small job 2
          Jev strict check -> pass: stop | fail: next round (the writer sees the best snippets)
  end:    best-checked answer; a second read in reversed order gives agree/disagree as the confidence

`agent.SPLIT` / `modal run ... --roles` switches it on.
"""

import json
import os
import time

import httpx

from bcp import agent, final, jev, recall
from bcp.agree import SAME
from bcp.corpus import Corpus
from bcp.data import Case

ROUNDS = 3
QUERIES = 20


def solve(case: Case, corpus: Corpus, client: httpx.Client, dense=None, bm25: bool = True) -> dict:
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0, search_secs=0.0, switched=0, final_secs=0.0, final_attempts=0, final_score=0.0)
    t_start = time.perf_counter()
    q = case.query
    seen: dict[str, float] = {}
    pool: dict[str, tuple[float, str]] = {}  # window -> (Jev score, docid)
    tried: list[str] = []
    attempts: list[tuple[str, str, float]] = []
    docs: list[str] = []

    for rnd in range(ROUNDS):
        top10 = sorted(pool, key=lambda w: -pool[w][0])[:10]
        snippets = "\n".join(f"- {w[:500]}" for w in top10)
        try:
            qs = agent.llm(client, [{"role": "system", "content": recall.PROMPT.format(n=QUERIES)}, {"role": "user", "content":
                f"Question: {q}\n\nEarlier queries: {json.dumps(tried[-40:])}\n\nTop snippets so far:\n{snippets or '(none yet)'}"}], stats, recall.schema(QUERIES)).get("queries", [])
        except RuntimeError:
            qs = []
        qs = [x for x in qs if isinstance(x, str) and x.strip() and x not in tried][:QUERIES]
        tried += qs
        t0 = time.perf_counter()
        found = agent.search(qs, corpus, dense, bm25, seen)
        stats["search_secs"] += time.perf_counter() - t0
        if found:
            scores, best, graded = agent.grade(client, corpus, q, found, stats, agent.DEEP)
            seen.update(scores)
            pool.update({best[d]: (scores[d], d) for d in found})
            for d in sorted(found, key=lambda d: -scores[d])[: agent.DEEP]:
                pool.update({w: (graded[w], d) for w in corpus.windows(d) if w in graded})
        docs, size = [], 0
        for w in sorted(pool, key=lambda w: -pool[w][0]):
            if size + len(w) > final.CHARS:
                break
            docs.append(w)
            size += len(w)
        if not docs:
            continue
        p = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
            f"Question: {q}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(docs))}], stats)
        answer, notes = str(p.get("answer") or ""), str(p.get("notes", ""))
        score = final.strict(client, q, answer, f"Notes: {notes}\n\n" + "\n\n".join(docs)) if answer else 0.0
        attempts.append((answer, notes, score))
        if score >= final.REDO_BELOW or time.perf_counter() - t_start > agent.DEADLINE:
            break

    answer, notes, score = max(attempts, key=lambda a: a[2], default=("", "", 0.0))
    agree = None
    if answer and docs:
        b = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
            f"Question: {q}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(docs[::-1]))}], stats)
        data, _ = jev._post(client, {"model": jev.JEV_MODEL, "state": {"question": q, "answer_a": answer, "answer_b": str(b.get("answer"))},
                                     "questions": {"same": {"type": "noul", "instructions": SAME}}}, os.environ["TYPESAFE_API_KEY"])
        agree = data["answers"]["same"]["noul"]
    stats["final_attempts"], stats["final_score"] = len(attempts), score
    read = {pool[w][1] for w in docs}
    evidence = {d.docid for d in case.docs if d.label == 1}
    return {
        "qid": case.qid, "answer": answer or None, "llm_answer": attempts[0][0] if attempts else None, "notes": notes, "gold": case.answer,
        "rounds": len(attempts), "secs": time.perf_counter() - t_start, "agree": agree, "queries": len(tried),
        "evidence_retrieved": len(evidence & set(seen)) / len(evidence), "evidence_read": len(evidence & read) / len(evidence),
        "gold_retrieved": len(set(case.gold) & set(seen)) / len(case.gold) if case.gold else None,
        "gold_read": len(set(case.gold) & read) / len(case.gold) if case.gold else None,
        "read": sorted(read), "docs_scored": len(seen), **stats,
    }
