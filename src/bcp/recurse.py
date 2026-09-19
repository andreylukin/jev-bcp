"""One recursive node instead of the search loop + redo loop (agent.solve + final.finalize):

    node(goal): gather evidence FOR THIS GOAL (Jev grades docs against the goal, not the root question)
                -> LLM reads, answers, names what is still unknown -> Jev strict check
                -> pass: return | fail: node(each unknown, one level down), then try the goal again with those facts

A sub-goal is a short question with its entities named, so a mid-chain page is graded against the fact it serves.
Depth 0 with no unknowns is today's round loop. `agent.RECURSIVE` / `modal run ... --recursive` switches it on.
"""

import json
import time

import httpx

from bcp import agent, final
from bcp.corpus import Corpus
from bcp.data import Case

DEPTH = 2  # levels of sub-goals under the root
TRIES = (5, 2, 1)  # attempts at a goal, by depth
SUBGOALS = 3  # per failed attempt
CALLS = 24  # LLM calls per question; past this, nodes stop expanding and return their best
NODE = """You are answering a research question using a search engine over a fixed corpus of web pages. Each query runs
through a keyword index and a semantic index: name concrete things (names, places, years, titles) when you know them.
"Known facts" were established by separate searches, each with a confidence; use them, and doubt the low ones.
If you cannot confirm the answer yet, list what is still unknown as sub-questions: each one short, self-contained,
answerable by its own web search, naming the entities involved (never "the person" or "the answer").
Reply with one JSON object:
{"notes": "facts established so far, with the candidate answers they point to",
 "answer": "best current answer, short, or null",
 "unknown": ["sub-question", "..."],
 "queries": ["next search queries for THIS question"]}"""


def solve(case: Case, corpus: Corpus, client: httpx.Client, dense=None, bm25: bool = True) -> dict:
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0, search_secs=0.0, switched=0, final_secs=0.0, final_attempts=0, final_score=0.0, nodes=0)
    t_start = time.perf_counter()
    all_seen: set[str] = set()
    all_read: set[str] = set()

    def node(goal: str, depth: int) -> tuple[str, str, float]:
        """-> (answer, notes, strict score) of the best attempt."""
        stats["nodes"] += 1
        scores: dict[str, float] = {}
        windows: dict[str, str] = {}
        pool: dict[str, float] = {}
        read: set[str] = set()
        facts: dict[str, str] = {}
        tried: list[str] = []
        best = ("", "", -1.0)
        state = agent.llm(client, [{"role": "system", "content": NODE}, {"role": "user", "content": f"Question: {goal}\n\nNo passages yet. Give your first queries."}], stats)
        for attempt in range(TRIES[depth]):
            t0 = time.perf_counter()
            queries = [q for q in state.get("queries") or [] if isinstance(q, str) and q.strip() and q not in tried][: agent.QUERIES]
            tried += queries
            found = agent.search(queries, corpus, dense, bm25, scores)
            stats["search_secs"] += time.perf_counter() - t0
            if found:
                s, b, graded = agent.grade(client, corpus, goal, found, stats, agent.DEEP)
                scores.update(s)
                windows.update(b)
                pool.update(graded)
            top = sorted((d for d in scores if d not in read), key=lambda d: -scores[d])[: agent.READ]
            read.update(top)
            passages = "\n\n".join(f"[doc {d}]\n{windows[d]}" for d in top)
            state = agent.llm(client, [{"role": "system", "content": NODE}, {"role": "user", "content":
                f"Question: {goal}\n\nKnown facts: {json.dumps(facts)}\n\nYour notes so far: {state.get('notes', '')}\n"
                f"Queries already tried: {json.dumps(tried)}\n\nNew passages:\n{passages}"}], stats)
            answer, notes = str(state.get("answer") or ""), str(state.get("notes", ""))
            ranked, size = [], 0
            for p in sorted(pool, key=lambda p: -pool[p]):
                if size + len(p) > final.CHARS:
                    break
                ranked.append(p)
                size += len(p)
            score = final.strict(client, goal, answer, f"Notes: {notes}\n\nKnown facts: {json.dumps(facts)}\n\n" + "\n\n".join(ranked)) if answer else 0.0
            if score > best[2]:
                best = (answer, notes, score)
            if score >= final.REDO_BELOW or stats["llm_calls"] >= CALLS:
                break
            if depth < DEPTH and attempt + 1 < TRIES[depth]:
                for sub in [u for u in state.get("unknown") or [] if isinstance(u, str) and u.strip() and u not in facts][:SUBGOALS]:
                    if stats["llm_calls"] >= CALLS:
                        break
                    a, _, sc = node(sub, depth + 1)
                    if a:
                        facts[sub] = f"{a} (confidence {max(sc, 0):.2f})"
        all_seen.update(scores)
        all_read.update(read)
        return best

    answer, notes, score = node(case.query, 0)
    stats["final_score"] = max(score, 0.0)
    evidence = {d.docid for d in case.docs if d.label == 1}
    return {
        "qid": case.qid, "answer": answer or None, "llm_answer": answer or None, "notes": notes, "gold": case.answer,
        "rounds": stats["nodes"], "secs": time.perf_counter() - t_start,
        "evidence_retrieved": len(evidence & all_seen) / len(evidence), "evidence_read": len(evidence & all_read) / len(evidence),
        "gold_retrieved": len(set(case.gold) & all_seen) / len(case.gold) if case.gold else None,
        "gold_read": len(set(case.gold) & all_read) / len(case.gold) if case.gold else None,
        "read": sorted(all_read), "docs_scored": len(all_seen), **stats,
    }
