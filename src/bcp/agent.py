"""End-to-end BrowseComp-Plus: an LLM (any OpenRouter model) proposes queries and reads, Jev grades what search returns.

Per round: LLM -> search queries; BM25 + dense retrieval over the 100k corpus; Jev scores every new doc against the
question and picks the window of each long page worth reading; the LLM reads the 8 best unread passages, updates its
notes, and answers or asks for more queries. When it is confident (or out of rounds) the final stage (final.py) has Jev
strictly check the answer against the best windows of everything graded, and re-reads on a rejection.
`--no-jev` is the ablation: the LLM reads the top search hits instead.

Switches that are OFF by default were measured and did not help (README, "What did not work"): EXCERPTS, RECURSIVE,
FINAL_NOTES=False, FINAL_SEARCH, --grid. SPLIT (split.py) is the shape for small models.

    uv run python -m bcp.agent --model deepseek/deepseek-v4-flash-0731 --n 50 --out out/agent
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import final, jev
from bcp import grid as gridlib
from bcp.corpus import WINDOW, Corpus, excerpts
from bcp.data import Case, load_cases

LLM_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_TEMPERATURE = None  # None: the provider's default
LLM_KEY = ""  # bearer for a self-hosted endpoint (modal_llm.py); empty: OPENROUTER_API_KEY
LLM_MODEL = ""  # set from the required --model (an OpenRouter id)
REASONING = {"enabled": False}  # --reasoning low|medium|high turns it on (DeepSeek flash: ~40 s/call vs ~2 s)
ROUNDS = 8
QUERIES = 8  # per round
HITS = 15  # BM25 hits per query
READ = 8  # passages the LLM reads per round
SPLIT = False  # split.solve: the small-model shape (query writer + Jev + reader), instead of solve
RECURSIVE = False  # recurse.solve (one recursive node) instead of solve (search loop + redo loop)
EXCERPTS = False  # the reader gets Jev-picked ~400-char excerpts from many docs instead of 8 whole windows (snip.py)
EXCERPT_DOCS = 40  # new docs per round whose window is cut into excerpts and graded
EXCERPT_CHARS = 32_000  # reader input per round, the size of 8 windows
FINAL_NOTES = True  # the strict check sees the agent's notes beside the passages (replay.py)
FINAL_SEARCH = False  # a rejected answer triggers new queries before the redo (replay.py)
DEEP = 8  # new docs per round that get every window graded, to pick the passage shown. More does not improve the
# doc ranking (rank.py: gold recall in top 8 is 0.75 at any depth, all windows of all docs included, at up to 9x the tokens)
DEADLINE = 240  # seconds of searching per question, then straight to the final stage: an input that reaches Modal's
# per-input timeout kills its container and the 15 other questions in it ("cannot schedule new futures after shutdown")
LLM_DEADLINE = 60  # seconds per LLM call before it is abandoned and retried
_POOL = ThreadPoolExecutor(64)

SYSTEM = """You are answering a hard research question using a search engine over a fixed corpus of web pages.
The question describes something indirectly through several constraints; work out what it is step by step.
Each query runs through both a keyword index (BM25) and a semantic embedding index: name concrete things
(names, places, years, titles) when you know them, describe what you are looking for when you do not,
and vary the queries - one per constraint, plus queries for candidate answers you want to verify.
Reply with one JSON object:
{"notes": "facts established so far with the candidate answers they point to",
 "answer": "best current answer, short, or null",
 "confident": true only if the passages confirm the answer against the constraints,
 "queries": ["next search queries"]}"""
REQUERY = """A strict checker rejected the answers so far to a research question. Write search queries for the constraints
those answers fail or leave unconfirmed, and for other candidate answers.
Reply with JSON: {"queries": ["..."]}"""
CANDIDATES = '\nAlso include "candidates": up to 3 distinct plausible short exact answers, best first.'


def llm(client: httpx.Client, messages: list[dict], stats: dict, schema: dict | None = None, text: bool = False):
    """`schema`: a JSON Schema the reply must match (constrained decoding), instead of free-form JSON.
    `text`: return the reply as a plain string (jevlog programs are not JSON)."""
    body = {
        "model": LLM_MODEL,
        "messages": messages,
        "response_format": {"type": "json_schema", "json_schema": {"name": "reply", "strict": True, "schema": schema}} if schema else {"type": "json_object"},
        "reasoning": REASONING,
        "provider": {"sort": "throughput"},
    }
    if text:
        del body["response_format"]
    if LLM_TEMPERATURE is not None:
        body["temperature"] = LLM_TEMPERATURE
        body["max_tokens"] = 3000  # self-hosted: constrained decoding can otherwise run one string to the context limit
    for attempt in range(6):
        t0 = time.perf_counter()
        try:
            # follow_redirects: a Modal web endpoint answers a request still running after 150 s with a 303 to
            # a result URL (without it every queued call failed as "HTTP 303").
            # wall-clock cap: OpenRouter keeps a stalled generation's connection alive, so httpx's read timeout
            # never fires (5 of 200 dev questions hung this way). The retry drops the provider pin.
            r = _POOL.submit(client.post, LLM_URL, json=body, follow_redirects=True, headers={"Authorization": f"Bearer {LLM_KEY or os.environ['OPENROUTER_API_KEY']}"}).result(timeout=LLM_DEADLINE if not REASONING.get("effort") else 4 * LLM_DEADLINE)
        except (httpx.TransportError, TimeoutError):
            body.pop("provider", None)
            stats["llm_retries"] = stats.get("llm_retries", 0) + 1
            time.sleep(2**attempt)
            continue
        if r.status_code == 200:
            data = r.json()
            stats["llm_secs"] += time.perf_counter() - t0
            stats["llm_calls"] += 1
            stats["llm_in"] += data["usage"]["prompt_tokens"]
            stats["llm_out"] += data["usage"]["completion_tokens"]
            try:
                reply = data["choices"][0]["message"]["content"]
                if text:
                    return reply
                return json.loads(reply[reply.index("{") : reply.rindex("}") + 1])
            except (ValueError, TypeError):
                continue
        elif r.status_code in (429, 503) or r.status_code >= 500:
            stats["llm_retries"] = stats.get("llm_retries", 0) + 1
            time.sleep(2**attempt)
        else:
            raise RuntimeError(f"llm: HTTP {r.status_code}: {r.text[:300]}")
    raise RuntimeError("llm: retries exhausted")


def use_endpoint(url: str) -> None:
    """Point every LLM call at a self-hosted OpenAI-compatible endpoint (modal_llm.py serves the model as "slm")."""
    global LLM_URL, LLM_MODEL, LLM_KEY
    from bcp.data import DATA_DIR

    global LLM_DEADLINE, LLM_TEMPERATURE
    LLM_URL, LLM_MODEL, LLM_KEY = url, "slm", (DATA_DIR / "slm_key").read_text().strip()
    LLM_TEMPERATURE = 0.3  # vLLM falls back to temperature 1.0 when the model ships no default: answers came back empty or rambling
    LLM_DEADLINE = 600  # one GPU queues requests under load; abandoning and retrying them only adds to the queue


def grade(client: httpx.Client, corpus: Corpus, question: str, found: dict[str, str], stats: dict, deep: int) -> tuple[dict, dict, dict]:
    """`found`: docid -> the query that found it. Jev grades each doc's term-overlap window, then every window of the
    `deep` best (that window missed the answer in 14 of 24 dev failures where every gold doc had been read).
    -> (docid -> score, docid -> its best window, window text -> score)."""
    t0 = time.perf_counter()
    best = {d: corpus.window(d, q + " " + question) for d, q in found.items()}
    scores, requests, tokens = jev.score_batched(client, question, list(best.items()))
    passages = {best[d]: scores[d] for d in found}
    parts = [(f"{d}#{i}", w) for d in sorted(found, key=lambda d: -scores[d])[:deep] if len(corpus.text[d]) > WINDOW for i, w in enumerate(corpus.windows(d))]
    if parts:
        s, r, t = jev.score_batched(client, question, parts)
        requests, tokens = requests + r, tokens + t
        for k, w in parts:
            d = k.split("#")[0]
            passages[w] = s[k]
            if s[k] > scores[d]:
                scores[d], best[d] = s[k], w
    stats["jev_secs"] += time.perf_counter() - t0
    stats["jev_requests"] += requests
    stats["jev_tokens"] += tokens
    return scores, best, passages


def search(queries: list[str], corpus: Corpus, dense, bm25: bool, skip) -> dict[str, str]:
    """-> new docid -> the query that found it. The retrievers only propose candidates; Jev does the ranking,
    so no score fusion is needed."""
    found: dict[str, str] = {}
    dense_hits = dense.search_many(queries, HITS) if dense and queries else [[] for _ in queries]
    for q, dh in zip(queries, dense_hits):
        for d in (corpus.search(q, HITS) if bm25 else []) + dh:
            if d not in skip and d not in found:
                found[d] = q
    return found


def requery(question: str, corpus: Corpus, dense, bm25: bool, client: httpx.Client, ask, seen: dict, stats: dict):
    """The final stage's `search`: new queries aimed at what the checker rejected, graded like any round."""
    def more(rejected: str, notes: str) -> dict[str, float]:
        p = ask([{"role": "system", "content": REQUERY}, {"role": "user", "content": f"Question: {question}\n\nRejected answers: {rejected}\n\nReasoning behind the last one: {notes}"}])
        found = search([q for q in p.get("queries", []) if isinstance(q, str) and q.strip()][:QUERIES], corpus, dense, bm25, seen)
        if not found:
            return {}
        scores, _, graded = grade(client, corpus, question, found, stats, DEEP)
        seen.update(scores)
        return graded
    return more


def solve(case: Case, corpus: Corpus, client: httpx.Client, use_jev: bool, dense=None, bm25: bool = True, use_grid: bool = False, use_final: bool = True, use_passages: bool = True, sink: dict | None = None) -> dict:
    """`sink`: receives the evidence (`pool`: window -> Jev grade, `scores`: docid -> grade) for a caller that goes on working with it."""
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0, search_secs=0.0, switched=0, final_secs=0.0, final_attempts=0, final_score=0.0)
    t_start = time.perf_counter()
    seen: dict[str, float] = {}  # docid -> jev score (or -rank without jev)
    read: set[str] = set()
    state = {"notes": "", "answer": None, "confident": False, "queries": []}
    system = SYSTEM + (CANDIDATES if use_grid else "")
    messages = [{"role": "system", "content": system}, {"role": "user", "content": f"Question: {case.query}\n\nNo passages yet. Give your first queries."}]
    grid = None
    if use_grid:
        constraints = gridlib.compile_question(lambda m: llm(client, m, stats), case.query)
        grid = gridlib.Grid(client, constraints) if constraints else None  # no usable program: plain loop
    shown: dict[str, str] = {}  # docid -> the passage text the LLM read
    windows: dict[str, str] = {}  # docid -> its best-graded window
    pool: dict[str, float] = {}  # every graded window's text -> Jev score: the final stage reads the best of all of them
    xpool: dict[str, tuple[float, str]] = {}  # "docid#i" -> (Jev score, excerpt)
    xshown: set[str] = set()
    hint = ""

    for rnd in range(ROUNDS + 1):
        state = {**state, **llm(client, messages, stats)}
        if grid and shown and state.get("answer"):
            t0 = time.perf_counter()
            names = [str(state["answer"])] + [str(c) for c in state.get("candidates") or [] if c]
            cands = list(dict.fromkeys(names))[:4]
            grid.update(shown, cands)
            stats["jev_secs"] += time.perf_counter() - t0
            leader = max(cands, key=grid.mean)
            claim, support = grid.weakest(leader)
            hint = (f'Evidence check: "{leader}" is the best-supported candidate so far (support {grid.mean(leader):.2f}). '
                    f'Its least-supported constraint is: "{claim}" ({support:.2f}). Search for evidence for or against exactly that.\n')
        if rnd == ROUNDS or (state.get("confident") and state.get("answer")) or time.perf_counter() - t_start > DEADLINE:
            break

        t0 = time.perf_counter()
        queries = [q for q in state.get("queries", []) if isinstance(q, str) and q.strip()][:QUERIES]
        found = search(queries, corpus, dense, bm25, seen)
        stats["search_secs"] += time.perf_counter() - t0

        if use_jev and found:
            scores, best, graded = grade(client, corpus, case.query, found, stats, DEEP if use_passages else 0)
            seen.update(scores)
            windows.update(best)
            pool.update(graded)
        else:
            # ablation: interleave the queries' BM25 rankings (found preserves that order)
            seen.update({d: -float(len(seen) + i) for i, d in enumerate(found)})
            windows.update({d: corpus.window(d, q + " " + case.query) for d, q in found.items()})

        if use_jev and EXCERPTS:
            t0 = time.perf_counter()
            parts = [(f"{d}#{i}", e) for d in sorted(found, key=lambda d: -seen[d])[:EXCERPT_DOCS] for i, e in enumerate(excerpts(windows[d]))]
            if parts:
                s, requests, tokens = jev.score_batched(client, case.query, parts)
                stats["jev_requests"] += requests
                stats["jev_tokens"] += tokens
                xpool.update({k: (s[k], e) for k, e in parts})
                pool.update({e: s[k] for k, e in parts})
            stats["jev_secs"] += time.perf_counter() - t0
            picked, size = [], 0
            for k in sorted((k for k in xpool if k not in xshown), key=lambda k: -xpool[k][0]):
                if size + len(xpool[k][1]) > EXCERPT_CHARS:
                    break
                picked.append(k)
                size += len(xpool[k][1])
            xshown.update(picked)
            top = list(dict.fromkeys(k.split("#")[0] for k in picked))
            by_doc = {d: sorted((k for k in picked if k.split("#")[0] == d), key=lambda k: int(k.split("#")[1])) for d in top}
            passages = "\n\n".join(f"[doc {d}]\n" + " ... ".join(xpool[k][1] for k in by_doc[d]) for d in top)
            shown.update({d: windows[d] for d in top})
        else:
            top = sorted((d for d in seen if d not in read), key=lambda d: -seen[d])[:READ]
            shown.update({d: windows[d] for d in top})
            passages = "\n\n".join(f"[doc {d}]\n{windows[d]}" for d in top)
        read.update(top)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {case.query}\n\nYour notes so far: {state.get('notes', '')}\n"
             f"Queries already tried: {json.dumps(state.get('queries', []))}\n\nNew passages:\n{passages}\n\n"
             + hint
             + (f"Round {rnd + 1} of {ROUNDS}. Update notes, answer if confirmed, else give new queries."
                if rnd + 1 < ROUNDS else "Final round: no more searches. Give your single best-guess answer; it must not be null.")},
        ]

    evidence = {d.docid for d in case.docs if d.label == 1}
    answer, notes = state.get("answer"), str(state.get("notes", ""))
    if use_jev and use_final and shown:  # also when the loop ended with no answer: the redo reader supplies one
        t0 = time.perf_counter()
        ask = lambda m: llm(client, m, stats)
        answer, notes, scores = final.finalize(client, ask, case.query, str(answer or ""), notes, pool, FINAL_NOTES,
                                               requery(case.query, corpus, dense, bm25, client, ask, seen, stats) if FINAL_SEARCH else None)
        stats["final_secs"], stats["final_attempts"], stats["final_score"] = time.perf_counter() - t0, len(scores), max(scores)
    if grid and answer:
        # leave the LLM's answer only for a candidate whose evidence leads by a margin (dev: fixes 15, breaks 4)
        cands = [c for c in dict.fromkeys(c for c, _ in grid.cells) if c != str(answer)]
        rival = max(cands, key=grid.mean, default=None)
        if rival and grid.mean(rival) - grid.mean(str(answer)) > gridlib.MARGIN:
            stats["switched"] = 1
            answer = rival
        stats["jev_requests"] += grid.requests
        stats["jev_tokens"] += grid.tokens
    if sink is not None:
        sink.update(pool=pool, scores=seen)
    return {
        "qid": case.qid,
        "answer": answer,
        "llm_answer": state.get("answer"),
        "notes": notes,
        "gold": case.answer,
        "rounds": rnd,
        "secs": time.perf_counter() - t_start,
        "evidence_retrieved": len(evidence & set(seen)) / len(evidence),
        "evidence_read": len(evidence & read) / len(evidence),
        "gold_retrieved": len(set(case.gold) & set(seen)) / len(case.gold) if case.gold else None,
        "gold_read": len(set(case.gold) & read) / len(case.gold) if case.gold else None,
        "read": sorted(read),
        "docs_scored": len(seen),
        **stats,
    }


def run_one(case: Case, corpus: Corpus, client: httpx.Client, use_jev: bool, dense, retriever: str, use_grid: bool = False, use_final: bool = True) -> dict:
    """solve + official judge. A question that raises becomes an incorrect row with an `error` field."""
    from bcp.judge import judge

    try:
        if SPLIT:
            from bcp import split

            row = split.solve(case, corpus, client, dense, retriever != "dense")
        elif RECURSIVE:
            from bcp import recurse

            row = recurse.solve(case, corpus, client, dense, retriever != "dense")
        else:
            row = solve(case, corpus, client, use_jev, dense, retriever != "dense", use_grid, use_final)
        row.update(judge(client, case.query, row["answer"], case.answer, row["notes"]))
        print(f"{row['qid']:>5} {'OK ' if row['correct'] else '-- '} {row['secs']:5.1f}s ev_read={row['evidence_read']:.2f} "
              f"{str(row['answer'])[:40]!r} / {row['gold'][:40]!r}", flush=True)
    except Exception as e:
        row = {"qid": case.qid, "gold": case.answer, "correct": False, "error": repr(e)}
        print(f"{case.qid:>5} ERR {e!r}", flush=True)
    return row


def summarize(rows: list[dict], meta: dict) -> dict:
    import random

    from bcp.judge import MODEL as JUDGE_MODEL

    n = len(rows)
    ok = [r for r in rows if "error" not in r]  # cost/latency means are over questions that ran; accuracy is over all
    mean = lambda k: sum(r[k] for r in ok) / len(ok)
    correct = [r["correct"] for r in rows]
    rng = random.Random(0)
    boot = sorted(sum(rng.choices(correct, k=n)) / n for _ in range(2000))
    return {
        "n": n, **meta, "judge": JUDGE_MODEL,
        "accuracy": sum(correct) / n, "accuracy_ci95": [boot[50], boot[1949]],
        "errors": n - len(ok), "unanswered": sum(not r["answer"] for r in ok),
        "secs_mean": mean("secs"), "secs_median": sorted(r["secs"] for r in ok)[len(ok) // 2],
        "evidence_retrieved": mean("evidence_retrieved"), "evidence_read": mean("evidence_read"),
        **{k: mean(k) for k in ("rounds", "docs_scored", "llm_retries", "llm_calls", "llm_secs", "llm_in", "llm_out", "jev_secs", "jev_requests", "jev_tokens", "search_secs", "final_secs", "final_attempts")},
    }


def main():
    global LLM_MODEL, REASONING
    import threading

    from bcp.data import load_all

    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--all", action="store_true", help="all 830 queries (6 shards) instead of the --n sample")
    ap.add_argument("--out", default="out/agent")
    ap.add_argument("--no-jev", action="store_true")
    ap.add_argument("--no-final", action="store_true", help="skip the strict-check / redo final stage")
    ap.add_argument("--grid", action="store_true", help="candidate x constraint grid drives search and picks the answer")
    ap.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reasoning", choices=["low", "medium", "high"])
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    a = ap.parse_args()
    LLM_MODEL = a.model
    if a.reasoning:
        REASONING = {"effort": a.reasoning}

    cases = load_all() if a.all else load_cases(a.n)
    corpus = Corpus()
    dense = None
    if a.retriever != "bm25":
        from bcp.dense import Dense

        dense = Dense()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    # resume: rows.jsonl holds every finished question; a torn last line from a kill is dropped
    rows_path = out / "rows.jsonl"
    rows = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
        rows_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    done = {r["qid"] for r in rows}
    todo = [c for c in cases if c.qid not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    lock = threading.Lock()

    with httpx.Client(timeout=120) as client:

        def one(case):
            row = run_one(case, corpus, client, not a.no_jev, dense, a.retriever, a.grid, not a.no_final)
            with lock, rows_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
            return row

        with ThreadPoolExecutor(a.workers) as pool:
            rows += list(pool.map(one, todo))

    summary = summarize(rows, {"jev": not a.no_jev, "grid": a.grid, "final": not a.no_final, "retriever": a.retriever, "model": LLM_MODEL, "reasoning": REASONING, "workers": a.workers})
    (out / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
