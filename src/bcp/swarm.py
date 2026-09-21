"""Does a Sail-shaped control loop crack what the pipeline cannot? (Sail: 90.7% with an orchestrator that never reads
documents and a swarm of cheap readers.) The hard core of dev is questions whose gold pages only become reachable once
a bridge entity has a NAME, and agent.py's reader is never asked to name one. Here Jev sits in the swarm's filter seat.

  orchestrator  LLM. Sees the question, a compact evidence ledger (constraint, quote <= 300 chars, docid, entities
                named) and the searches done. Emits 1-4 DIRECTED searches - each with an intent: the claim a page
                should state, names filled in - or an answer. Never sees a document. Up to ROUNDS rounds.
  retrieval     BM25 + dense, K hits per index per search (liberal), unseen docs only
  filter        Jev grades each new doc against the search's INTENT (the whole-question grade ranks bridge pages
                out), the best of those also against the whole question; max of the two. The KEEP best go to readers,
                each on its best window for the intent (every window graded).
  readers       LLM, parallel, fresh context, one doc each: drop | evidence {constraint, quote, entities that could
                be the bridge} | extend (once: the whole doc). Not the final read.
  final         answering from extracted facts costs ~8 points (oracle.py --view hops), so the answer comes from a
                coherent read of the ledger docs' windows (around the quote) + final.py's strict check and redo.

Eval: the 15 hard-core dev questions (right in none of ~9 flash runs under out/) + 25 random others, paired against
out/dev_base_v3.

    uv run python -m bcp.swarm --model deepseek/deepseek-v4-flash-0731 --n 40 --workers 5

Result (40 q, flash): NULL. baseline 23/40, swarm 20/40 (fixed 1, broke 4); hard core 1/15, and that one with no gold doc
read. "Every gold doc retrieved" rose 0.62 -> 0.80 but readers dropped 95% of what Jev kept (3357 drop / 160 evidence):
a page read alone, without the chain so far, does not look like evidence - the same failure as unbound fact grading
(atoms.py). 11 of the 20 wrong answers had every gold doc in front of a reader. 8 final reads refused to answer.
4x the baseline's cost (105 LLM calls, 1.6M Jev tokens, ~106 s per question). out/swarm/report.txt.
"""

import argparse
import glob
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev
from bcp.corpus import WINDOW, Corpus
from bcp.data import Case, split

ROUNDS = 14
K = 40  # hits per index per search
KEEP = 8  # docs per round that reach a reader
WHOLE = 24  # best-by-intent docs per round that also get the whole-question grade
FULL_CHARS = 24_000  # an extended read
DEADLINE = 420  # seconds of searching per question (local run; agent.DEADLINE is Modal's)
BASE = "out/dev_base_v3"
FLASH_RUNS = ["dev_base_v3", "modal_dev_final", "modal_dev_passages", "dev_excerpts", "modal_dev_grid", "modal_dev_hybrid", "jevlog_v33", "jevlog_v34", "jevlog_v35"]

ORCH = """You direct a research team answering a hard question over a fixed corpus of web pages. The question describes
something indirectly through several constraints. You never see pages: readers report evidence into a ledger.
Work like a detective: find a constraint specific enough to search, get a NAME (a person, place, work, organisation,
year) into the ledger, then search BY THAT NAME for the next link of the chain. A page about a bridge entity often
looks nothing like the question, so searching for the question's wording will not find it; the name will.
Each search has a "query" (runs on a keyword index and a semantic index) and an "intent": one sentence stating what
a useful page should say, with every known name filled in - pages are filtered against the intent.
Do not repeat a search. If a line of attack is dry, attack a different constraint or a different candidate.
Answer only when the ledger confirms a candidate against the constraints, or when told it is the last round.
Reply with one JSON object:
{"thinking": "what is established, which names are candidates, what is missing (brief)",
 "searches": [{"query": "...", "intent": "..."}],   // 1-4, or [] when answering
 "answer": "short exact answer, or null"}"""
READER = """You are one reader in a research team. You get a research question, the intent of the search that found
this page, and one page (or a part of it). Decide what the page contributes. The page rarely answers the question; it
is useful if it states a fact matching one of the question's constraints, or identifies BY NAME something the question
only describes. A partial match is evidence; drop only pages that bear on nothing. Reply with one JSON object:
{"verdict": "drop" | "evidence" | "extend",   // extend: relevant but the needed part seems to be outside this excerpt
 "constraint": "which part of the question or intent this page bears on",
 "quote": "the sentence(s) from the page that state it, verbatim, at most 300 characters",
 "entities": ["names on this page that could be the thing described, or the next link to search"]}"""


def around(text: str, quote: str) -> str | None:
    """The WINDOW-char slice centred on the quote, if the quote is verbatim."""
    i = text.find(quote[:80]) if quote else -1
    if i < 0:
        return None
    start = max(0, min(i - WINDOW // 2, len(text) - WINDOW))
    return text[start : start + WINDOW]


def solve(case: Case, corpus: Corpus, client: httpx.Client, dense) -> dict:
    stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0, jev_secs=0.0, jev_requests=0, jev_tokens=0)
    t_start = time.perf_counter()
    ask = lambda m: agent.llm(client, m, stats)
    seen: dict[str, float] = {}  # docid -> grade
    pool: dict[str, float] = {}  # window text -> grade, the final read's filler
    ledger: list[dict] = []
    read: set[str] = set()
    done: list[str] = []
    answer, thinking, extends = None, "", 0
    verdicts: dict[str, int] = {}

    def grade(query: str, docs: list[tuple[str, str]]) -> dict[str, float]:
        t0 = time.perf_counter()
        s, r, t = jev.score_batched(client, query, docs)
        stats["jev_secs"] += time.perf_counter() - t0
        stats["jev_requests"] += r
        stats["jev_tokens"] += t
        return s

    def reader(d: str, intent: str, text: str, may_extend: bool) -> dict:
        p = ask([{"role": "system", "content": READER}, {"role": "user", "content": f"Question: {case.query}\n\nSearch intent: {intent}\n\nPage [doc {d}]:\n{text}"}])
        if p.get("verdict") == "extend" and may_extend and len(corpus.text[d]) > len(text):
            return reader(d, intent, corpus.text[d][:FULL_CHARS], False) | {"extended": True}
        return p

    for rnd in range(ROUNDS + 1):
        last = rnd == ROUNDS or time.perf_counter() - t_start > DEADLINE
        lines = "\n".join(f"- [doc {e['docid']}] {e['constraint']}: \"{e['quote']}\" entities: {', '.join(e['entities'])}" for e in ledger) or "(empty)"
        p = ask([{"role": "system", "content": ORCH}, {"role": "user", "content":
                  f"Question: {case.query}\n\nLedger:\n{lines}\n\nSearches done: {json.dumps(done)}\n\nYour last thinking: {thinking}\n\n"
                  + ("LAST ROUND: no more searches. Give your single best answer; it must not be null." if last else f"Round {rnd + 1} of {ROUNDS}.")}])
        thinking = str(p.get("thinking", ""))
        searches = [s for s in p.get("searches") or [] if isinstance(s, dict) and str(s.get("query", "")).strip()][:4]
        if p.get("answer") and (last or not searches):
            answer = str(p["answer"])
            break
        if last or not searches:
            break

        queries = [str(s["query"]) for s in searches]
        intents = [str(s.get("intent") or s["query"]) for s in searches]
        done += queries
        found: dict[str, int] = {}  # new docid -> index of the search that found it
        for i, (q, dh) in enumerate(zip(queries, dense.search_many(queries, K))):
            for d in corpus.search(q, K) + dh:
                if d not in seen and d not in found:
                    found[d] = i
        if not found:
            continue
        scores: dict[str, float] = {}
        best: dict[str, str] = {}
        for i, intent in enumerate(intents):
            docs = [(d, corpus.window(d, queries[i] + " " + intent)) for d, j in found.items() if j == i]
            if docs:
                scores.update(grade(intent, docs))
                best.update(docs)
        ranked = sorted(found, key=lambda d: -scores[d])
        whole = grade(case.query, [(d, best[d]) for d in ranked[:WHOLE]])
        by_intent = dict(scores)
        scores.update({d: max(scores[d], s) for d, s in whole.items()})
        keep = list(dict.fromkeys(ranked[: KEEP // 2] + sorted(found, key=lambda d: -scores[d])))[:KEEP]  # half by intent alone
        # the passage picker: every window of the kept long pages, graded against the intent that found the page
        for i, intent in enumerate(intents):
            parts = [(f"{d}#{n}", w) for d in keep if found[d] == i and len(corpus.text[d]) > WINDOW for n, w in enumerate(corpus.windows(d, 8))]
            for k, s in (grade(intent, parts) if parts else {}).items():
                d, n = k.split("#")
                if s > by_intent[d]:
                    by_intent[d], best[d] = s, corpus.windows(d, 8)[int(n)]
                    scores[d] = max(scores[d], s)
        seen.update(scores)
        pool.update({best[d]: scores[d] for d in found})
        read.update(keep)
        with ThreadPoolExecutor(KEEP) as ex:
            reports = list(ex.map(lambda d: reader(d, intents[found[d]], best[d], True), keep))
        for d, r in zip(keep, reports):
            extends += bool(r.get("extended"))
            verdicts[str(r.get("verdict"))] = verdicts.get(str(r.get("verdict")), 0) + 1
            if r.get("verdict") == "evidence":
                quote = str(r.get("quote", ""))
                ledger.append({"docid": d, "constraint": str(r.get("constraint", ""))[:200], "quote": quote[:300], "entities": [str(e) for e in r.get("entities") or []][:6], "grade": scores[d]})
                pool[around(corpus.text[d], quote) or best[d]] = 1 + scores[d]  # ledger docs lead the final read

    orch_answer = answer
    # the coherent read: ledger windows first, the best graded windows as filler (final.CHARS in all)
    top, size = [], 0
    for w in sorted(pool, key=lambda w: -pool[w]):
        if size + len(w) > final.CHARS:
            break
        top.append(w)
        size += len(w)
    p = ask([{"role": "system", "content": final.READER}, {"role": "user", "content": f"Question: {case.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(top))}]) if top else {}
    read_answer = str(p.get("answer") or orch_answer or "")
    answer, notes, fscores = final.finalize(client, ask, case.query, read_answer, str(p.get("notes", "")), pool)

    gold, led = set(case.gold), {e["docid"] for e in ledger}
    share = lambda s: len(gold & s) / len(gold) if gold else None
    return {"qid": case.qid, "answer": answer, "orch_answer": orch_answer, "read_answer": read_answer, "notes": notes, "gold": case.answer,
            "rounds": rnd, "searches": len(done), "secs": time.perf_counter() - t_start, "ledger": len(ledger), "extends": extends, "verdicts": verdicts,
            "gold_retrieved": share(set(seen)), "gold_read": share(read), "gold_ledger": share(led), "docs_scored": len(seen),
            "final_attempts": len(fscores), "final_score": max(fscores), "thinking": thinking, **stats}


def pick(n: int) -> tuple[list[str], set[str]]:
    """-> (qids: the hard core + a seeded random fill, hard core). Hard core: right in none of FLASH_RUNS / jevlog seeds."""
    base = {json.loads(line)["qid"] for line in open(f"{BASE}/rows.jsonl")}
    right: set[str] = set()
    for r in FLASH_RUNS:
        for line in open(f"out/{r}/rows.jsonl"):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            right |= {x["qid"]} if x.get("correct") else set()
    right |= {f.split("/")[-1][:-5] for f in glob.glob("out/jevlog_seeds/*.json") if json.load(open(f)).get("agent_ok")}
    hard = sorted(base - right)
    rest = sorted(base - set(hard))
    return hard + random.Random(0).sample(rest, max(n - len(hard), 0)), set(hard)


def report(rows: list[dict], hard: set[str]) -> str:
    base = {x["qid"]: x for x in map(json.loads, open(f"{BASE}/rows.jsonl"))}
    rows = [r for r in rows if "error" not in r] + [r for r in rows if "error" in r]
    n = len(rows)
    b = [bool(base[r["qid"]]["correct"]) for r in rows]
    s = [bool(r["correct"]) for r in rows]
    ok = [r for r in rows if "error" not in r]
    mean = lambda k: sum(r[k] for r in ok) / len(ok)
    allgold = lambda rs, k: sum(r.get(k) == 1.0 for r in rs) / max(len(rs), 1)
    cost = mean("jev_tokens") * 0.042e-6
    return "\n".join([
        f"n={n} (hard core {sum(r['qid'] in hard for r in rows)}), errors {n - len(ok)}",
        f"baseline {sum(b)}/{n}   swarm {sum(s)}/{n}   fixed {sum(y and not x for x, y in zip(b, s))}  broke {sum(x and not y for x, y in zip(b, s))}",
        f"hard core solved: {sum(r['correct'] for r in rows if r['qid'] in hard)}/{sum(r['qid'] in hard for r in rows)}",
        f"rest: baseline {sum(x for x, r in zip(b, rows) if r['qid'] not in hard)}  swarm {sum(y for y, r in zip(s, rows) if r['qid'] not in hard)}  of {sum(r['qid'] not in hard for r in rows)}",
        f"orchestrator's own answer right: {sum(bool(r.get('orch_correct')) for r in ok)}/{len(ok)}",
        f"every gold doc retrieved: baseline {allgold([base[r['qid']] for r in ok], 'gold_retrieved'):.2f}  swarm {allgold(ok, 'gold_retrieved'):.2f}",
        f"every gold doc read:      baseline {allgold([base[r['qid']] for r in ok], 'gold_read'):.2f}  swarm {allgold(ok, 'gold_read'):.2f}   in ledger {allgold(ok, 'gold_ledger'):.2f}",
        f"  hard core only:         baseline {allgold([base[r['qid']] for r in ok if r['qid'] in hard], 'gold_read'):.2f}  swarm {allgold([r for r in ok if r['qid'] in hard], 'gold_read'):.2f}",
        f"rounds {mean('rounds'):.1f}  searches {mean('searches'):.1f}  ledger {mean('ledger'):.1f}  secs {mean('secs'):.0f}",
        f"llm calls {mean('llm_calls'):.0f}  in {mean('llm_in') / 1e3:.0f}k out {mean('llm_out') / 1e3:.1f}k   jev {mean('jev_tokens') / 1e3:.0f}k tokens = {100 * cost:.1f} cents",
    ])


def main():
    from bcp.dense import Dense
    from bcp.judge import judge

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="OpenRouter model id")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--out", default="out/swarm")
    ap.add_argument("--limit", type=int, help="run only the first questions still to do (smoke test)")
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    qids, hard = pick(a.n)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    rows = []
    if rows_path.exists():
        for line in rows_path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except ValueError:
                pass
    todo = [c for c in split()[0] if c.qid in set(qids) - {r["qid"] for r in rows}]
    todo = todo[: a.limit]
    print(f"{len(rows)} done, {len(todo)} to go", flush=True)
    if todo:
        corpus, dense, lock = Corpus(), Dense(), threading.Lock()
        with httpx.Client(timeout=120) as client:

            def one(case):
                try:
                    row = solve(case, corpus, client, dense)
                    row.update(judge(client, case.query, row["answer"], case.answer, row["notes"]))
                    row["orch_correct"] = bool(row["orch_answer"]) and judge(client, case.query, row["orch_answer"], case.answer, row["thinking"])["correct"]
                    print(f"{case.qid:>5} {'OK ' if row['correct'] else '-- '}{'H ' if case.qid in hard else '  '}{row['secs']:4.0f}s r={row['rounds']} gold_read={row['gold_read']} {str(row['answer'])[:40]!r} / {case.answer[:40]!r}", flush=True)
                except Exception as e:
                    row = {"qid": case.qid, "gold": case.answer, "correct": False, "error": repr(e)}
                    print(f"{case.qid:>5} ERR {e!r}", flush=True)
                with lock, rows_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                return row

            with ThreadPoolExecutor(a.workers) as ex:
                rows += list(ex.map(one, todo))
    text = report(rows, hard)
    (out / "report.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
