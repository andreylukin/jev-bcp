"""Does a late-interaction retriever reach gold docs that BM25 + Qwen3-Embedding-8B miss? Every search experiment
so far used the same two indexes. Third index: lightonai/Reason-ModernColBERT (149M) over LightOn's prebuilt
FastPLAID index (first 512 tokens of each document), served by modal_colbert.py. Retrieval only: no Jev, no reader.

  raw      the question itself is the query; gold recall at 15 / 100 / 1000 per index and for the unions
  written  8 queries per question written by the LLM from the question alone (recall.py's writer prompt, round 1);
           a doc counts if any query reaches it within k hits of that index
  new      gold docs ColBERT has in its top 100 that BM25 + dense do not (same queries, same depth), and that the
           live agent never pooled (out/jevlog_seeds: every doc the flash pipeline graded, ~240 per question)
  hard     the dev questions no run in out/*/rows.jsonl ever answered right

    uv run python -m bcp.lateint --stage queries --model deepseek/deepseek-v4-flash-0731
    modal run modal_colbert.py --queries out/lateint/q.json --out out/lateint/hits.json --k 1000
    uv run python -m bcp.lateint --stage eval

Result (dev 200, 600 gold docs; hard core 14 questions, 27 gold docs; full table in out/lateint/results.json):
  written queries, 15 hits each: gold recall bm25 0.26, dense 0.38, colbert 0.51; bm25+dense 0.48 -> all three 0.63;
  questions with every gold doc 30.5% -> 43%. At 100 hits: 0.72 -> 0.85. As the raw-question retriever it is WORSE
  than dense (0.36 vs 0.49 at 100): it wants short written queries.
  Of the 95 gold docs the live flash agent never pooled, ColBERT reaches 30 at 15 hits and 51 at 100 (35 questions, 13
  of the agent's 45 wrong answers). Hard core: 0 of its 7 never-pooled gold docs; +2 docs over bm25+dense at 100.
  So: a real recall gain on ordinary questions, nothing for the hard core. Whether recall turns into accuracy is
  open (claimsweep.py raised reader-visible gold 59.5% -> 81% for +0 points).
"""

import argparse
import glob
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent
from bcp.data import split
from bcp.recall import PROMPT, schema

OUT = Path("out/lateint")
N_WRITTEN = 8
KS = (15, 100, 1000)


def hard_core(ids: set[str]) -> set[str]:
    right = dict.fromkeys(ids, 0)
    for f in glob.glob("out/*/rows.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if str(r["qid"]) in right:
                right[str(r["qid"])] += bool(r.get("correct"))
    return {q for q, n in right.items() if n == 0}


def stage_queries(model: str):
    agent.LLM_MODEL = model
    dev = split()[0]
    path = OUT / "written.json"
    written = json.loads(path.read_text()) if path.exists() else {}
    with httpx.Client(timeout=120) as client:

        def one(case):
            if case.qid not in written:
                msgs = [{"role": "system", "content": PROMPT.format(n=N_WRITTEN)},
                        {"role": "user", "content": f"Question: {case.query}\n\nEarlier queries: []\n\nTop snippets so far:\n(none yet)"}]
                qs = agent.llm(client, msgs, dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0), schema(N_WRITTEN)).get("queries", [])
                written[case.qid] = [q for q in qs if isinstance(q, str) and q.strip()][:N_WRITTEN]

        with ThreadPoolExecutor(8) as pool:
            list(pool.map(one, dev))
    path.write_text(json.dumps(written))
    q = {f"raw:{c.qid}": c.query for c in dev}
    q |= {f"w{i}:{qid}": text for qid, qs in written.items() for i, text in enumerate(qs)}
    (OUT / "q.json").write_text(json.dumps(q))
    print(len(q), "queries ->", OUT / "q.json")


def stage_eval():
    from bcp.corpus import Corpus
    from bcp.dense import Dense

    dev = split()[0]
    hard = hard_core({c.qid for c in dev})
    q = json.loads((OUT / "q.json").read_text())
    hits = {"colbert": json.loads((OUT / "hits.json").read_text())}
    cache = OUT / "base_hits.json"
    if cache.exists():
        hits |= json.loads(cache.read_text())
    else:
        corpus, dense = Corpus(), Dense()
        keys = list(q)
        hits["bm25"] = {k: corpus.search(q[k], KS[-1]) for k in keys}
        hits["dense"] = {}
        for i in range(0, len(keys), 64):
            chunk = keys[i : i + 64]
            hits["dense"] |= dict(zip(chunk, dense.search_many([q[k] for k in chunk], KS[-1])))
        cache.write_text(json.dumps({"bm25": hits["bm25"], "dense": hits["dense"]}))

    def reached(index: str, qid: str, mode: str, k: int) -> set[str]:
        keys = [f"raw:{qid}"] if mode == "raw" else [x for x in (f"w{i}:{qid}" for i in range(N_WRITTEN)) if x in q]
        return {d for key in keys for d in hits[index][key][:k]}

    arms = {"bm25": ["bm25"], "dense": ["dense"], "colbert": ["colbert"], "bm25+dense": ["bm25", "dense"], "all three": ["bm25", "dense", "colbert"]}
    res = {"hard_qids": sorted(hard, key=int), "recall": {}, "new": {}}
    for name, cases in (("dev", dev), ("hard", [c for c in dev if c.qid in hard])):
        n_gold = sum(len(c.gold) for c in cases)
        for mode in ("raw", "written"):
            for arm, idx in arms.items():
                row = {}
                for k in KS:
                    got = sum(len(set(c.gold) & set().union(*(reached(i, c.qid, mode, k) for i in idx))) for c in cases)
                    allq = sum(set(c.gold) <= set().union(*(reached(i, c.qid, mode, k) for i in idx)) for c in cases)
                    row[k] = [round(got / n_gold, 3), round(allq / len(cases), 3)]
                res["recall"][f"{name} {mode} {arm}"] = row
        # new gold docs: ColBERT top 100 (raw or written) minus BM25+dense at the same depth, minus the live agent's pool
        new_vs_union = new_vs_agent = agent_missed = q_helped = 0
        for c in cases:
            col = reached("colbert", c.qid, "raw", 100) | reached("colbert", c.qid, "written", 100)
            base = set().union(*(reached(i, c.qid, m, 100) for i in ("bm25", "dense") for m in ("raw", "written")))
            seed = Path(f"out/jevlog_seeds/{c.qid}.json")
            pool = set(json.loads(seed.read_text())["scores"]) if seed.exists() else set()
            gold = set(c.gold)
            new_vs_union += len(gold & col - base)
            agent_missed += len(gold - pool)
            new_vs_agent += len((gold - pool) & col)
            q_helped += bool((gold - pool) & col)
        res["new"][name] = {"questions": len(cases), "gold_docs": n_gold, "colbert_top100_not_in_bm25_dense_top100": new_vs_union,
                            "gold_the_live_agent_never_pooled": agent_missed, "of_those_in_colbert_top100": new_vs_agent, "questions_gaining_a_gold_doc": q_helped}
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    print("gold recall (share of gold docs) / share of questions with every gold doc, at k =", KS)
    for name, row in res["recall"].items():
        print(f"  {name:28}", "   ".join(f"{row[k][0]:.3f} / {row[k][1]:.3f}" for k in KS))
    print(json.dumps(res["new"], indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["queries", "eval"], required=True)
    ap.add_argument("--model")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_queries(a.model) if a.stage == "queries" else stage_eval()
