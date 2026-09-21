"""Jev decides WHEN to pay for a strong reader. Four cheap trajectories per question (bestofn.py): where their answers
all agree the answer is right ~95% of the time and is kept; where they disagree (~27% of dev) the summed-strict-score
pick is right only ~60%. Here those questions get ONE read by a strong model of the four runs' pooled evidence:
every doc any run read, Jev-graded, best --top windows.

  keep     bestofn's voteown pick everywhere                                   (the baseline, no strong model)
  blind    the strong reader sees the question and the windows
  shown    ... and the four runs' answers with their notes, to confirm or overrule
  control  the CHEAP model doing the blind read: separates "stronger reader" from "pooled evidence, fresh read"

    uv run python -m bcp.route --runs out/cb_1 out/cb_2 out/cb_3 out/cb_4 --selected out/bestofn/results_cb4.json \\
        --strong openai/gpt-5 --reasoning medium --cheap deepseek/deepseek-v4-flash-0731

Result (dev 200): keep 85.5%, control 82.5% (fixed 5, broke 11), blind 89.5% (+4.0 [+1.0, +7.0], fixed 9, broke 1),
shown 89.0%. The gain is the reader, not the pooled evidence. `final` below is the blind arm as one label-free system:

    uv run python -m bcp.route --final --split held --runs out/held_cb_1 out/held_cb_2 out/held_cb_3 out/held_cb_4 \\
        --strong openai/gpt-5 --reasoning medium --out out/publish/held
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final
from bcp.bestofn import cluster, load_run
from bcp.corpus import Corpus
from bcp.data import split
from bcp.judge import judge
from bcp.metrics import bootstrap_ci, paired_bootstrap

OUT = Path("out/route")
SHOWN = "\n\nFour independent attempts answered as follows. They disagree, so at least some are wrong; confirm one only if the documents really support it.\n"


def final_run(a):
    """The system, label-free: cluster the runs' answers (Jev), keep the cluster with the largest summed strict score
    where there is one cluster, else one strong blind read. Rows are appended as they finish, so a rerun resumes."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in a.runs]
    cases = {"dev": split()[0], "held": split()[1]}[a.split]
    rows_path = out / "rows.jsonl"
    rows = [json.loads(x) for x in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r["qid"] for r in rows}
    corpus = Corpus()
    agent.LLM_MODEL, agent.REASONING, agent.LLM_DEADLINE = a.strong, {"effort": a.reasoning}, 75  # llm() allows 4x with reasoning on: 5 minutes a read

    with httpx.Client(timeout=600) as client:

        def one(c):
            trajs = [r.get(c.qid) or {} for r in runs]  # a missing or errored row has no answer
            answers = [str(t.get("answer")) if t.get("answer") not in (None, "None") else "" for t in trajs]
            groups = cluster(client, c.query, answers)
            row = {"qid": c.qid, "gold": c.answer, "answers": answers, "clusters": len(groups)}
            if len(groups) == 1:
                i = groups[0][0]
                return row | {"source": "agree", "answer": answers[i], "correct": bool(trajs[i].get("correct"))}
            best = max(groups, key=lambda g: sum(trajs[i].get("final_score") or 0.0 for i in g))[0] if groups else None
            row["cheap_pick"], row["cheap_correct"] = (answers[best], bool(trajs[best].get("correct"))) if groups else (None, False)
            stats = dict(jev_secs=0.0, jev_requests=0, jev_tokens=0, llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            docs = {d: c.query for t in trajs for d in t.get("read") or []}
            try:
                _, _, passages = agent.grade(client, corpus, c.query, docs, stats, deep=len(docs))
                top = sorted(passages, key=lambda w: -passages[w])[: a.top]
                p = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content":
                    f"Question: {c.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(top))}], stats)
                row |= {"source": "strong", "answer": str(p.get("answer")), **judge(client, c.query, p.get("answer"), c.answer, str(p.get("notes", "")))}
            except Exception as e:  # counted as wrong, never dropped
                row |= {"source": "strong", "answer": None, "correct": False, "error": repr(e)}
            return row | stats

        with ThreadPoolExecutor(a.workers) as ex, rows_path.open("a") as f:
            for row in ex.map(one, [c for c in cases if c.qid not in done]):
                rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                if len(rows) % 50 == 0:
                    print(f"{len(rows)}/{len(cases)}  acc {sum(bool(r['correct']) for r in rows) / len(rows):.3f}", flush=True)

    n = len(rows)
    strong = [r for r in rows if r["source"] == "strong"]
    agree = [r for r in rows if r["source"] == "agree"]
    acc = [float(bool(r["correct"])) for r in rows]
    cheap = [float(bool(r["correct"] if r["source"] == "agree" else r["cheap_correct"])) for r in rows]
    _, lo, hi = paired_bootstrap(acc, cheap)
    runs_acc = [sum(bool((r.get(x["qid"]) or {}).get("correct")) for x in rows) / n for r in runs]
    summary = {"split": a.split, "n": n, "runs": a.runs, "strong": a.strong, "reasoning": a.reasoning, "top": a.top,
               "accuracy": sum(acc) / n, "accuracy_ci95": bootstrap_ci(acc), "single_runs": runs_acc,
               "cheap_only": sum(cheap) / n, "cheap_only_ci95": bootstrap_ci(cheap), "strong_vs_cheap": [sum(acc) / n - sum(cheap) / n, lo, hi],
               "agree_n": len(agree), "agree_accuracy": sum(bool(r["correct"]) for r in agree) / max(len(agree), 1),
               "strong_n": len(strong), "strong_accuracy": sum(bool(r["correct"]) for r in strong) / max(len(strong), 1),
               "strong_errors": sum("error" in r for r in strong),
               "strong_in": sum(r.get("llm_in", 0) for r in strong) / max(len(strong), 1), "strong_out": sum(r.get("llm_out", 0) for r in strong) / max(len(strong), 1),
               "strong_jev_tokens": sum(r.get("jev_tokens", 0) for r in strong) / max(len(strong), 1)}
    (out / "results.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--selected", default="", help="bestofn results for the same runs (the arms experiment)")
    ap.add_argument("--strong", required=True)
    ap.add_argument("--reasoning", default="medium")
    ap.add_argument("--cheap", default="")
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--final", action="store_true", help="the label-free system, one row per question")
    ap.add_argument("--split", default="dev", choices=["dev", "held"])
    ap.add_argument("--out", default="out/route/final")
    a = ap.parse_args()
    if a.final:
        return final_run(a)
    OUT.mkdir(parents=True, exist_ok=True)
    runs = [load_run(p) for p in a.runs]
    sel = {r["qid"]: r for r in json.loads(Path(a.selected).read_text())}
    cases = [c for c in split()[0] if c.qid in sel]
    split_q = [c for c in cases if sel[c.qid]["clusters"] > 1]
    print(f"{len(cases)} questions, the runs disagree on {len(split_q)}", flush=True)
    corpus = Corpus()

    with httpx.Client(timeout=600) as client:

        def evidence(c):
            stats = dict(jev_secs=0.0, jev_requests=0, jev_tokens=0)
            docs = {d: c.query for r in runs for d in r[c.qid].get("read") or []}
            _, _, passages = agent.grade(client, corpus, c.query, docs, stats, deep=len(docs))
            return sorted(passages, key=lambda w: -passages[w])[: a.top], stats["jev_tokens"]

        with ThreadPoolExecutor(a.workers) as ex:
            ev = dict(zip([c.qid for c in split_q], ex.map(evidence, split_q)))
        print(f"evidence graded: {sum(t for _, t in ev.values()) / len(ev):,.0f} Jev tokens per question", flush=True)

        def read(c, shown: bool, stats: dict) -> bool:
            user = f"Question: {c.query}\n\nDocuments:\n" + "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(ev[c.qid][0]))
            if shown:
                user += SHOWN + "\n".join(f"- {r[c.qid].get('answer')}: {str(r[c.qid].get('notes'))[:600]}" for r in runs)
            try:
                p = agent.llm(client, [{"role": "system", "content": final.READER}, {"role": "user", "content": user}], stats)
                return bool(judge(client, c.query, p.get("answer"), c.answer, str(p.get("notes", "")))["correct"])
            except Exception as e:
                print(f"{c.qid:>5} ERR {e!r}", flush=True)
                return False

        res = {c.qid: {} for c in split_q}
        for arm, model, reasoning, shown in (("control", a.cheap, "", False), ("blind", a.strong, a.reasoning, False), ("shown", a.strong, a.reasoning, True)):
            agent.LLM_MODEL = model
            agent.REASONING = {"effort": reasoning} if reasoning else {"enabled": False}
            agent.LLM_DEADLINE = 180
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            with ThreadPoolExecutor(a.workers) as ex:
                for c, ok in zip(split_q, ex.map(lambda c: read(c, shown, stats), split_q)):
                    res[c.qid][arm] = ok
            n = max(stats["llm_calls"], 1)
            print(f"{arm}: {model} {reasoning or 'no reasoning'}; per read {stats['llm_in'] / n:,.0f} in, {stats['llm_out'] / n:,.0f} out, {stats['llm_secs'] / n:.0f} s", flush=True)
            (OUT / "results.json").write_text(json.dumps(res, indent=1))

    keep = [float(sel[c.qid]["voteown"]) for c in cases]
    print(f"\nkeep (best-of-{len(runs)}, no strong model)  {sum(keep) / len(keep):.3f}   on the {len(split_q)} split questions: {sum(sel[c.qid]['voteown'] for c in split_q)}   oracle there: {sum(sel[c.qid]['oracle'] for c in split_q)}")
    for arm in ("control", "blind", "shown"):
        routed = [float(res[c.qid][arm]) if c.qid in res else float(sel[c.qid]["voteown"]) for c in cases]
        _, lo, hi = paired_bootstrap(routed, keep)
        fixed = sum(res[q][arm] and not sel[q]["voteown"] for q in res)
        broke = sum(sel[q]["voteown"] and not res[q][arm] for q in res)
        print(f"{arm:8} {sum(routed) / len(routed):.3f}   vs keep {sum(routed) / len(routed) - sum(keep) / len(keep):+.3f} [{lo:+.3f}, {hi:+.3f}]   split questions right: {sum(res[q][arm] for q in res)}   fixed {fixed} broke {broke}")


if __name__ == "__main__":
    main()
