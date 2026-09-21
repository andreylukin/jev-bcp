"""Given N independent full-pipeline runs of a question, can a label-free selector beat the single run, and how
close does it get to the "any run right" oracle? (Six READS of one evidence set had no voting headroom; these are
whole TRAJECTORIES: different searches, different evidence.)

  phase A  (--runs) finished dev runs are the trajectories: same 200 questions, same model, official judge, different
           configurations. Free except for the Jev calls below.
  phase B  (--gen) N fresh trajectories per question that differ in how the query writer is told to search
           (STYLES), each with its own evidence pool; then the same selectors.

  cluster  answers are grouped by Jev's "same thing?" noul (agree.py's question), exact matches first
  majority the biggest cluster; ties go to the earlier run (the baseline is listed first)
  own      the answer whose own run's strict check scored highest (`final_score`)
  voteown  the cluster with the largest SUM of its members' own strict scores (agreement-weighted)
  cross    every cluster's answer is strict-checked against ONE evidence set: the union of the trajectories' pools
           (phase A: the pool saved in out/jevlog_seeds), notes-free, so no run marks its own homework
  vote*x   cluster size x cross grade
  oracle   any trajectory right

    uv run python -m bcp.bestofn --runs out/dev_base_v3 out/modal_dev_passages out/dev_excerpts out/modal_dev_final
    uv run python -m bcp.bestofn --gen --model deepseek/deepseek-v4-flash-0731 --n 40

Result (dev 200): 5 existing runs: single 80.0%, own / voteown 85.5 / 85.0%, majority 82.0%, cross 80.5% (null), oracle 92.0%.
4 fresh flash + ColBERT runs: single 80.0, voteown 85.5% (+5.5 [+2.5, +9.0], fixed 12, broke 1), oracle 91.5%; all four agree
on 145 questions, right 94.5%. Prompt-style diversity (phase B) adds nothing over reruns; hard core 0 of 60 trajectories.
voteown is what route.py builds on. Reports: out/bestofn/report_*.txt.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev
from bcp.agree import SAME
from bcp.data import split
from bcp.metrics import bootstrap_ci, paired_bootstrap

OUT = Path("out/bestofn")
STYLES = {
    "base": "",
    "entity": "\nSearch style for this run: entity first. Find the NAME of the most identifiable person, work, place or organisation "
              "the question describes, then search that name together with each remaining constraint.",
    "constraint": "\nSearch style for this run: one constraint at a time. Take the constraints in the order given; for each, write "
                  "queries that would list every candidate satisfying it alone, then intersect the candidates in your notes.",
    "rare": "\nSearch style for this run: rarest detail first. Start from the most unusual, specific detail in the question (an odd "
            "number, date, phrase or event) and search it verbatim and paraphrased before anything else.",
}


def same(client, q: str, a: str, b: str) -> float:
    if a.strip().lower() == b.strip().lower():
        return 1.0
    data, _ = jev._post(client, {"model": jev.JEV_MODEL, "state": {"question": q, "answer_a": a, "answer_b": b},
                                 "questions": {"same": {"type": "noul", "instructions": SAME}}}, os.environ["TYPESAFE_API_KEY"])
    return data["answers"]["same"]["noul"]


def cluster(client, q: str, answers: list[str]) -> list[list[int]]:
    """Greedy: an answer joins the first cluster whose first member Jev calls the same thing."""
    groups: list[list[int]] = []
    for i, a in enumerate(answers):
        if not a:
            continue
        for g in groups:
            if same(client, q, answers[g[0]], a) >= 0.5:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def findings(pool: dict[str, float]) -> str:
    top, size = [], 0
    for p in sorted(pool, key=lambda p: -pool[p]):
        if size + len(p) > final.CHARS:
            break
        top.append(p)
        size += len(p)
    return "\n\n".join(top)


def select(client, q: str, trajs: list[dict], pool: dict[str, float]) -> dict:
    """`trajs`: [{answer, correct, final_score}] in run order, the baseline first. -> selector -> correct?"""
    answers = [str(t["answer"] or "") if t.get("answer") not in (None, "None") else "" for t in trajs]
    groups = cluster(client, q, answers)
    if not groups:
        return {k: False for k in ("single", "majority", "own", "voteown", "cross", "votecross", "oracle")} | {"clusters": 0, "top_share": 0.0}
    ok = lambda g: bool(trajs[g[0]]["correct"])  # one judgement per cluster: its first member's
    f = findings(pool)
    x = {id(g): final.strict(client, q, answers[g[0]], f) if f else 0.0 for g in groups}
    own = max(range(len(trajs)), key=lambda i: (trajs[i].get("final_score") or 0.0) if answers[i] else -1)
    big = max(groups, key=len)
    return {
        "single": bool(trajs[0]["correct"]),
        "majority": ok(big),
        "own": bool(trajs[own]["correct"]),
        "voteown": ok(max(groups, key=lambda g: sum(trajs[i].get("final_score") or 0.0 for i in g))),
        "cross": ok(max(groups, key=lambda g: x[id(g)])),
        "votecross": ok(max(groups, key=lambda g: len(g) * x[id(g)])),
        "oracle": any(bool(t["correct"]) for t in trajs),
        "clusters": len(groups), "top_share": len(big) / len(trajs),
        "detail": [{"answer": answers[g[0]], "n": len(g), "cross": round(x[id(g)], 3), "ok": ok(g)} for g in groups],
    }


def report(res: list[dict], name: str, header: str) -> None:
    n = len(res)
    lines = [header, f"{n} questions"]
    base = [float(r["single"]) for r in res]
    for k in ("single", "majority", "own", "voteown", "cross", "votecross", "oracle"):
        v = [float(r[k]) for r in res]
        lo, hi = bootstrap_ci(v)
        d, dlo, dhi = paired_bootstrap(v, base)
        lines.append(f"{k:10} {sum(v) / n:.3f} [{lo:.3f}, {hi:.3f}]   vs single {d:+.3f} [{dlo:+.3f}, {dhi:+.3f}]   "
                     f"fixed {sum(a and not b for a, b in zip(v, base)):.0f} broke {sum(b and not a for a, b in zip(v, base)):.0f}")
    uni = [r for r in res if r["top_share"] == 1.0]
    lines.append(f"all trajectories agree: {len(uni)} questions, right {sum(r['majority'] for r in uni) / max(1, len(uni)):.3f}; "
                 f"otherwise {n - len(uni)} questions, majority right {sum(r['majority'] for r in res if r['top_share'] < 1) / max(1, n - len(uni)):.3f}")
    text = "\n".join(lines)
    print(text)
    (OUT / f"report_{name}.txt").write_text(text + "\n")
    (OUT / f"results_{name}.json").write_text(json.dumps(res, indent=1))


def load_run(path: str) -> dict[str, dict]:
    p = Path(path)
    if (p / "rows.jsonl").exists():
        return {r["qid"]: r for r in map(json.loads, (p / "rows.jsonl").read_text().splitlines())}
    out = {}
    for f in p.glob("*.json"):  # a jevlog seeds directory: the agent stage
        s = json.loads(f.read_text())
        out[f.stem] = {"qid": f.stem, "answer": s["answers"][0] if s["answers"] else None, "correct": s["agent_ok"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=[], help="phase A: finished dev runs, the baseline first")
    ap.add_argument("--gen", action="store_true", help="phase B: generate one trajectory per style")
    ap.add_argument("--model", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--qids", default="", help="file with one qid per line (phase B sample)")
    ap.add_argument("--styles", default=",".join(STYLES))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev, _ = split()
    if a.qids:
        keep = set(Path(a.qids).read_text().split())
        dev = [c for c in dev if c.qid in keep]
    dev = dev[: a.n or None]

    if not a.gen:
        runs = [load_run(r) for r in a.runs]
        seeds = Path("out/jevlog_seeds")
        with httpx.Client(timeout=300) as client:
            def one(case):
                trajs = [r.get(case.qid) or {"answer": None, "correct": False} for r in runs]  # a missing or errored row is a wrong answer
                pool = json.loads((seeds / f"{case.qid}.json").read_text())["pool"]
                return {"qid": case.qid, **select(client, case.query, trajs, pool)}
            with ThreadPoolExecutor(a.workers) as ex:
                res = list(ex.map(one, dev))
        report(res, a.name or "A", "phase A: " + " ".join(a.runs))
        return

    from bcp.corpus import Corpus
    from bcp.dense import Dense
    from bcp.judge import judge

    agent.LLM_MODEL = a.model
    styles = a.styles.split(",")
    corpus, dense = Corpus(), Dense()
    base_system = agent.SYSTEM
    tdir = OUT / "traj"
    tdir.mkdir(exist_ok=True)
    with httpx.Client(timeout=300) as client:
        def traj(case, style):
            path = tdir / f"{case.qid}_{style}.json"
            if path.exists():
                return json.loads(path.read_text())
            sink: dict = {}
            try:
                # agent.SYSTEM is module state, so one style runs at a time (see the loop below)
                row = agent.solve(case, corpus, client, True, dense, sink=sink)
                row.update(judge(client, case.query, row["answer"], case.answer, row["notes"]))
                top = dict(sorted(sink["pool"].items(), key=lambda kv: -kv[1])[:24])
                row["pool"] = top
            except Exception as e:
                row = {"qid": case.qid, "answer": None, "correct": False, "error": repr(e)[:200], "pool": {}}
            path.write_text(json.dumps(row))
            print(f"{case.qid:>5} {style:10} {'OK ' if row['correct'] else '-- '} {str(row.get('answer'))[:50]!r}", flush=True)
            return row

        rows = {}
        for s in styles:
            agent.SYSTEM = base_system + STYLES[s]
            with ThreadPoolExecutor(a.workers) as ex:
                rows[s] = list(ex.map(lambda c: traj(c, s), dev))
        agent.SYSTEM = base_system
        for s in styles:
            print(f"{s:10} {sum(bool(r['correct']) for r in rows[s])}/{len(dev)}")

        def one(i):
            case = dev[i]
            trajs = [rows[s][i] for s in styles]
            pool: dict[str, float] = {}
            for t in trajs:
                for w, g in t.get("pool", {}).items():
                    pool[w] = max(g, pool.get(w, 0.0))
            return {"qid": case.qid, **select(client, case.query, trajs, pool)}
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, range(len(dev))))
    cost = [sum(rows[s][i].get("llm_in", 0) * 0.14e-6 + rows[s][i].get("llm_out", 0) * 0.28e-6 + rows[s][i].get("jev_tokens", 0) * 0.042e-6 for s in styles) for i in range(len(dev))]
    report(res, a.name or "B", f"phase B: styles {styles}, model {a.model}; ~${sum(cost) / max(1, len(cost)):.3f}/question for all trajectories (LLM price assumed $0.14/$0.28 per M)")


if __name__ == "__main__":
    main()
