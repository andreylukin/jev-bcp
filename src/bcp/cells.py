"""Go/no-go for the program runtime: does a candidate x constraint grid, filled by Jev from passages the
agent actually read, tell right answers from wrong ones better than the notes verifier (AUC 0.84)?

  1. compile  LLM turns each question into constraints: claims with an {answer} slot.
  2. observe  one Jev request per read passage (state = passage, billed once); questions = every
              (candidate, constraint) claim as a noul. cell = max over passages (copies count once),
              candidate = min over constraints (the weakest constraint is the score).
  3. measure  candidates are the agent's own final answer (label = judged correct) and, where it was wrong,
              the gold answer: AUC over agent answers, and how often gold outscores the wrong answer.

    uv run python -m bcp.cells --run out/modal_dev_hybrid
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, grid, metrics
from bcp.corpus import Corpus
from bcp.data import load_all


def compile_question(client, query: str) -> dict:
    stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
    return {"constraints": grid.compile_question(lambda m: agent.llm(client, m, stats), query)}


def observe(client, passage, claims):
    return grid.observe(client, passage, claims)[0]


def fill(client, corpus, read: list[str], cons: list[str], cands: dict[str, str]) -> dict[str, float]:
    """cells["who|i"] = max over the read passages of Jev's support for constraint i with candidate `who`."""
    claims = {f"{who}|{i}": c.replace("{answer}", name) for who, name in cands.items() for i, c in enumerate(cons)}
    keys = {k: f"q{j}" for j, k in enumerate(claims)}
    cells = {k: 0.0 for k in claims}
    for d in read:
        obs = observe(client, corpus.window(d, " ".join(claims.values())), {keys[k]: c for k, c in claims.items()})
        for k in claims:
            cells[k] = max(cells[k], obs[keys[k]])
    return cells


PROPOSE = """From your research notes and the passages, list the 3 most plausible distinct answers to the question,
best first. Each must be a short exact answer (a name, title, date or number), not a description.
Reply with JSON: {"candidates": ["...", "...", "..."]}"""


def select(client, corpus, cases, rows, programs, out):
    """Propose-then-select: the LLM proposes 3 candidates, the grid picks one (mean cell), the official judge
    grades it. Compared with the agent's own answer on the same questions."""
    from bcp.judge import judge

    def one(r):
        cons = programs[r["qid"]]["constraints"]
        if not cons:
            return None
        q = cases[r["qid"]].query
        passages = "\n\n".join(f"[doc {d}]\n{corpus.window(d, q)}" for d in r["read"])
        stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
        p = agent.llm(client, [{"role": "system", "content": PROPOSE}, {"role": "user", "content": f"Question: {q}\n\nNotes: {r.get('notes', '')}\n\nPassages:\n{passages}"}], stats)
        names = [str(r["answer"])] + [str(c) for c in p.get("candidates", []) if c]
        cands = {f"c{j}": n for j, n in enumerate(dict.fromkeys(names))}  # c0 is the agent's own answer
        cells = fill(client, corpus, r["read"], cons, cands)
        mean = {w: sum(cells[f"{w}|{i}"] for i in range(len(cons))) / len(cons) for w in cands}
        pick = max(mean, key=mean.get)
        verdicts = {w: (bool(r["correct"]) if w == "c0" else judge(client, q, n, cases[r["qid"]].answer)["correct"]) for w, n in cands.items()}
        return {"qid": r["qid"], "cands": cands, "mean": mean, "pick": pick, "verdicts": verdicts, "gold": cases[r["qid"]].answer}

    with ThreadPoolExecutor(12) as pool:
        res = [x for x in pool.map(one, rows) if x]
    (out / "select.json").write_text(json.dumps(res, indent=1))
    n = len(res)
    print(f"{n} questions, mean {sum(len(x['cands']) for x in res) / n:.1f} distinct candidates")
    print(f"agent's own answer:           {sum(x['verdicts']['c0'] for x in res) / n:.3f}")
    print(f"grid picks among candidates:  {sum(x['verdicts'][x['pick']] for x in res) / n:.3f}")
    print(f"upper bound (any candidate):  {sum(any(x['verdicts'].values()) for x in res) / n:.3f}")
    print(f"fixed (wrong -> right): {sum(not x['verdicts']['c0'] and x['verdicts'][x['pick']] for x in res)}   "
          f"broken (right -> wrong): {sum(x['verdicts']['c0'] and not x['verdicts'][x['pick']] for x in res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/modal_dev_hybrid")
    ap.add_argument("--model", default="deepseek/deepseek-v4-flash-0731")
    ap.add_argument("--select", action="store_true", help="propose-then-select instead of the right/wrong AUC")
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    cases = {c.qid: c for c in load_all()}
    rows = [r for r in map(json.loads, (Path(a.run) / "rows.jsonl").read_text().splitlines()) if "error" not in r and r.get("answer")]
    corpus = Corpus()
    out = Path("out/cells")
    out.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=120) as client:
        prog_path = out / "programs.json"
        programs = json.loads(prog_path.read_text()) if prog_path.exists() else {}
        todo = [r["qid"] for r in rows if r["qid"] not in programs]
        with ThreadPoolExecutor(16) as pool:
            for qid, p in zip(todo, pool.map(lambda q: compile_question(client, cases[q].query), todo)):
                programs[qid] = p
        prog_path.write_text(json.dumps(programs, indent=1))

        def grid(r):
            cons = programs[r["qid"]]["constraints"]
            if not cons:
                return None
            cands = {"agent": str(r["answer"])} | ({} if r["correct"] else {"gold": cases[r["qid"]].answer})
            cells = fill(client, corpus, r["read"], cons, cands)
            score = {who: min(cells[f"{who}|{i}"] for i in range(len(cons))) for who in cands}
            return {"qid": r["qid"], "correct": bool(r["correct"]), "gold_read": r["gold_read"], "cands": cands, "cells": cells, "score": score, "constraints": cons}

        if a.select:
            return select(client, corpus, cases, rows, programs, out)
        with ThreadPoolExecutor(12) as pool:
            grids = [g for g in pool.map(grid, rows) if g]
    (out / "grids.json").write_text(json.dumps(grids, indent=1))

    y = [int(g["correct"]) for g in grids]
    print(f"{len(grids)} questions, {sum(y)} agent answers right; mean constraints {sum(len(g['constraints']) for g in grids) / len(grids):.1f}")
    print(f"AUC, agent answer right vs wrong:  min-cell {metrics.auc(y, [g['score']['agent'] for g in grids]):.3f}   "
          f"mean-cell {metrics.auc(y, [sum(v for k, v in g['cells'].items() if k.startswith('agent')) / len(g['constraints']) for g in grids]):.3f}   (notes verifier: 0.84)")
    wrong = [g for g in grids if not g["correct"]]
    for name, gs in (("all wrong answers", wrong), ("wrong, but every gold doc was read", [g for g in wrong if g["gold_read"] == 1])):
        if gs:
            print(f"{name}: gold outscores the agent's wrong answer in {sum(g['score']['gold'] > g['score']['agent'] for g in gs)}/{len(gs)}")


if __name__ == "__main__":
    main()
