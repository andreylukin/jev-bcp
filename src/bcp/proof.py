"""Does a step-by-step proof, checked by Jev, tell right answers from wrong ones better than the whole-answer
strict check (AUC 0.90)? Offline over a finished dev run; labels are the official judge's verdicts.

  prove   the reader lists the question's constraints and, for each, a step: claim + doc + VERBATIM quote
  code    a quote that is not in its document scores 0 (free; catches invented evidence)
  jev     per step, two nouls: does the passage around the quote state the claim; does the claim establish
          the constraint for the proposed answer. cell = stated * establishes
  score   mean / min / product of the constraint cells (a constraint with no step is 0)

    uv run python -m bcp.proof --run out/dev_base_v3 --model deepseek/deepseek-v4-flash-0731 --n 40
"""

import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev, metrics
from bcp.corpus import Corpus
from bcp.data import load_all

CONTEXT = 600  # chars of passage kept either side of the quote for Jev
PROVE = """You are given a research question, a proposed answer and numbered documents. Write the proof that the answer is
right, or as much of it as the documents support. Do not change the answer.
1. List every constraint of the question, each as one short statement.
2. For each constraint give one step: the claim (a full sentence that names the entities, no pronouns), the doc number,
   and a quote copied VERBATIM from that doc (one or two sentences) that states the claim.
   If no document supports a constraint, give the step with doc and quote set to null.
Reply with JSON: {"constraints": ["..."], "steps": [{"constraint": 0, "claim": "...", "doc": 3, "quote": "..."}]}"""


def squash(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dev_base_v3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    cases = {c.qid: c for c in load_all()}
    rows = [r for r in json.load(open(Path(a.run) / "results.json"))["rows"] if "error" not in r and r.get("answer")]
    wrong = [r for r in rows if not r["correct"]][: a.n // 2]  # wrong answers are ~20% of a run: take them all, and as many right ones
    rows = wrong + [r for r in rows if r["correct"]][: a.n - len(wrong)]
    corpus = Corpus()

    with httpx.Client(timeout=300) as client:

        def one(r):
            q = cases[r["qid"]].query
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            parts = [(f"{d}#{i}", w) for d in r["read"] for i, w in enumerate(corpus.windows(d))]
            s, _, _ = jev.score_batched(client, q, parts)
            docs, size = [], 0
            for k, w in sorted(parts, key=lambda kw: -s[kw[0]]):  # the final stage's reader input
                if size + len(w) > final.CHARS:
                    break
                docs.append(w)
                size += len(w)
            shown = "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(docs))
            try:
                p = agent.llm(client, [{"role": "system", "content": PROVE}, {"role": "user", "content": f"Question: {q}\n\nProposed answer: {r['answer']}\n\nDocuments:\n{shown}"}], stats)
            except Exception:
                p = {}
            cons = [str(c) for c in p.get("constraints") or []]
            steps = [x for x in p.get("steps") or [] if isinstance(x, dict) and isinstance(x.get("constraint"), int) and 0 <= x["constraint"] < len(cons)]
            questions, verbatim = {}, 0
            for i, x in enumerate(steps):
                doc = docs[x["doc"] - 1] if isinstance(x.get("doc"), int) and 1 <= x["doc"] <= len(docs) else ""
                at = squash(doc).find(squash(str(x.get("quote") or ""))) if x.get("quote") and doc else -1
                if at < 0:
                    continue
                verbatim += 1
                flat = re.sub(r"\s+", " ", doc)
                passage = flat[max(0, at - CONTEXT) : at + len(str(x["quote"])) + CONTEXT]
                questions[f"stated{i}"] = {"type": "noul", "instructions": f"Does the passage explicitly state the claim, about the same entities?\n\nClaim: {x.get('claim')}\n\nPassage: {passage}"}
                questions[f"serves{i}"] = {"type": "noul", "instructions": f"If the claim is true, does it establish this constraint of the question for the proposed answer?\n\nClaim: {x.get('claim')}\n\nConstraint: {cons[x['constraint']]}"}
            cells = [0.0] * len(cons)
            if questions:
                data, _ = jev._post(client, {"model": jev.JEV_MODEL, "state": {"question": q, "proposed_answer": str(r["answer"])}, "questions": questions}, os.environ["TYPESAFE_API_KEY"])
                for i, x in enumerate(steps):
                    if f"stated{i}" in questions:
                        v = data["answers"][f"stated{i}"]["noul"] * data["answers"][f"serves{i}"]["noul"]
                        cells[x["constraint"]] = max(cells[x["constraint"]], v)
            prod = 1.0
            for c in cells:
                prod *= c
            out = {"qid": r["qid"], "correct": bool(r["correct"]), "strict": r.get("final_score", 0.0), "constraints": len(cons), "steps": len(steps), "verbatim": verbatim,
                   "mean": sum(cells) / len(cells) if cells else 0.0, "min": min(cells, default=0.0), "product": prod if cells else 0.0, "cells": cells, "proof": p}
            print(f"{r['qid']:>5} {'OK' if out['correct'] else '--'}  strict {out['strict']:.2f}  proof mean {out['mean']:.2f} min {out['min']:.2f}  {verbatim}/{len(steps)} quotes verbatim, {len(cons)} constraints", flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(one, rows))
    Path("out/proof").mkdir(parents=True, exist_ok=True)
    Path("out/proof/results.json").write_text(json.dumps(res, indent=1))
    y = [int(x["correct"]) for x in res]
    print(f"{len(res)} answers, {sum(y)} right; {sum(x['constraints'] for x in res) / len(res):.1f} constraints, quotes verbatim {sum(x['verbatim'] for x in res) / max(1, sum(x['steps'] for x in res)):.2f}")
    for k in ("strict", "mean", "min", "product"):
        print(f"  AUC right vs wrong, {k:8} {metrics.auc(y, [x[k] for x in res]):.3f}")
    print(f"  AUC, strict * proof mean  {metrics.auc(y, [x['strict'] * x['mean'] for x in res]):.3f}")


if __name__ == "__main__":
    main()
