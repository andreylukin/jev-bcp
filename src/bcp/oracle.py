"""Reader ceiling: hand the LLM the labelled evidence docs directly (perfect retrieval) and judge its answer.
If a reader cannot answer from the evidence, no retrieval improvement will fix it.

  --view full    every evidence doc, up to DOC_CHARS each
  --view window  the agent's view: corpus.window() (4000 chars chosen by term overlap with the question)
  --view hops    two small jobs instead of one big read: per doc, extract the facts that bear on the question;
                 then answer from the extracted facts alone (is the ceiling the model, or the size of the job?)

    uv run python -m bcp.oracle --model deepseek/deepseek-v4-flash-0731 --view full
"""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent
from bcp.corpus import Corpus
from bcp.data import split
from bcp.final import strict
from bcp.judge import judge

DOC_CHARS = 60_000
EXTRACT = """From the document, list the facts that bear on the research question: every name, date, place, number, title or
relationship that matches or could help check any part of it. Full sentences that name their entities; quote exact
names and figures. If the document has nothing relevant, return an empty list.
Reply with JSON: {"facts": ["...", "..."]}"""
SYSTEM = """Answer the research question using the documents provided; together they contain what is needed.
Work through the constraints step by step, then reply with one JSON object:
{"notes": "the chain of facts from the documents that determines the answer", "answer": "short exact answer"}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="slm")
    ap.add_argument("--reasoning", choices=["low", "medium", "high"])
    ap.add_argument("--view", choices=["full", "window", "hops"], default="full")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--samples", type=int, default=1, help=">1: sample several answers, Jev's strict check picks one")
    ap.add_argument("--only-wrong-in", help="an earlier oracle result file: run only the questions it got wrong")
    ap.add_argument("--endpoint", help="self-hosted OpenAI-compatible chat URL (modal_llm.py)")
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    if a.endpoint:
        agent.use_endpoint(a.endpoint)
    if a.reasoning:
        agent.REASONING = {"effort": a.reasoning}
    dev, _ = split()
    if a.only_wrong_in:
        wrong = {r["qid"] for r in json.loads(Path(a.only_wrong_in).read_text())["rows"] if not r["correct"]}
        dev = [c for c in dev if c.qid in wrong]
    corpus = Corpus()
    out = Path("out/oracle") / f"{a.model.replace('/', '_')}_{a.reasoning or 'off'}_{a.view}{'_x%d' % a.samples if a.samples > 1 else ''}{'_retry' if a.only_wrong_in else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=300) as client:

        def one(case):
            ids = [d.docid for d in case.docs]
            random.Random(case.qid).shuffle(ids)
            text = (lambda d: corpus.window(d, case.query)) if a.view == "window" else (lambda d: corpus.text[d][:DOC_CHARS])
            stats = dict(llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            try:
                if a.view == "hops":
                    facts = [agent.llm(client, [{"role": "system", "content": EXTRACT}, {"role": "user", "content": f"Question: {case.query}\n\nDocument:\n{text(d)}"}], stats).get("facts") or [] for d in ids]
                    docs = "\n\n".join(f"[doc {d}]\n" + "\n".join(f"- {f}" for f in fs) for d, fs in zip(ids, facts) if fs)
                else:
                    docs = "\n\n".join(f"[doc {d}]\n{text(d)}" for d in ids)
                msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": f"Question: {case.query}\n\nDocuments:\n{docs}"}]
                outs = [agent.llm(client, msgs, stats) for _ in range(a.samples)]
                if a.samples > 1:
                    findings = docs[:90_000]
                    outs.sort(key=lambda o: -strict(client, case.query, str(o.get("answer")), findings))
                r = outs[0]
                row = {"qid": case.qid, "answer": r.get("answer"), "gold": case.answer, "samples": [str(o.get("answer")) for o in outs], **stats}
                row.update(judge(client, case.query, row["answer"], case.answer, str(r.get("notes", ""))))
            except Exception as e:  # one bad question must not kill the probe; it counts as wrong
                row = {"qid": case.qid, "answer": None, "gold": case.answer, "correct": False, "error": repr(e)[:200], **stats}
            print(f"{row['qid']:>5} {'OK ' if row['correct'] else '-- '} {str(row.get('answer'))[:40]!r} / {row['gold'][:40]!r}", flush=True)
            return row

        with ThreadPoolExecutor(a.workers) as pool:
            rows = list(pool.map(one, dev))

    n = len(rows)
    summary = {"model": a.model, "reasoning": a.reasoning or "off", "view": a.view, "n": n,
               "accuracy": sum(bool(r["correct"]) for r in rows) / n, "errors": sum("error" in r for r in rows),
               "secs_mean": sum(r["llm_secs"] for r in rows) / n, "llm_in": sum(r["llm_in"] for r in rows) / n}
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
