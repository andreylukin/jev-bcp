"""Every check so far grades something the SAME reader wrote, so a misreading passes. Here a second reader answers
from the same passages without seeing the first answer; Jev says whether the two answers agree.

  A        the agent's answer from a finished dev run
  B        a fresh read of the final stage's passages, in reversed order, no first answer shown
  agree    Jev noul: do A and B name the same thing?
  resolve  on disagreement: (strict) Jev's strict check picks A or B; (third) a third read sees both and picks

    uv run python -m bcp.agree --run out/dev_base_v3 --model deepseek/deepseek-v4-flash-0731
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent, final, jev
from bcp.corpus import Corpus
from bcp.data import load_all
from bcp.judge import judge

SAME = "Do answer_a and answer_b name the same thing, so that if one is the correct answer to the question the other is too?"
PICK = """Two researchers gave different answers to the question. Check each constraint of the question against the documents
for both answers, then pick the one the documents support. Reply with JSON: {"notes": "the deciding facts", "answer": "one of the two answers, copied exactly"}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="out/dev_base_v3")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    agent.LLM_MODEL = a.model
    cases = {c.qid: c for c in load_all()}
    rows = [r for r in json.load(open(Path(a.run) / "results.json"))["rows"] if "error" not in r and r.get("answer")][: a.n or None]
    corpus = Corpus()

    with httpx.Client(timeout=300) as client:

        def one(r):
            case = cases[r["qid"]]
            q = case.query
            stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
            ask = lambda m: agent.llm(client, m, stats)
            parts = [(f"{d}#{i}", w) for d in r["read"] for i, w in enumerate(corpus.windows(d))]
            s, _, _ = jev.score_batched(client, q, parts)
            docs, size = [], 0
            for k, w in sorted(parts, key=lambda kw: -s[kw[0]]):  # the final stage's reader input
                if size + len(w) > final.CHARS:
                    break
                docs.append(w)
                size += len(w)
            show = lambda ds: "\n\n".join(f"[doc {i + 1}]\n{w}" for i, w in enumerate(ds))
            A = str(r["answer"])
            out = {"qid": r["qid"], "A": A, "A_ok": bool(r["correct"])}
            try:
                b = ask([{"role": "system", "content": final.READER}, {"role": "user", "content": f"Question: {q}\n\nDocuments:\n{show(docs[::-1])}"}])
                B = str(b.get("answer"))
                data, _ = jev._post(client, {"model": jev.JEV_MODEL, "state": {"question": q, "answer_a": A, "answer_b": B},
                                             "questions": {"same": {"type": "noul", "instructions": SAME}}}, os.environ["TYPESAFE_API_KEY"])
                out.update(B=B, same=data["answers"]["same"]["noul"], B_ok=bool(judge(client, q, B, case.answer, str(b.get("notes", "")))["correct"]))
                if out["same"] < 0.5:
                    findings = "\n\n".join(docs)
                    sa, sb = final.strict(client, q, A, findings), final.strict(client, q, B, findings)
                    out["strict_ok"] = out["A_ok"] if sa >= sb else out["B_ok"]
                    p = ask([{"role": "system", "content": PICK}, {"role": "user", "content": f"Question: {q}\n\nAnswer 1: {A}\nAnswer 2: {B}\n\nDocuments:\n{show(docs)}"}])
                    out["third"] = str(p.get("answer"))
                    out["third_ok"] = bool(judge(client, q, out["third"], case.answer, str(p.get("notes", "")))["correct"])
            except Exception as e:
                out["error"] = repr(e)[:200]
            print(f"{r['qid']:>5} A {'OK' if out['A_ok'] else '--'}  B {'OK' if out.get('B_ok') else '--'}  same {out.get('same', -1):.2f}"
                  + (f"  -> strict {'OK' if out['strict_ok'] else '--'}  third {'OK' if out['third_ok'] else '--'}" if "third_ok" in out else ""), flush=True)
            return out

        with ThreadPoolExecutor(a.workers) as ex:
            res = [x for x in ex.map(one, rows)]
    Path("out/agree").mkdir(parents=True, exist_ok=True)
    Path("out/agree/results.json").write_text(json.dumps(res, indent=1))
    ok = [x for x in res if "error" not in x]
    agree, split = [x for x in ok if x["same"] >= 0.5], [x for x in ok if x["same"] < 0.5]
    n = len(ok)
    print(f"{n} answers ({len(res) - n} errors): A right {sum(x['A_ok'] for x in ok) / n:.3f}   B right {sum(x['B_ok'] for x in ok) / n:.3f}")
    print(f"agree    {len(agree):3}: A right {sum(x['A_ok'] for x in agree) / max(1, len(agree)):.3f}")
    print(f"disagree {len(split):3}: A right {sum(x['A_ok'] for x in split) / max(1, len(split)):.3f}  B right {sum(x['B_ok'] for x in split) / max(1, len(split)):.3f}"
          f"  either right {sum(x['A_ok'] or x['B_ok'] for x in split) / max(1, len(split)):.3f}")
    print(f"wrong A answers caught by disagreement: {sum(not x['A_ok'] for x in split)}/{sum(not x['A_ok'] for x in ok)}")
    for k in ("strict_ok", "third_ok"):
        print(f"policy agree->A, disagree->{k[:-3]:6}: {(sum(x['A_ok'] for x in agree) + sum(x[k] for x in split)) / n:.3f}")
    print(f"ceiling (disagree -> whichever is right): {(sum(x['A_ok'] for x in agree) + sum(x['A_ok'] or x['B_ok'] for x in split)) / n:.3f}")


if __name__ == "__main__":
    main()
