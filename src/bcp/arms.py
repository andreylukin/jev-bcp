"""Which Jev request shape should the agent use? Measured on the labelled evidence/hard-negative pools.

  single   state={question, document}, one noul                       (the AUC experiment's shape)
  batch    state={question}, one noul PER DOC with the doc text inside that noul's instructions.
           Questions in a request are evaluated independently against the shared state (docs:
           primitives "Ask multiple questions together", cookbooks/skill_suggestion `fits::{name}`),
           so unlike several docs in one state, no doc's score can depend on its batch-mates.
  skim     batch shape, but each doc is cut to SKIM chars (progressive disclosure: skim wide, then
           read the shortlist properly - cookbooks/skill_suggestion).

    uv run python -m bcp.arms --n 50
"""

import argparse
import time

import httpx

from bcp import jev, metrics
from bcp.corpus import WINDOW
from bcp.data import load_cases

SKIM = 600


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    cases = load_cases(a.n)
    rows = {k: [] for k in ("single", "batch", "skim")}
    with httpx.Client(timeout=120) as client:
        for c in cases:
            labels = {d.docid: d.label for d in c.docs}
            full = [(d.docid, d.text[:WINDOW]) for d in c.docs]
            t0 = time.perf_counter()
            r = jev.score(c, doc_chars=WINDOW)
            runs = {"single": (r.scores, len(full), r.tokens, time.perf_counter() - t0)}
            for arm, docs in (("batch", full), ("skim", [(d, t[:SKIM]) for d, t in full])):
                t0 = time.perf_counter()
                runs[arm] = (*jev.score_batched(client, c.query, docs), time.perf_counter() - t0)
            for arm, (scores, reqs, toks, secs) in runs.items():
                ids = list(labels)
                rows[arm].append(dict(
                    auc=metrics.auc([labels[i] for i in ids], [scores[i] for i in ids]),
                    ndcg=metrics.ndcg_at([labels[i] for i in ids], [scores[i] for i in ids], 10),
                    scores=scores, reqs=reqs, toks=toks, secs=secs,
                ))
    single = rows["single"]
    for arm, rs in rows.items():
        aucs = [r["auc"] for r in rs]
        lo, hi = metrics.bootstrap_ci(aucs)
        d, dlo, dhi = metrics.paired_bootstrap(aucs, [r["auc"] for r in single])
        n = len(rs)
        print(f"{arm:7} AUC {sum(aucs)/n:.3f} [{lo:.3f},{hi:.3f}]  vs single {d:+.3f} [{dlo:+.3f},{dhi:+.3f}]  "
              f"nDCG@10 {sum(r['ndcg'] for r in rs)/n:.3f}  req/q {sum(r['reqs'] for r in rs)/n:.1f}  "
              f"tok/q {sum(r['toks'] for r in rs)/n:,.0f}  wall/q {sum(r['secs'] for r in rs)/n:.2f}s (cache-cold only)")
    # the cascade: skim everything, read the top K properly. How much evidence survives the skim cut?
    for k in (5, 8, 12):
        rec = []
        for c, s in zip(cases, rows["skim"]):
            ev = {d.docid for d in c.docs if d.label == 1}
            top = set(sorted(s["scores"], key=lambda i: -s["scores"][i])[:k])
            rec.append(len(ev & top) / min(len(ev), k))
        print(f"skim top-{k}: evidence recall {sum(rec)/len(rec):.2f} (of min(|evidence|,{k}))")


if __name__ == "__main__":
    main()
