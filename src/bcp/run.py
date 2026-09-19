"""Run the experiment: can Jev separate evidence docs from mined hard negatives?

Four arms over the SAME per-query candidate pool, all seeing the same truncated
text: jev (one noul per doc), bm25 (lexical baseline), length (-len of the
UNtruncated doc) and length_capped (-len after the cap). The bar Jev must clear
is max(bm25, length), not BM25 alone.
"""

import argparse
import dataclasses
import json
import time
from pathlib import Path

from bcp import bm25, data, jev, metrics

COST_PER_MTOK = 0.042
ARMS = ["jev", "bm25", "length", "length_capped"]
MIN_N_FOR_CONCLUSION = 30


def cap(case, doc_chars):
    docs = [dataclasses.replace(d, text=d.text[:doc_chars]) for d in case.docs]
    return dataclasses.replace(case, docs=docs)


def score_case(case, doc_chars, raw_lens=None):
    """-> (scores per arm, jev JevResult). `case` is already capped; `raw_lens`
    maps docid -> pre-truncation length (defaults to the capped length)."""
    raw_lens = raw_lens or {d.docid: len(d.text) for d in case.docs}
    jr = jev.score(case, doc_chars=doc_chars)
    return {
        "jev": jr.scores,
        "bm25": bm25.score(case),
        "length": {d.docid: -float(raw_lens[d.docid]) for d in case.docs},
        "length_capped": {d.docid: -float(len(d.text)) for d in case.docs},
    }, jr


def evaluate(rows, k=10):
    """rows: [{qid, labels, scores:{arm:[...]}}] -> summary dict."""
    per_query = {
        arm: {
            "auc": [metrics.auc(r["labels"], r["scores"][arm]) for r in rows],
            "ndcg": [metrics.ndcg_at(r["labels"], r["scores"][arm], k) for r in rows],
        }
        for arm in ARMS
    }
    summary = {}
    for arm in ARMS:
        pooled_labels = [lab for r in rows for lab in r["labels"]]
        pooled_scores = [s for r in rows for s in r["scores"][arm]]
        summary[arm] = {
            "pooled_auc": metrics.auc(pooled_labels, pooled_scores),
            "mean_auc": _mean_ci(per_query[arm]["auc"]),
            "mean_ndcg@10": _mean_ci(per_query[arm]["ndcg"]),
        }
    for baseline in ("bm25", "length"):
        for metric in ("auc", "ndcg"):
            diff, lo, hi = metrics.paired_bootstrap(
                per_query["jev"][metric], per_query[baseline][metric]
            )
            summary[f"paired_jev_minus_{baseline}_{metric}"] = {
                "diff": diff,
                "lo": lo,
                "hi": hi,
            }

    probs = [s for r in rows for s in r["scores"]["jev"]]
    labels = [lab for r in rows for lab in r["labels"]]
    err, bins = metrics.ece(labels, probs)
    summary["calibration"] = {
        "ece": err,
        "bins": bins,
        "positive_rate": sum(labels) / len(labels),
    }
    return summary, per_query


def _mean_ci(values):
    vals = [v for v in values if v is not None]
    lo, hi = metrics.bootstrap_ci(values)
    return {"mean": sum(vals) / len(vals), "lo": lo, "hi": hi, "n": len(vals)}


def _truncation(raw_cases, doc_chars):
    """Fraction of each class that the cap actually truncated."""
    out = {}
    for name, label in (("evidence", 1), ("negative", 0)):
        lens = [len(d.text) for c in raw_cases for d in c.docs if d.label == label]
        out[name] = {
            "n": len(lens),
            "median_chars": sorted(lens)[len(lens) // 2],
            "frac_truncated": sum(1 for x in lens if x > doc_chars) / len(lens),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", type=Path, default=Path("out"))
    ap.add_argument("--doc-chars", type=int, default=4000)
    ap.add_argument("--max-neg", type=int, default=20)
    a = ap.parse_args()

    raw_cases = data.load_cases(a.n, max_neg=a.max_neg)
    raw_lens = {d.docid: len(d.text) for c in raw_cases for d in c.docs}
    cases = [cap(c, a.doc_chars) for c in raw_cases]
    skipped = [c.qid for c in cases if len({d.label for d in c.docs}) < 2]
    cases = [c for c in cases if len({d.label for d in c.docs}) == 2]

    rows, jev_requests, jev_tokens, jev_secs = [], 0, 0, 0.0
    t0 = time.perf_counter()
    for c in cases:
        scores, jr = score_case(c, a.doc_chars, raw_lens)
        jev_requests += jr.requests
        jev_tokens += jr.tokens
        jev_secs += jr.secs
        rows.append(
            {
                "qid": c.qid,
                "query": c.query,
                "docids": [d.docid for d in c.docs],
                "labels": [d.label for d in c.docs],
                "scores": {arm: [scores[arm][d.docid] for d in c.docs] for arm in ARMS},
            }
        )
    wall = time.perf_counter() - t0

    summary, per_query = evaluate(rows)
    results = {
        "config": {
            "n_requested": a.n,
            "n_evaluated": len(rows),
            "skipped_single_class": skipped,
            "doc_chars": a.doc_chars,
            "max_neg": a.max_neg,
            "model": jev.JEV_MODEL,
            "truncation": _truncation([c for c in raw_cases if c.qid not in skipped], a.doc_chars),
        },
        "counts": {
            "docs": sum(len(r["labels"]) for r in rows),
            "positives": sum(sum(r["labels"]) for r in rows),
            "negatives": sum(len(r["labels"]) - sum(r["labels"]) for r in rows),
        },
        "jev_usage": {
            "network_requests": jev_requests,
            "input_tokens": jev_tokens,
            "cost_usd": jev_tokens / 1e6 * COST_PER_MTOK,
            "jev_secs": jev_secs,
            "mean_request_secs": jev_secs / jev_requests if jev_requests else None,
            "wall_secs": wall,
        },
        "summary": summary,
        "per_query": per_query,
        "rows": rows,
    }
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "results.json").write_text(json.dumps(results, indent=2))
    (a.out / "REPORT.md").write_text(report(results))
    print(f"wrote {a.out / 'results.json'} and {a.out / 'REPORT.md'}")


def report(res):
    c, s, u = res["config"], res["summary"], res["jev_usage"]
    n = res["counts"]
    underpowered = c["n_evaluated"] < MIN_N_FOR_CONCLUSION
    L = [
        "# BrowseComp-Plus: Jev vs BM25 on evidence-vs-hard-negative separation",
        "",
    ]
    if underpowered:
        L += [
            f"> **PLUMBING CHECK, NOT A MEASUREMENT.** Only {c['n_evaluated']} queries "
            f"(< {MIN_N_FOR_CONCLUSION}). The bootstrap intervals below are not 95% "
            "intervals at this n — under the null a paired CI excludes 0 roughly "
            "19% of the time at n=5. **No go/no-go conclusion may be drawn from "
            "this report.**",
            "",
        ]
    L += [
        f"{c['n_evaluated']} queries, {n['docs']} candidate docs "
        f"({n['positives']} evidence / {n['negatives']} hard negatives), "
        f"max_neg={c['max_neg']}, doc_chars={c['doc_chars']}, model={c['model']}.",
        f"Skipped (single-class pool): {len(c['skipped_single_class'])} "
        f"{c['skipped_single_class']}",
        "",
        "## Ranking (95% CI = percentile bootstrap over queries)",
        "",
        "| arm | mean per-query AUC | nDCG@10 | pooled AUC |",
        "| --- | --- | --- | --- |",
    ]
    for arm in ARMS:
        a_, d_ = s[arm]["mean_auc"], s[arm]["mean_ndcg@10"]
        L.append(f"| {arm} | {_ci(a_)} | {_ci(d_)} | {s[arm]['pooled_auc']:.3f} |")
    bar = max(s["bm25"]["mean_auc"]["mean"], s["length"]["mean_auc"]["mean"])
    L += [
        "",
        "Headline = **mean per-query AUC**. Pooled AUC is informational only: BM25 "
        "scores are not comparable across queries, so pooling them mixes scales.",
        "",
        f"The bar is **max(bm25, length) = {bar:.3f}**, not BM25 alone: the trivial "
        "length arm is often the stronger floor.",
        "",
        "## Paired Jev - baseline (same resampled queries, positive = Jev ahead)",
        "",
        "| baseline | metric | diff | 95% CI |",
        "| --- | --- | --- | --- |",
    ]
    for baseline in ("bm25", "length"):
        for metric, label in (("auc", "AUC"), ("ndcg", "nDCG@10")):
            p = s[f"paired_jev_minus_{baseline}_{metric}"]
            L.append(
                f"| {baseline} | {label} | {p['diff']:+.3f} | "
                f"[{p['lo']:+.3f}, {p['hi']:+.3f}] |"
            )
    if underpowered:
        L += ["", "(Intervals above are not valid at this n — see the banner.)"]

    L += [
        "",
        "## Jev cost and latency",
        "",
        f"- network requests: {u['network_requests']} (cache hits excluded)",
    ]
    if u["network_requests"]:
        L.append(
            f"- mean latency per request: {u['mean_request_secs']:.2f}s; "
            f"total network time {u['jev_secs']:.1f}s; wall {u['wall_secs']:.1f}s"
        )
    else:
        L.append(
            "- latency: not measured (fully cached run; wall "
            f"{u['wall_secs']:.1f}s is cache-read time, not Jev latency)"
        )
    L += [
        f"- input tokens: {u['input_tokens']:,} → ${u['cost_usd']:.4f} "
        f"at ${COST_PER_MTOK}/Mtok (counted over every response, cached or fresh: "
        "this is what a cold run costs)",
        "",
        "## Calibration of the noul probability",
        "",
        f"ECE = {s['calibration']['ece']:.3f} (10 equal-width bins, pooled over all docs)",
        "",
        f"**Measured at a positive rate of {s['calibration']['positive_rate']:.3f}** "
        f"because max_neg={c['max_neg']} caps the negatives. The dataset's natural "
        "per-query pool is ~6 evidence / ~76 negatives (~0.074), so this ECE describes "
        "an enriched pool and does not transfer to the deployment prevalence.",
        "",
        "| bin | n | mean prob | frac evidence |",
        "| --- | --- | --- | --- |",
    ]
    for b in s["calibration"]["bins"]:
        mp = "-" if b["mean_prob"] is None else f"{b['mean_prob']:.3f}"
        fp = "-" if b["frac_pos"] is None else f"{b['frac_pos']:.3f}"
        L.append(f"| [{b['lo']:.1f}, {b['hi']:.1f}) | {b['n']} | {mp} | {fp} |")

    t = c.get("truncation")
    L += ["", "## Caveats", ""]
    if t:
        L.append(
            f"- The cap removes most but not all of the length gap: evidence median "
            f"{t['evidence']['median_chars']:,} chars "
            f"({t['evidence']['frac_truncated']:.0%} truncated) vs negative median "
            f"{t['negative']['median_chars']:,} chars "
            f"({t['negative']['frac_truncated']:.0%} truncated). The `length` arm uses "
            "true pre-cap length and is the real floor; `length_capped` is the residual "
            "confound a scorer can still see."
        )
    L += [
        "- Each document is scored in its own request and its own state, so no score "
        "is conditional on batch-mates.",
        "- Positives are `evidence_docs` (gold ⊆ evidence). Negatives are the "
        "dataset's mined hard negatives, capped at max_neg per query and deduped "
        "against evidence, so BM25 near 0.5–0.7 is the expected baseline.",
        f"- Only {c['max_neg']} negatives per query are sampled out of ~76 available. "
        "ROC AUC is approximately invariant to uniform negative subsampling, so the "
        "AUC column transfers to the full pool; **nDCG@10 does not** — it falls by "
        "roughly half on the full pool (BM25: 0.354 → 0.185), so read nDCG@10 only as "
        "a within-report comparison between arms at this max_neg.",
        "- The answer string is never shown to any scorer.",
    ]
    return "\n".join(L) + "\n"


def _ci(m):
    return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]"


if __name__ == "__main__":
    main()
