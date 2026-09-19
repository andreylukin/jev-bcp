"""Ranking + calibration metrics. Pure python, no deps."""

import math
import random
from statistics import mean


def auc(labels, scores):
    """ROC AUC via rank statistic; ties get average rank (0.5 credit).

    Returns None if labels are single-class (AUC undefined).
    """
    pairs = sorted(zip(scores, labels))
    n = len(pairs)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2 + 1  # 1-based average rank of the tied block
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1

    n_pos = sum(1 for _, lab in pairs if lab == 1)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    rank_sum = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def ndcg_at(labels, scores, k):
    """Binary-gain nDCG@k. Ties broken pessimistically (negatives first),
    so input order cannot inflate the result. None if no positives."""
    order = sorted(zip(scores, labels), key=lambda sl: (-sl[0], sl[1]))
    dcg = sum(lab / math.log2(rank + 2) for rank, (_, lab) in enumerate(order[:k]))
    n_pos = sum(labels)
    if n_pos == 0:
        return None
    idcg = sum(1 / math.log2(rank + 2) for rank in range(min(k, n_pos)))
    return dcg / idcg


def ece(labels, probs, bins=10):
    """Expected calibration error with equal-width bins over [0, 1].

    Returns (ece, bins) where each bin is a dict with lo, hi, n, mean_prob,
    frac_pos. Empty bins are included with n=0.
    """
    buckets = [[] for _ in range(bins)]
    for lab, p in zip(labels, probs):
        idx = min(int(p * bins), bins - 1)
        buckets[idx].append((lab, p))

    n = len(probs)
    out = []
    err = 0.0
    for i, bucket in enumerate(buckets):
        lo, hi = i / bins, (i + 1) / bins
        if bucket:
            conf = mean(p for _, p in bucket)
            acc = mean(lab for lab, _ in bucket)
            err += len(bucket) / n * abs(acc - conf)
        else:
            conf = acc = None
        out.append(
            {"lo": lo, "hi": hi, "n": len(bucket), "mean_prob": conf, "frac_pos": acc}
        )
    return err, out


def bootstrap_ci(per_query_values, n=1000, seed=0):
    """Percentile 95% CI of the mean, resampling QUERIES with replacement.
    None values (undefined for that query) are dropped first."""
    vals = [v for v in per_query_values if v is not None]
    rng = random.Random(seed)
    means = sorted(
        mean(vals[rng.randrange(len(vals))] for _ in vals) for _ in range(n)
    )
    return _pct(means, 0.025), _pct(means, 0.975)


def paired_bootstrap(a, b, n=1000, seed=0):
    """Mean of (a - b) per query with a 95% CI, resampling the same query
    indices for both arms. Queries where either value is None are dropped."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None]
    rng = random.Random(seed)
    means = sorted(
        mean(diffs[rng.randrange(len(diffs))] for _ in diffs) for _ in range(n)
    )
    return mean(diffs), _pct(means, 0.025), _pct(means, 0.975)


def _pct(sorted_vals, q):
    idx = min(max(int(round(q * (len(sorted_vals) - 1))), 0), len(sorted_vals) - 1)
    return sorted_vals[idx]
