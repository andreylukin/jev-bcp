import math

from bcp.metrics import auc, bootstrap_ci, ece, ndcg_at, paired_bootstrap


def test_auc_perfect_and_inverted():
    labels = [1, 1, 0, 0]
    assert auc(labels, [0.9, 0.8, 0.2, 0.1]) == 1.0
    assert auc(labels, [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auc_hand_computed():
    # pos scores 3,1 ; neg scores 2,0 -> pairs: 3>2,3>0,1<2,1>0 = 3/4
    assert auc([1, 1, 0, 0], [3, 1, 2, 0]) == 0.75


def test_auc_all_ties_is_half():
    assert auc([1, 0, 1, 0], [0.5] * 4) == 0.5


def test_auc_partial_ties():
    # pos 2,1 ; neg 1,0 -> 1 + 1 + 0.5 + 1 = 3.5 of 4
    assert auc([1, 1, 0, 0], [2, 1, 1, 0]) == 0.875


def test_auc_single_class_is_none():
    assert auc([1, 1, 1], [0.3, 0.6, 0.9]) is None
    assert auc([0, 0], [0.3, 0.6]) is None


def test_ndcg_perfect_and_worst():
    labels = [1, 1, 0, 0]
    assert ndcg_at(labels, [4, 3, 2, 1], 10) == 1.0
    # single positive at rank 4 -> (1/log2(5)) / 1
    assert ndcg_at([1, 0, 0, 0], [1, 4, 3, 2], 10) == 1 / math.log2(5)


def test_ndcg_k_truncates():
    assert ndcg_at([1, 0, 0], [1, 3, 2], 2) == 0.0


def test_ndcg_ties_pessimistic():
    # all tied, 1 positive, k=2: positive sorts last -> 0
    assert ndcg_at([1, 0, 0], [0.5, 0.5, 0.5], 2) == 0.0


def test_ndcg_no_positives_is_none():
    assert ndcg_at([0, 0], [0.9, 0.1], 10) is None


def test_ece_perfectly_calibrated_is_zero():
    labels = [1, 0, 1, 0]
    probs = [1.0, 0.0, 1.0, 0.0]
    err, bins = ece(labels, probs)
    assert err == 0.0
    assert len(bins) == 10
    assert bins[0]["n"] == 2 and bins[9]["n"] == 2
    assert bins[5]["n"] == 0 and bins[5]["mean_prob"] is None


def test_ece_worst_case():
    err, _ = ece([0, 0], [1.0, 1.0])
    assert err == 1.0


def test_ece_hand_computed():
    # bin 0.9-1.0: probs .9,.9 acc .5 -> |0.5-0.9| = 0.4, weight 2/4
    # bin 0.0-0.1: probs .1,.1 acc .5 -> 0.4, weight 2/4
    err, _ = ece([1, 0, 1, 0], [0.9, 0.9, 0.1, 0.1])
    assert abs(err - 0.4) < 1e-12


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = [0.5 + 0.01 * i for i in range(40)]
    lo, hi = bootstrap_ci(vals, n=500, seed=1)
    m = sum(vals) / len(vals)
    assert lo < m < hi
    assert (lo, hi) == bootstrap_ci(vals, n=500, seed=1)


def test_bootstrap_ci_constant_values():
    assert bootstrap_ci([0.7] * 10, n=100) == (0.7, 0.7)


def test_bootstrap_ci_drops_none():
    assert bootstrap_ci([0.3, None, 0.3], n=50) == (0.3, 0.3)


def test_paired_bootstrap_sign_and_exclusion():
    a = [0.8] * 20
    b = [0.5] * 20
    diff, lo, hi = paired_bootstrap(a, b, n=200)
    assert abs(diff - 0.3) < 1e-12
    assert lo > 0


def test_paired_bootstrap_drops_query_if_either_is_none():
    diff, lo, hi = paired_bootstrap([0.9, None, 0.9], [0.4, 0.1, 0.4], n=50)
    assert diff == 0.5 and (lo, hi) == (0.5, 0.5)
