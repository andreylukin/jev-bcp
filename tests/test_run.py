import json

import pytest

from bcp import run
from bcp.data import Case, Doc
from bcp.jev import JevResult


def mkcase(qid, pos=2, neg=4, chars=50):
    docs = [Doc(f"{qid}p{i}", f"evidence {i} " + "a" * chars, 1) for i in range(pos)]
    docs += [Doc(f"{qid}n{i}", f"noise {i} " + "b" * chars * 2, 0) for i in range(neg)]
    return Case(qid, "who won the 2002 prize?", "SECRET ANSWER", docs)


@pytest.fixture
def fake_jev(monkeypatch):
    """Perfect-but-not-certain scorer: evidence 0.8, negatives 0.2."""

    def score(case, *, doc_chars=4000):
        return JevResult({d.docid: 0.8 if d.label else 0.2 for d in case.docs}, 1, 1000, 0.5)

    monkeypatch.setattr(run.jev, "score", score)


def test_cap_truncates_every_doc_identically():
    capped = run.cap(mkcase("q1", chars=500), 100)
    assert {len(d.text) for d in capped.docs} == {100}


def test_perfect_scorer_gives_auc_1_and_beats_bm25(fake_jev):
    rows = []
    for qid in ("q1", "q2", "q3"):
        c = run.cap(mkcase(qid), 4000)
        scores, _ = run.score_case(c, 4000)
        rows.append(
            {
                "qid": qid,
                "labels": [d.label for d in c.docs],
                "scores": {a: [scores[a][d.docid] for d in c.docs] for a in run.ARMS},
            }
        )
    summary, per_query = run.evaluate(rows)
    assert summary["jev"]["mean_auc"]["mean"] == 1.0
    assert summary["jev"]["mean_ndcg@10"]["mean"] == 1.0
    assert summary["paired_jev_minus_bm25_auc"]["diff"] >= 0
    assert per_query["jev"]["auc"] == [1.0, 1.0, 1.0]


def test_end_to_end_writes_report_and_excludes_single_class(fake_jev, monkeypatch, tmp_path):
    cases = [mkcase("q1"), mkcase("q2"), mkcase("allpos", pos=3, neg=0)]
    monkeypatch.setattr(run.data, "load_cases", lambda n, max_neg=20: cases)
    monkeypatch.setattr(
        "sys.argv", ["run", "--n", "3", "--out", str(tmp_path), "--doc-chars", "200"]
    )
    run.main()

    res = json.loads((tmp_path / "results.json").read_text())
    assert res["config"]["skipped_single_class"] == ["allpos"]
    assert res["config"]["n_evaluated"] == 2
    assert res["counts"] == {"docs": 12, "positives": 4, "negatives": 8}
    assert res["jev_usage"]["cost_usd"] == pytest.approx(2000 / 1e6 * 0.042)
    assert len(res["rows"][0]["scores"]["jev"]) == 6

    md = (tmp_path / "REPORT.md").read_text()
    assert "mean per-query AUC" in md and "| jev |" in md and "ECE" in md
    assert "SECRET ANSWER" not in md
    assert "PLUMBING CHECK, NOT A MEASUREMENT" in md  # n=2 < 30


def test_report_renders_empty_calibration_bins(fake_jev):
    rows = []
    for qid in ("q1", "q2"):
        c = run.cap(mkcase(qid), 4000)
        scores, _ = run.score_case(c, 4000)
        rows.append(
            {
                "qid": qid,
                "labels": [d.label for d in c.docs],
                "scores": {a: [scores[a][d.docid] for d in c.docs] for a in run.ARMS},
            }
        )
    summary, _ = run.evaluate(rows)
    assert any(b["n"] == 0 for b in summary["calibration"]["bins"])
    md = run.report(
        {
            "config": {
                "n_evaluated": 2,
                "skipped_single_class": [],
                "doc_chars": 4000,
                "max_neg": 20,
                "model": "jev-latest",
            },
            "counts": {"docs": 12, "positives": 4, "negatives": 8},
            "jev_usage": {
                "network_requests": 2,
                "input_tokens": 100,
                "cost_usd": 0.1,
                "jev_secs": 1.0,
                "mean_request_secs": 0.5,
                "wall_secs": 1.0,
            },
            "summary": summary,
        }
    )
    assert "| - | - |" in md


def test_length_arm_uses_pre_cap_length(fake_jev):
    raw = mkcase("q1", chars=500)
    raw_lens = {d.docid: len(d.text) for d in raw.docs}
    capped = run.cap(raw, 100)
    scores, _ = run.score_case(capped, 100, raw_lens)
    assert set(scores["length_capped"].values()) == {-100.0}
    assert len(set(scores["length"].values())) > 1
    assert scores["length"] == {d.docid: -float(raw_lens[d.docid]) for d in raw.docs}
