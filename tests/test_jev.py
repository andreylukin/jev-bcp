import json
from dataclasses import dataclass

from bcp.jev import build_requests, parse_response


@dataclass
class Doc:
    docid: str
    text: str
    label: int


@dataclass
class Case:
    qid: str
    query: str
    docs: list


def case(n=5, chars=100, qid="q1"):
    return Case(qid, "who won?", [Doc(f"doc{i}", f"body{i} " + "x" * chars, i % 2) for i in range(n)])


def test_one_request_per_doc_in_order():
    c = case()
    reqs = build_requests(c)
    assert [docid for _, docid in reqs] == [d.docid for d in c.docs]
    for body, _ in reqs:
        assert body["state"]["question"] == c.query
        assert set(body["questions"]) == {"q"}
        q = body["questions"]["q"]
        assert q["type"] == "noul"
        assert set(q["criteria"]) == {"true", "false"}


def test_each_body_carries_exactly_one_document():
    c = case()
    for (body, docid), doc in zip(build_requests(c), c.docs):
        assert body["state"]["document"] == doc.text
        assert docid == doc.docid


def test_no_docid_or_label_leaks_into_body():
    c = case()
    for body, _ in build_requests(c):
        blob = json.dumps(body)
        for d in c.docs:
            assert d.docid not in blob


def test_truncation_applies_identically():
    c = case(n=3, chars=9000)
    reqs = build_requests(c, doc_chars=100)
    assert {len(body["state"]["document"]) for body, _ in reqs} == {100}


def test_parse_response_reads_the_noul():
    assert parse_response({"answers": {"q": {"type": "noul", "noul": 0.9}}}) == 0.9
