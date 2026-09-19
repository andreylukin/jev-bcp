from dataclasses import dataclass

from bcp.bm25 import score, tokenize


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


def test_tokenize_lowercases_and_splits():
    assert tokenize("Queen Arwa, University-2002!") == ["queen", "arwa", "university", "2002"]


def test_ranks_matching_doc_first():
    case = Case(
        qid="1",
        query="queen arwa university cultural activities",
        docs=[
            Doc("a", "unrelated text about submarine engineering", 0),
            Doc("b", "Queen Arwa University holds annual cultural activities", 1),
            Doc("c", "a university somewhere", 0),
        ],
    )
    s = score(case)
    assert set(s) == {"a", "b", "c"}
    assert s["b"] > s["c"] > s["a"]


def test_scores_are_floats_one_per_doc():
    case = Case("2", "anything", [Doc("x", "anything at all", 1), Doc("y", "nothing", 0)])
    s = score(case)
    assert len(s) == 2
    assert all(isinstance(v, float) for v in s.values())
