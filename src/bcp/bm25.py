"""BM25 baseline: score a case's query against that case's own candidate docs."""

import re

from rank_bm25 import BM25Okapi

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def score(case) -> dict[str, float]:
    docs = case.docs
    bm25 = BM25Okapi([tokenize(d.text) for d in docs])
    scores = bm25.get_scores(tokenize(case.query))
    return {d.docid: float(s) for d, s in zip(docs, scores)}
