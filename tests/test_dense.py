import numpy as np

from bcp import dense


def test_top_k_orders_by_inner_product():
    index = np.array([[1, 0], [0, 1], [0.6, 0.8], [-1, 0]], dtype=np.float32)
    queries = np.array([[1, 0], [0, 1]], dtype=np.float32)
    assert dense.top_k(index, queries, 3).tolist() == [[0, 2, 1], [1, 2, 0]]


def test_top_k_clamps_k():
    index = np.eye(2, dtype=np.float32)
    assert dense.top_k(index, index[:1], 10).tolist() == [[0, 1]]


def test_search_many_maps_docids_and_prefixes_queries(monkeypatch):
    d = dense.Dense.__new__(dense.Dense)
    d.index = np.eye(3, dtype=np.float32)
    d.docids = ["a", "b", "c"]
    monkeypatch.setattr(dense, "embed", lambda qs: np.array([[0, 1, 0], [0.1, 0.2, 0.9]], dtype=np.float32))
    assert d.search_many(["x", "y"], 2) == [["b", "a"], ["c", "b"]]
    assert d.search("x", 1) == ["b"]
