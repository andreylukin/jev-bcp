"""Dense retrieval over the benchmark authors' prebuilt Qwen3-Embedding-8B index of the corpus
(HF dataset Tevatron/browsecomp-plus-indexes, qwen3-embedding-8b/corpus.shard*.pkl: Tevatron encode
output, a tuple (float32 [n,4096] L2-normalized embeddings, list of docids)).

Query encoding follows the authors' recipe
(https://github.com/texttron/BrowseComp-Plus/blob/main/scripts_build_index/qwen3-embed.md and
searcher/searchers/faiss_searcher.py --task-prefix): QUERY_PREFIX + query, eos (last-token) pooling,
L2-normalized, query_max_len 512, inner product (= cosine). Documents were encoded with no prefix.
Queries are embedded by the same model served on OpenRouter; pooling/eos/normalization happen
server-side and were checked by re-embedding corpus docs (cosine 0.9999 with the index vectors).
"""

import glob
import os
import pickle
import time

import httpx
import numpy as np
from huggingface_hub import snapshot_download

QUERY_PREFIX = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
MODEL = "qwen/qwen3-embedding-8b"
URL = "https://openrouter.ai/api/v1/embeddings"


def top_k(index: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    """Exact inner-product search: [q,k] row indices into `index`, best first."""
    scores = queries @ index.T
    k = min(k, index.shape[0])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def embed(queries: list[str]) -> np.ndarray:
    for attempt in range(4):
        try:
            r = httpx.post(
                URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                # nebius is the provider the doc-vector compatibility check was run against
                json={"model": MODEL, "input": [QUERY_PREFIX + q for q in queries], "provider": {"only": ["nebius"]}},
                timeout=30,
            )
            if r.status_code == 200:
                data = sorted(r.json()["data"], key=lambda d: d["index"])
                return np.array([d["embedding"] for d in data], dtype=np.float32)
        except httpx.TransportError:
            pass
        time.sleep(2**attempt)
    raise RuntimeError(f"embed: retries exhausted ({r.status_code}: {r.text[:200]})")


class Dense:
    def __init__(self):
        path = snapshot_download(
            "Tevatron/browsecomp-plus-indexes", repo_type="dataset", allow_patterns="qwen3-embedding-8b/*.pkl"
        )
        embs, self.docids = [], []
        for f in sorted(glob.glob(path + "/qwen3-embedding-8b/*.pkl")):
            with open(f, "rb") as fh:
                e, ids = pickle.load(fh)
            embs.append(e)
            self.docids += ids
        self.index = np.concatenate(embs)

    def search_many(self, queries: list[str], k: int) -> list[list[str]]:
        return [[self.docids[i] for i in row] for row in top_k(self.index, embed(queries), k)]

    def search(self, query: str, k: int) -> list[str]:
        return self.search_many([query], k)[0]
