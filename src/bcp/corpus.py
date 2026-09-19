"""The 100k-document BrowseComp-Plus corpus behind a local BM25 index (built once, cached in data/)."""

import glob
import re

import bm25s
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download

from bcp.data import DATA_DIR

INDEX_DIR = DATA_DIR / "bm25_index"
WINDOW = 4000
SNIP = 400


def excerpts(text: str) -> list[str]:
    """Sentence-aligned pieces of at most ~SNIP chars."""
    out, cur = [], ""
    for s in re.split(r"(?<=[.!?])\s+|\n+", text):
        if cur and len(cur) + len(s) > SNIP:
            out.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    return [e for e in out + [cur] if e]


def _rows() -> list[dict]:
    path = snapshot_download(
        "Tevatron/browsecomp-plus-corpus", repo_type="dataset", allow_patterns="data/*.parquet"
    )
    rows = []
    for f in sorted(glob.glob(path + "/data/*.parquet")):
        rows += pq.ParquetFile(f).read(columns=["docid", "text"]).to_pylist()
    return rows


class Corpus:
    def __init__(self):
        rows = _rows()
        self.docids = [r["docid"] for r in rows]
        self.text = {r["docid"]: r["text"] for r in rows}
        if INDEX_DIR.exists():
            self.bm25 = bm25s.BM25.load(INDEX_DIR)
        else:
            self.bm25 = bm25s.BM25()
            self.bm25.index(bm25s.tokenize([r["text"] for r in rows], stopwords="en"))
            self.bm25.save(INDEX_DIR)

    def search(self, query: str, k: int) -> list[str]:
        idx, _ = self.bm25.retrieve(bm25s.tokenize(query, stopwords="en", show_progress=False), k=k, show_progress=False)
        return [self.docids[i] for i in idx[0]]

    def window(self, docid: str, query: str) -> str:
        """The WINDOW-char slice of the doc sharing the most distinct terms with the query
        (Jev reads 4000 chars; evidence in a long page is often past the head)."""
        text = self.text[docid]
        if len(text) <= WINDOW:
            return text
        terms = set(re.findall(r"\w{3,}", query.lower()))
        best, best_n = 0, -1
        for start in range(0, len(text), WINDOW // 2):
            n = len(terms & set(re.findall(r"\w{3,}", text[start : start + WINDOW].lower())))
            if n > best_n:
                best, best_n = start, n
        return text[best : best + WINDOW]

    def windows(self, docid: str, limit: int = 12) -> list[str]:
        """Every WINDOW-char slice of the doc (small overlap), at most `limit`: Jev picks which one to read."""
        text = self.text[docid]
        return [text[i : i + WINDOW] for i in range(0, max(len(text) - 500, 1), WINDOW - 500)][:limit]

