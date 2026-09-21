"""Reason-ModernColBERT (149M, late interaction) search over the corpus, served from a Modal GPU.

The index is LightOn's own prebuilt FastPLAID index (HF dataset lightonai/browsecomp-plus-indexes, the one behind
their leaderboard rows): every document truncated to its first 512 tokens, 4-bit residuals, 42.5M token vectors.
Nothing is encoded here except queries ("[Q] " prefix, QUERY_LEN tokens, no query expansion: the model's config).

    modal run modal_colbert.py::prepare                       # once: index + model into the volume (~6 GB)
    modal run modal_colbert.py --queries q.json --out hits.json --k 1000
                                                              # q.json: {"key": "query text", ...} -> {"key": [docid, ...]}
From code:  modal.Cls.from_name("jev-bcp-colbert", "Searcher")().search.remote(["query", ...], 100)
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("jev-bcp-colbert")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("pylate>=1.2", "fast-plaid", "huggingface_hub>=0.25")
    .env({"HF_HOME": "/data/hf"})
)
volume = modal.Volume.from_name("jev-bcp-data", create_if_missing=True)
NAME = "Reason-ModernColBERT"
INDEX = f"/data/colbert/pylate/{NAME}"
QUERY_LEN = 256  # benchmark questions run to ~150 tokens; the config's 128 would cut them


@app.function(image=image, volumes={"/data": volume}, timeout=3600)
def prepare():
    from huggingface_hub import snapshot_download

    snapshot_download("lightonai/browsecomp-plus-indexes", repo_type="dataset", allow_patterns=f"pylate/{NAME}/*", local_dir="/data/colbert")
    snapshot_download(f"lightonai/{NAME}")
    volume.commit()
    print(sorted(p.name for p in Path(INDEX, "fast_plaid_index").iterdir()))


@app.cls(image=image, volumes={"/data": volume}, gpu="L4", timeout=1800, scaledown_window=120, min_containers=int(os.environ.get("COLBERT_WARM", "0")))
class Searcher:
    @modal.enter()
    def load(self):
        import pickle

        from fast_plaid import search
        from pylate import models

        self.model = models.ColBERT(f"lightonai/{NAME}", query_length=QUERY_LEN, device="cuda")
        self.plaid = search.FastPlaid(index=f"{INDEX}/fast_plaid_index")
        with open(f"{INDEX}/plaid_ids_to_documents_ids.pkl", "rb") as f:
            self.docid = pickle.load(f)

    @modal.method()
    def search(self, queries: list[str], k: int) -> list[list[str]]:
        import torch

        out = []
        for i in range(0, len(queries), 32):
            embs = self.model.encode(queries[i : i + 32], is_query=True, convert_to_tensor=True, show_progress_bar=False)
            embs = embs if isinstance(embs, torch.Tensor) else torch.nn.utils.rnn.pad_sequence(embs, batch_first=True)
            for hits in self.plaid.search(queries_embeddings=embs.float(), top_k=k):
                out.append([str(self.docid[pid]) for pid, _ in hits])
        return out


@app.local_entrypoint()
def main(queries: str, out: str, k: int = 1000):
    qs = json.loads(Path(queries).read_text())
    keys = list(qs)
    s = Searcher()
    chunks = [keys[i : i + 256] for i in range(0, len(keys), 256)]
    res = {}
    for chunk, hits in zip(chunks, s.search.map([[qs[x] for x in c] for c in chunks], kwargs={"k": k})):
        res.update(zip(chunk, hits))
    Path(out).write_text(json.dumps(res))
    print(len(res), "queries ->", out)
