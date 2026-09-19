"""Run the agent on Modal: the whole benchmark in minutes instead of an hour.

    modal run modal_app.py::prepare                                   # once: fill the volume (indexes, corpus, cases)
    modal run modal_app.py --model deepseek/deepseek-v4-flash-0731    # dev split (200 q)
    modal run modal_app.py --model ... --split held                   # the 630 held-out questions
    modal run modal_app.py --model ... --split all --out out/x        # all 830

    modal run modal_app.py --model mistralai/ministral-8b-2512 --roles   # the small-model pipeline (split.py)

Wall clock is set by API rate limits, not compute: a question sends ~500k tokens to Jev (250k tok/s
account limit) and ~55k to the LLM, so `--concurrency` questions in flight is the knob.
"""

import json
import os
from pathlib import Path

import modal

app = modal.App("jev-bcp")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("httpx>=0.27", "pyarrow>=17", "huggingface_hub>=0.25", "bm25s", "numpy", "rank-bm25")
    .env({"BCP_DATA": "/data", "HF_HOME": "/data/hf"})
    .add_local_python_source("bcp")
)
volume = modal.Volume.from_name("jev-bcp-data", create_if_missing=True)
secret = modal.Secret.from_dict({k: os.environ.get(k, "") for k in ("TYPESAFE_API_KEY", "OPENROUTER_API_KEY")})
PER_CONTAINER = 16  # questions in flight per container (threads; the work is network-bound)


@app.function(image=image, volumes={"/data": volume}, timeout=3600, memory=16384, cpu=8)
def prepare():
    """Download corpus + dense index into the volume's HF cache, build BM25, cache the 830 cases."""
    from bcp.corpus import Corpus
    from bcp.data import load_all
    from bcp.dense import Dense

    print(len(load_all()), "cases")
    print(len(Corpus().docids), "docs indexed")
    print(Dense().index.shape, "dense index")
    volume.commit()


@app.cls(image=image, volumes={"/data": volume}, secrets=[secret], timeout=900, memory=12288, cpu=4, max_containers=8)
@modal.concurrent(max_inputs=PER_CONTAINER)
class Worker:
    @modal.enter()
    def load(self):
        import httpx

        from bcp import jev
        from bcp.corpus import Corpus
        from bcp.data import load_all
        from bcp.dense import Dense

        jev.CACHE_DIR = Path("/tmp/jev_cache")  # never thousands of small files on the shared volume
        self.cases = {c.qid: c for c in load_all()}
        self.corpus = Corpus()
        self.dense = Dense()
        self.client = httpx.Client(timeout=120)

    @modal.method()
    def run(self, qid: str, model: str, reasoning: str, retriever: str, use_jev: bool, use_grid: bool, use_final: bool, use_excerpts: bool = False, recursive: bool = False, roles: bool = False) -> dict:
        from bcp import agent

        agent.LLM_MODEL = model
        agent.EXCERPTS = use_excerpts
        agent.RECURSIVE = recursive
        agent.SPLIT = roles
        agent.REASONING = {"effort": reasoning} if reasoning else {"enabled": False}
        dense = self.dense if retriever != "bm25" else None
        return agent.run_one(self.cases[qid], self.corpus, self.client, use_jev, dense, retriever, use_grid, use_final)


@app.local_entrypoint()
def main(model: str, split: str = "dev", retriever: str = "hybrid", reasoning: str = "", no_jev: bool = False, grid: bool = False, no_final: bool = False,
         excerpts: bool = False, recursive: bool = False, roles: bool = False, limit: int = 0, out: str = "", concurrency: int = 32):
    import time

    from bcp.agent import summarize
    from bcp.data import load_all
    from bcp.data import split as dev_held

    dev, held = dev_held()
    cases = {"dev": dev, "held": held, "all": load_all()}[split][: limit or None]
    out_dir = Path(out or f"out/modal_{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"  # resume, as in bcp.agent
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()] if rows_path.exists() else []
    done = {r["qid"] for r in rows}
    todo = [c.qid for c in cases if c.qid not in done]
    print(f"{len(done)} done, {len(todo)} to go")

    worker = Worker.with_options(max_containers=max(1, -(-concurrency // PER_CONTAINER)))()
    t0 = time.time()
    with rows_path.open("a") as f:
        for row in worker.run.map(todo, kwargs=dict(model=model, reasoning=reasoning, retriever=retriever, use_jev=not no_jev, use_grid=grid, use_final=not no_final, use_excerpts=excerpts, recursive=recursive, roles=roles),
                                  order_outputs=False, return_exceptions=True):
            if isinstance(row, Exception):  # a question over the 15-minute per-input timeout; recorded below
                continue
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()
            if len(rows) % 25 == 0:
                print(f"{len(rows)}/{len(cases)}  acc {sum(bool(r['correct']) for r in rows) / len(rows):.3f}  {time.time() - t0:.0f}s")
    gold = {c.qid: c.answer for c in cases}
    for qid in set(todo) - {r["qid"] for r in rows}:  # hung LLM calls: counted as wrong, never dropped
        rows.append({"qid": qid, "gold": gold[qid], "correct": False, "error": "timeout"})
    summary = summarize(rows, {"split": split, "jev": not no_jev, "grid": grid, "final": not no_final, "excerpts": excerpts, "recursive": recursive, "roles": roles, "retriever": retriever, "model": model,
                               "reasoning": reasoning or "off", "concurrency": concurrency, "wall_secs": time.time() - t0})
    (out_dir / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
