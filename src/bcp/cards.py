"""The corpus-wide card store that store.py / store2.py / store3.py tested on a subset: one index card per document
(subject, entities, facts), written by a small model from the page alone. No question is ever seen.

  data/cards*.jsonl  {"docid", "card"} per line, appended as written, so a rerun resumes. Seeded from out/store/cards.json.
                     Several writers can run at once, each with its own --out file; --reverse starts from the far end.

    uv run python -m bcp.cards --writer mistralai/ministral-8b-2512
    uv run python -m bcp.cards --writer deepseek/deepseek-v4-flash-0731 --reverse --out cards_b.jsonl
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp import agent
from bcp.corpus import Corpus
from bcp.data import DATA_DIR
from bcp.store import CARD

PATH = DATA_DIR / "cards.jsonl"


def load() -> dict[str, dict]:
    out = {}
    for p in sorted(DATA_DIR.glob("cards*.jsonl")):
        for line in p.read_text().splitlines():
            if line.endswith("}"):  # a writer may be mid-line
                r = json.loads(line)
                out[r["docid"]] = r["card"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", required=True)
    ap.add_argument("--threads", type=int, default=192)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--reverse", action="store_true")
    ap.add_argument("--out", default=PATH.name)
    a = ap.parse_args()
    agent.LLM_MODEL = a.writer
    agent._POOL = ThreadPoolExecutor(a.threads)  # llm() posts through this pool; its default 64 would be the cap
    corpus = Corpus()
    cards = load()
    seed = Path("out/store/cards.json")
    with (DATA_DIR / a.out).open("a") as f:
        if seed.exists():
            for d, c in json.loads(seed.read_text()).items():
                if c and d not in cards:
                    cards[d] = c
                    f.write(json.dumps({"docid": d, "card": c}) + "\n")
        todo = [d for d in (reversed(corpus.docids) if a.reverse else corpus.docids) if d not in cards][: a.limit or None]
        print(f"{len(cards)} cards on disk, {len(todo)} to write", flush=True)
        stats = dict(llm_retries=0, llm_secs=0.0, llm_calls=0, llm_in=0, llm_out=0)
        t0 = time.time()

        with httpx.Client(timeout=300, limits=httpx.Limits(max_connections=a.threads)) as client:

            def card(d):
                try:
                    return d, agent.llm(client, [{"role": "system", "content": CARD}, {"role": "user", "content": corpus.text[d][:12000]}], stats)
                except RuntimeError:
                    return d, None  # left out of the file: the next run retries it

            with ThreadPoolExecutor(a.threads) as ex:
                for i, (d, c) in enumerate(ex.map(card, todo)):
                    if c:
                        f.write(json.dumps({"docid": d, "card": c}) + "\n")
                    if i % 1000 == 999:
                        f.flush()
                        print(f"  {i + 1}/{len(todo)}  {(i + 1) / (time.time() - t0) * 60:.0f}/min  in {stats['llm_in'] / 1e6:.1f}M out {stats['llm_out'] / 1e6:.1f}M tokens", flush=True)
    print(f"done: {len(load())} cards", flush=True)


if __name__ == "__main__":
    main()
