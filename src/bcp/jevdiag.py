"""Where does a jevlog run lose its questions? The official judge grades EVERY proposed binding's answer, so a miss is
either 'never proposed' (the search / propose side) or 'proposed, not picked' (the program / Jev side).

    uv run python -m bcp.jevdiag out/jevlog_v2a out/jevlog_v2b
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from bcp.data import load_all
from bcp.judge import judge


def main():
    cases = {c.qid: c for c in load_all()}
    rows = [r for d in sys.argv[1:] for r in map(json.loads, (Path(d) / "rows.jsonl").read_text().splitlines())]
    with httpx.Client(timeout=120) as client:

        def one(r):
            if "error" in r or r["correct"] or not r.get("bindings"):
                return r, None
            c = cases[r["qid"]]
            key = next(k for k, v in r["bindings"][0]["binding"].items() if v == r["answer"])
            verdicts = [bool(judge(client, c.query, str(b["binding"].get(key)), c.answer)["correct"]) if b["binding"].get(key) else False for b in r["bindings"]]
            return r, verdicts

        with ThreadPoolExecutor(8) as ex:
            res = list(ex.map(one, rows))
    n = len(rows)
    right = sum(bool(r["correct"]) for r in rows)
    errors = sum("error" in r for r in rows)
    empty = sum("error" not in r and not r.get("bindings") for r in rows)
    lost = [(r, v) for r, v in res if v is not None]
    unpicked = [(r, v) for r, v in lost if any(v)]
    print(f"{n} questions: right {right} | error {errors} | no binding proposed {empty} | never proposed {len(lost) - len(unpicked)} | proposed, not picked {len(unpicked)}")
    for r, v in unpicked:
        i = v.index(True)
        print(f"  {r['qid']:>5} picked score {r['bindings'][0]['score']:.2f}  right binding rank {i + 1}/{len(v)} score {r['bindings'][i]['score']:.2f}  "
              f"facts picked {[round(x, 2) for x in r['bindings'][0]['facts'].values()]} right {[round(x, 2) for x in r['bindings'][i]['facts'].values()]}")


if __name__ == "__main__":
    main()
