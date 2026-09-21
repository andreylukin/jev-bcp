"""The leaderboard run directory for the published no-frontier-model system (route.py --final, flash reasoning low as the
routed reader): one JSON per query in BrowseComp-Plus's run format, for their scripts_evaluation/evaluate_run.py.

  output             "Explanation: ...\\nExact Answer: ...\\nConfidence: N%". Confidence is label-free: the dev-measured
                     accuracy of the answer's route (all four trajectories agree / the routed read), so the judge's
                     calibration column measures the agreement signal
  retrieved_docids   the union of the pages the four trajectories READ. The runs did not keep every page their
                     searches returned, so the Recall column under-states retrieval (the best single run alone retrieved
                     94.1% of evidence pages; the pages read cover 83.7%)
  tool_call_counts   search ROUNDS summed over the four trajectories (each round sends up to 8 queries)

    uv run python -m bcp.export --out out/submission/runs/jev-flash
"""

import argparse
import json
import re
from pathlib import Path

from bcp.data import load_all

SYSTEM = {  # split -> (routed rows, the four trajectories)
    "dev": ("out/route/test_final", [f"out/cb_{i}" for i in range(1, 5)]),
    "held": ("out/publish/held_cheap", [f"out/held_cb_{i}" for i in range(1, 5)]),
}
DEV_SUMMARY = "out/route/test_final/results.json"
MODEL = "deepseek/deepseek-v4-flash-0731"


def rows(path: str) -> dict[str, dict]:
    return {r["qid"]: r for r in map(json.loads, Path(path, "rows.jsonl").read_text().splitlines())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ground_truth", default="", help="also write the decrypted ground truth their script reads (stays local)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = json.loads(Path(DEV_SUMMARY).read_text())
    conf = {"agree": round(100 * dev["agree_accuracy"]), "strong": round(100 * dev["strong_accuracy"])}
    print(f"confidence from dev: {conf}", flush=True)
    n = 0
    for routed_path, run_paths in SYSTEM.values():
        routed, runs = rows(routed_path), [rows(p) for p in run_paths]
        for qid, r in routed.items():
            trajs = [x.get(qid) or {} for x in runs]
            answer = r.get("answer") if r.get("answer") not in (None, "None") else ""
            notes = next((str(t.get("notes") or "") for t in trajs if r["source"] == "agree" and str(t.get("answer")) == str(answer)), "")
            notes = re.sub(r"\[[^\]]*\]", "", notes).strip()  # "[doc 3]" would be scored as a citation of docid 3
            text = (f"Explanation: {notes}\n" if notes else "") + f"Exact Answer: {answer}\nConfidence: {conf[r['source']]}%"
            run = {
                "query_id": qid,
                "tool_call_counts": {"search": sum(t.get("rounds") or 0 for t in trajs)},
                "status": "completed",
                "retrieved_docids": sorted(set().union(*[set(t.get("read") or []) for t in trajs])),
                "result": [{"type": "output_text", "output": text}],
                "metadata": {"model": MODEL},
            }
            (out / f"{qid}.json").write_text(json.dumps(run, ensure_ascii=False))
            n += 1
    print(f"{n} run files -> {out}", flush=True)
    if a.ground_truth:
        with open(a.ground_truth, "w") as f:
            for c in load_all():
                f.write(json.dumps({"query_id": c.qid, "query": c.query, "answer": c.answer}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
