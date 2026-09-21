"""BrowseComp-Plus's own evaluation script (scripts_evaluation/evaluate_run.py: Qwen3-32B in vLLM, their prompt and
parser) over a run directory, on one Modal H200 (an H100 leaves too little KV cache for
the model's 40k context). Their repo, the run files and the decrypted ground truth are mounted
from this machine; the judged results come back into the run's eval directory.

    modal run modal_judge.py --bcp ../BrowseComp-Plus --runs out/submission/runs/jev-flash \\
        --ground-truth out/submission/browsecomp_plus_decrypted.jsonl --out out/submission/evals
"""

import subprocess
from pathlib import Path

import modal

app = modal.App("jev-bcp-judge")
image = (  # CUDA devel base: vLLM's FlashInfer backend compiles a kernel at start-up and needs nvcc
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential")
    .pip_install("vllm", "numpy", "tqdm")
    .env({"HF_HOME": "/data/hf"})
)
volume = modal.Volume.from_name("jev-bcp-data", create_if_missing=True)


@app.function(image=image, volumes={"/data": volume}, gpu="H200", timeout=3600)
def judge(bcp: dict[str, bytes], runs: dict[str, bytes], ground_truth: bytes) -> dict[str, bytes]:
    root = Path("/work/bcp")
    for name, blob in bcp.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(blob)
    run_dir = Path("/work/runs/run")
    run_dir.mkdir(parents=True)
    for name, blob in runs.items():
        (run_dir / name).write_bytes(blob)
    Path("/work/gt.jsonl").write_bytes(ground_truth)
    subprocess.run(["python", "scripts_evaluation/evaluate_run.py", "--input_dir", str(run_dir), "--ground_truth", "/work/gt.jsonl",
                    "--eval_dir", "/work/evals"], cwd=root, check=True)
    volume.commit()  # the Qwen3-32B download stays cached
    return {str(p.relative_to("/work/evals")): p.read_bytes() for p in Path("/work/evals").rglob("*") if p.is_file()}


@app.local_entrypoint()
def main(bcp: str, runs: str, ground_truth: str, out: str):
    root = Path(bcp)
    files = {str(p.relative_to(root)): p.read_bytes() for d in ("scripts_evaluation", "topics-qrels") for p in (root / d).rglob("*") if p.is_file()}
    run_files = {p.name: p.read_bytes() for p in Path(runs).glob("*.json")}
    got = judge.remote(files, run_files, Path(ground_truth).read_bytes())
    for name, blob in got.items():
        (Path(out) / name).parent.mkdir(parents=True, exist_ok=True)
        (Path(out) / name).write_bytes(blob)
    print(f"{len(got)} files -> {out}")
