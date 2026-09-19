"""Self-hosted small model on Modal: vLLM behind an OpenAI-compatible endpoint (scales to zero when idle).

    BCP_SLM=ibm-granite/granite-4.2-8b modal deploy modal_llm.py     # prints the URL; first start downloads weights
    uv run python -m bcp.recall --endpoint https://<workspace>--jev-bcp-llm-serve.modal.run/v1/chat/completions

The bearer key is data/slm_key (gitignored), created on first use.
"""

import os
import subprocess
from pathlib import Path

import modal

MODEL = os.environ.get("BCP_SLM", "ibm-granite/granite-4.2-8b")
KEY_FILE = Path(__file__).parent / "data" / "slm_key"

app = modal.App("jev-bcp-llm")
image = (
    # CUDA devel image: vLLM JIT-compiles kernels at startup and needs nvcc
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .pip_install("vllm", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/models/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "BCP_SLM": MODEL})
)
models = modal.Volume.from_name("jev-bcp-models", create_if_missing=True)
secret = modal.Secret.from_dict({"BCP_SLM_KEY": KEY_FILE.read_text().strip() if KEY_FILE.exists() else ""})


@app.function(image=image, gpu="L40S", volumes={"/models": models}, secrets=[secret], scaledown_window=300, timeout=3600, max_containers=1)
@modal.concurrent(max_inputs=128)
@modal.web_server(port=8000, startup_timeout=1500)
def serve():
    subprocess.Popen([
        "vllm", "serve", os.environ["BCP_SLM"], "--served-model-name", "slm", "--port", "8000",
        "--max-model-len", "65536", "--gpu-memory-utilization", "0.92", "--api-key", os.environ["BCP_SLM_KEY"],
    ])
