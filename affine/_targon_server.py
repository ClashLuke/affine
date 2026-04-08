import os
import targon

image = (
    targon.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("vllm==0.10.2", "torch==2.8.0", "huggingface_hub==0.35.0", "hf_transfer")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "MODEL_NAME": os.environ.get("MODEL_NAME", ""),
        "MODEL_REVISION": os.environ.get("MODEL_REVISION", "main"),
    })
)

app = targon.App("affine-slot", image=image)


@app.function(resource=targon.Compute.H200_SMALL, max_replicas=1)
@targon.web_server(port=8000)
def serve():
    import subprocess
    import torch

    model = os.environ["MODEL_NAME"]
    revision = os.environ.get("MODEL_REVISION", "main")
    n_gpu = torch.cuda.device_count()
    subprocess.Popen([
        "vllm", "serve", model,
        "--revision", revision,
        "--served-model-name", model,
        "--host", "0.0.0.0",
        "--port", "8000",
        "--tensor-parallel-size", str(n_gpu),
    ])
