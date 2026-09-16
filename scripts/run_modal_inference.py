import modal
import subprocess
import os

# GPU image: CUDA base plus everything the pipeline imports
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.10")
    .env({"FORCE_BUILD": "phi4-fix-v3"})
    .apt_install("git", "build-essential", "ffmpeg")
    .pip_install("packaging", "ninja", "wheel", "setuptools")
    .pip_install("torch==2.5.1", "torchaudio==2.5.1", "torchvision==0.20.1")
    .pip_install(
        "timm",
        "transformers==4.48.2",
        "accelerate",
        "bitsandbytes",
        "soundfile",
        "librosa",
        "pyloudnorm",
        "numpy",
        "scipy",
        "openai-whisper",
        "jiwer",
        "pandas",
        "pyarrow",
        "backoff",
        "datasets",
        "huggingface_hub",
        "statsmodels",
        "matplotlib",
        "seaborn",
        "tqdm",
        "requests",
        "peft==0.11.1",
        "loguru",
        "omegaconf",
        "conformer",
        "diffusers",
        "torchdyn",
        "decord",
        "blobfile",
        "deepspeed",
        "easydict",
        "fire",
        "hyperpyyaml",
        "immutabledict",
        "sacrebleu",
        "jsonlines",
        "validators",
        "sty",
        "colorama",
        "ujson",
        "cairosvg",
        "wget",
        "gdown",
        "sentencepiece"
    )
    .run_commands("CC=gcc CXX=g++ pip install flash-attn --no-build-isolation")
)

LOCAL_AIP_SPEECH_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

image = image.add_local_dir(
    LOCAL_AIP_SPEECH_DIR,
    remote_path="/workspace",
    ignore=[
        ".venv",
        ".git",
        "inference",
        "inference_backup",
        "models",
        "data",
        "data_subset",
        "checks",
        "__pycache__"
    ]
)

# only the 49 MB subset the active manifests actually reference
image = image.add_local_dir(
    os.path.join(LOCAL_AIP_SPEECH_DIR, "data_subset", "data"),
    remote_path="/workspace/data"
)

app = modal.App("aip-speech-inference")

# model weights, so a rerun doesn't re-download them
cache_volume = modal.Volume.from_name("aip-models-cache", create_if_missing=True)
# inference output, pulled down to the local machine as the run progresses
inference_volume = modal.Volume.from_name("aip-inference-out", create_if_missing=True)

hf_secret = modal.Secret.from_dict({"HF_TOKEN": os.environ["HF_TOKEN"]}) if os.environ.get("HF_TOKEN") else None
secrets = [hf_secret] if hf_secret else []

@app.function(
    image=image,
    gpu="L4", 
    timeout=86400, # 24 hours
    volumes={
        "/workspace/models/cache/hub": cache_volume,
        "/workspace/inference": inference_volume,
    },
    secrets=secrets,
)
def run_inference(model: str = "all", task: str = "all"):
    import sys
    import time

    os.chdir("/workspace")

    cmd = ["python", "scripts/05_inference.py", "--model", model, "--task", task]

    print(f"Running command: {' '.join(cmd)}")

    proc = subprocess.Popen(cmd)

    # commit as we go, otherwise nothing is visible to sync until the run ends
    while True:
        ret = proc.poll()
        if ret is not None:
            break
        try:
            inference_volume.commit()
        except Exception as e:
            print(f"Volume commit failed: {e}")
        time.sleep(300)  # commit every 5 minutes
    
    inference_volume.commit()
    
    if proc.returncode != 0:
        print(f"Error: inference script failed with exit code {proc.returncode}", file=sys.stderr)
        raise subprocess.CalledProcessError(proc.returncode, cmd)
    else:
        print("Inference completed successfully!")

@app.local_entrypoint()
def main(model: str = "all", task: str = "all"):
    import time
    import threading
    import subprocess

    timestamp = int(time.time())
    # per-run folder, so a sync never overwrites another model's local data
    sync_dir = f"inference_sync_{model}_{timestamp}"
    print(f"Starting Modal run on L4 GPU. model={model}, task={task}")
    print(f"Local sync directory for this run: {sync_dir}/")
    
    def sync_volume():
        print(f"Starting periodic volume sync to local ./{sync_dir} directory (every 5 mins)...")
        os.makedirs(sync_dir, exist_ok=True)
        # pull only the model being run, not the whole volume
        remote_path = f"/{model}" if model != "all" else "/"
        while True:
            time.sleep(300)
            print(f"Syncing {remote_path} from Modal volume to local {sync_dir}...")
            try:
                subprocess.run(["modal", "volume", "get", "aip-inference-out", remote_path, f"{sync_dir}/", "--force"], check=False)
                print("Sync complete.")
            except Exception as e:
                print(f"Sync failed: {e}")
                
    # daemon, so it dies with main rather than hanging the process
    sync_thread = threading.Thread(target=sync_volume, daemon=True)
    sync_thread.start()

    try:
        run_inference.remote(model, task)
    finally:
        print("Run finished or interrupted. Final sync...")
        remote_path = f"/{model}" if model != "all" else "/"
        subprocess.run(["modal", "volume", "get", "aip-inference-out", remote_path, f"{sync_dir}/", "--force"], check=False)
