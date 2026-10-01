#!/usr/bin/env python3
"""
setup.py — local environment setup for AIP-Speech.

Uses the active virtual environment, or creates and uses ./.venv. Creates the
folders the pipeline writes to, checks that the environment has a CUDA build of
PyTorch (a prerequisite, see requirements.txt), installs requirements.txt
without letting pip replace that torch, and checks that PyTorch sees a GPU.

Usage (Windows, Linux — same command everywhere):
  python setup.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"

# transformers 5.12.1 (requirements.txt) disables itself on older torch
MIN_TORCH = (2, 4)

DIRS = [
    "data/bg",
    "data/speech_asr", "data/speech_kws",
    "data/esc50", "data/ms_snsd", "data/musan", "data/noisex",
    "descriptors", "itembanks", "manifests", "inference",
    "results/scores", "results/figures", "results/tables",
    "checks/inspection",
    "models/cache",
]

# run inside the target interpreter, which may be a different venv
TORCH_PROBE = """
import json, torch, torchvision
gpu = torch.cuda.is_available()
print(json.dumps({"torch": torch.__version__, "torchvision": torchvision.__version__,
                  "cuda": torch.version.cuda,
                  "gpu": torch.cuda.get_device_name(0) if gpu else None}))
"""


def venv_python(venv_dir: Path) -> Path:
    """Path to the python executable inside a venv, for this platform."""
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def ensure_venv() -> Path:
    """Create .venv if missing. Return the path to its python executable."""
    if not VENV_DIR.exists():
        print(f"[setup] Creating virtual environment at {VENV_DIR} ...")
        venv.create(VENV_DIR, with_pip=True)
    else:
        print(f"[setup] Using existing virtual environment at {VENV_DIR}")
    return venv_python(VENV_DIR)


def torch_status(python_exe: str) -> dict | None:
    """torch / torchvision versions, CUDA build and visible GPU; None if not importable."""
    r = subprocess.run([python_exe, "-c", TORCH_PROBE], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout.strip().splitlines()[-1])


def torch_version(status: dict) -> tuple[int, int]:
    major, minor = status["torch"].split("+")[0].split(".")[:2]
    return int(major), int(minor)


def main() -> None:
    print("=== AIP-Speech environment setup ===")
    if sys.version_info < (3, 10):
        print(f"[ERROR] Python 3.10 or newer is required (found {sys.version.split()[0]}).")
        sys.exit(1)

    # install into the active venv if there is one, else into ./.venv
    already_in_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV")
    python_exe = sys.executable if already_in_venv else str(ensure_venv())

    print("\n--- Creating Directories ---")
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {d}")

    print("\n--- Checking PyTorch ---")
    status = torch_status(python_exe)
    if status is None or status["cuda"] is None or torch_version(status) < MIN_TORCH:
        found = ("torch/torchvision not importable" if status is None else
                 f"torch {status['torch']}, torchvision {status['torchvision']}, "
                 f"CUDA build: {status['cuda'] or 'no (CPU-only)'}")
        print(f"[ERROR] This environment needs a CUDA build of PyTorch first "
              f"(torch >= {MIN_TORCH[0]}.{MIN_TORCH[1]} and its matching torchvision); found: {found}.\n"
              f"Pick the command for your GPU and driver at https://pytorch.org/get-started/locally/ "
              f"and run it with this environment's python, for example:\n"
              f"  \"{python_exe}\" -m pip install torch==2.5.1 torchvision==0.20.1 "
              f"--index-url https://download.pytorch.org/whl/cu124\n"
              f"Then run python setup.py again.")
        sys.exit(1)
    print(f"torch {status['torch']} (CUDA {status['cuda']}), torchvision {status['torchvision']}")

    print("\n--- Installing Dependencies ---")
    requirements_file = ROOT / "requirements.txt"
    if not requirements_file.exists():
        print("[ERROR] requirements.txt not found!")
        sys.exit(1)
    # constraints pin the installed torch/torchvision, so a conflicting
    # requirement fails loudly instead of pulling another (possibly CPU-only) torch
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"torch=={status['torch']}\ntorchvision=={status['torchvision']}\n")
        constraints = f.name
    try:
        print("Upgrading pip...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        print(f"Installing packages from {requirements_file.name}...")
        subprocess.run([python_exe, "-m", "pip", "install", "-r", str(requirements_file),
                        "-c", constraints], check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        sys.exit(1)
    finally:
        os.unlink(constraints)

    print("\n--- Checking the GPU ---")
    status = torch_status(python_exe)
    if status is None:
        print(f"[ERROR] torch/torchvision stopped importing after the install; see\n"
              f"  \"{python_exe}\" -c \"import torch, torchvision\"")
        sys.exit(1)
    if not status["gpu"]:
        print("[ERROR] PyTorch sees no CUDA GPU. Stage 7 (inference) runs the models on one;\n"
              "check the NVIDIA driver (nvidia-smi) and CUDA_VISIBLE_DEVICES, and that the\n"
              "PyTorch build's CUDA version is supported by the driver. The other stages\n"
              "run on the CPU.")
        sys.exit(1)
    print(f"GPU: {status['gpu']} (torch {status['torch']}, CUDA {status['cuda']})")

    print("\n=== Setup Completed Successfully! ===")
    print(f"Hugging Face & Torch cache directory: {ROOT / 'models' / 'cache'}")
    print("google/gemma-3n-E4B-it is gated: accept its licence on the Hugging Face Hub and\n"
          "log in (hf auth login) or set HF_TOKEN before Stage 7.")

    if not already_in_venv:
        activate = r".\.venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
        print(f"\nActivate the virtual environment with:\n  {activate}")


if __name__ == "__main__":
    main()
