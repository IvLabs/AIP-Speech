#!/usr/bin/env python3
"""
setup.py — single cross-platform environment setup for AIP-Speech.

Creates .venv if it doesn't exist, installs requirements.txt into it,
and creates all data/cache/results directories used by the pipeline.

Usage (Windows, Linux, macOS — same command everywhere):
  python setup.py
"""
import os
import sys
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


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


def main() -> None:
    print("=== AIP-Speech environment setup ===")

    # If already running inside some venv, install into that one directly.
    # Otherwise, create/reuse ./.venv and install into it.
    already_in_venv = sys.prefix != sys.base_prefix or os.environ.get("VIRTUAL_ENV")
    python_exe = sys.executable if already_in_venv else str(ensure_venv())

    requirements_file = ROOT / "requirements.txt"

    print("\n--- Installing Dependencies ---")
    try:
        print("Upgrading pip...")
        subprocess.run([python_exe, "-m", "pip", "install", "--upgrade", "pip"], check=True)

        if requirements_file.exists():
            print(f"Installing packages from {requirements_file.name}...")
            subprocess.run([python_exe, "-m", "pip", "install", "-r", str(requirements_file)], check=True)
        else:
            print("[ERROR] requirements.txt not found!")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install dependencies: {e}")
        sys.exit(1)

    # Create necessary folders
    print("\n--- Creating Directories ---")
    dirs_to_create = [
        "data/bg", "data/bg_raw",
        "data/speech_asr", "data/speech_kws",
        "data/esc50", "data/ms_snsd", "data/musan", "data/noisex",
        "descriptors", "itembanks", "prereg", "manifests",
        "inference", "scoring", "results", "checks/inspection",
        "models/cache",
    ]
    for d in dirs_to_create:
        dir_path = ROOT / d
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  Created/verified: {d}")

    print("\n=== Setup Completed Successfully! ===")
    print(f"Hugging Face & Torch cache directory: {ROOT / 'models' / 'cache'}")

    if not already_in_venv:
        activate = r".\.venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
        print(f"\nActivate the virtual environment with:\n  {activate}")


if __name__ == "__main__":
    main()
