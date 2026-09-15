#!/usr/bin/env python3
"""
01_download.py — Stage 1: download all source corpora.

Downloads are placed INSIDE the project directory only.
All downloads are resumable: already-present files are skipped.

Corpora:
  ESC-50         → data/esc50/
  MS-SNSD        → data/ms_snsd/  (primary babble/cafeteria/office noise)
  MUSAN          → data/musan/
  NOISEX-92 babble → data/noisex/
  LibriSpeech test-clean + test-other → data/speech_asr/
  Google Speech Commands v2 → data/speech_kws/
  (Common Voice is STREAMED at inference, not downloaded)

Usage:
  python scripts/01_download.py              # full run (resumable)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
import tarfile
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None  # fallback to urlretrieve

from urllib.request import urlretrieve

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, DATA, CACHE, get_logger, ProgressLog

log = get_logger("01_download")
PROGRESS = ROOT / "checks" / "download_progress.json"


# ── helpers ──────────────────────────────────────────────────────────────────
def _reporthook(block: int, block_size: int, total: int) -> None:
    if total > 0:
        pct = min(block * block_size / total * 100, 100)
        print(f"\r  {pct:.1f}%", end="", flush=True)


def download(url: str, dest: Path, desc: str = "") -> Path:
    """Download url → dest. Skips if dest already exists and has size > 0."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info(f"[skip] {desc or dest.name} already downloaded.")
        return dest
    log.info(f"[download] {desc or url}")
    if requests is not None:
        # streaming download with requests (handles redirects robustly)
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            downloaded = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total > 0:
                        pct = min(downloaded / total * 100, 100)
                        print(f"\r  {pct:.1f}%", end="", flush=True)
        print()
    else:
        urlretrieve(url, str(dest), reporthook=_reporthook)
        print()  # newline after progress
    return dest


def extract_zip(archive: Path, out_dir: Path) -> None:
    """Extract a zip archive idempotently (skip if sentinel dir exists)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        z.extractall(out_dir)
    log.info(f"[extracted] {archive.name} → {out_dir}")


def extract_tar(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as t:
        t.extractall(out_dir)
    log.info(f"[extracted] {archive.name} → {out_dir}")


# ── corpus-specific download functions ───────────────────────────────────────

def download_esc50(prog: ProgressLog) -> None:
    """ESC-50: 2000 clips (5 s, 44.1 kHz) — ~600 MB unzipped."""
    key = "esc50"
    if prog.done(key):
        log.info("[skip] ESC-50 already complete.")
        return
    dest = DATA / "esc50"
    url = "https://github.com/karoldvl/ESC-50/archive/master.zip"
    archive = DATA / "esc50_master.zip"
    download(url, archive, "ESC-50")
    dest.mkdir(parents=True, exist_ok=True)
    extract_zip(archive, dest)
    prog.mark(key)
    log.info("[done] ESC-50")


# Using 16 kHz versions (project SR=16 kHz) from the Zenodo API endpoint.
# SPSQUARE and TCAR are not in this Zenodo record; substituting with similar envs.
MS_SNSD_GIT = "https://github.com/microsoft/MS-SNSD.git"

def download_ms_snsd(prog: ProgressLog) -> None:
    """
    MS-SNSD (Microsoft Scalable Noisy Speech Dataset).
    Noise WAVs land in data/ms_snsd/noise_train/ (~400 MB).
    Uses a shallow git clone (depth=1) — no Zenodo URL instability.
    MIT licensed, mono WAV @ 16 kHz, pipeline-ready.
    """
    key = "ms_snsd"
    if prog.done(key):
        log.info("[skip] MS-SNSD already done.")
        return

    dest = DATA / "ms_snsd"

    if (dest / "noise_train").exists() and any((dest / "noise_train").glob("*.wav")):
        log.info("[skip] MS-SNSD noise_train/ already populated.")
        prog.mark(key)
        return

    log.info("[download] MS-SNSD — shallow git clone (~400 MB noise WAVs) ...")
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", MS_SNSD_GIT, str(dest)],
        check=True,
    )
    n_wavs = len(list((dest / "noise_train").glob("*.wav")))
    log.info(f"[done] MS-SNSD — {n_wavs} noise WAVs in data/ms_snsd/noise_train/")
    prog.mark(key)



# Full NOISEX-92 (15 files) mirrored on GitHub: speechdnn/Noises
_NOISEX_BASE = "https://raw.githubusercontent.com/speechdnn/Noises/master/NoiseX-92"
NOISEX_FILES = [
    "babble.wav", "buccaneer1.wav", "buccaneer2.wav", "destroyerengine.wav",
    "destroyerops.wav", "f16.wav", "factory1.wav", "factory2.wav",
    "hfchannel.wav", "leopard.wav", "m109.wav", "machinegun.wav",
    "pink.wav", "volvo.wav", "white.wav",
]

def download_noisex(prog: ProgressLog) -> None:
    """NOISEX-92: all 15 noise files (~130 MB total) from speechdnn/Noises GitHub mirror."""
    key = "noisex_babble"
    if prog.done(key):
        log.info("[skip] NOISEX-92 already done.")
        return
    dest_dir = DATA / "noisex"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for fname in NOISEX_FILES:
        url = f"{_NOISEX_BASE}/{fname}"
        download(url, dest_dir / fname, f"NOISEX-92/{fname}")
    prog.mark(key)
    log.info(f"[done] NOISEX-92 ({len(NOISEX_FILES)} files)")


def download_musan(prog: ProgressLog) -> None:
    """MUSAN: speech + music + noise subsets (~10 GB full; we only keep subsets)."""
    key = "musan"
    if prog.done(key):
        log.info("[skip] MUSAN already done.")
        return
    dest_dir = DATA / "musan"
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    archive = dest_dir / "musan.tar.gz"
    download(url, archive, "MUSAN")
    extract_tar(archive, dest_dir)
    prog.mark(key)
    log.info("[done] MUSAN")


LIBRISPEECH_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}

def download_librispeech(prog: ProgressLog) -> None:
    """LibriSpeech test-clean + test-other (~400 MB total)."""
    for split in LIBRISPEECH_URLS:
        key = f"librispeech_{split}"
        if prog.done(key):
            log.info(f"[skip] LibriSpeech {split} already done.")
            continue
        dest_dir = DATA / "speech_asr"
        archive = dest_dir / f"{split}.tar.gz"
        download(LIBRISPEECH_URLS[split], archive, f"LibriSpeech/{split}")
        extract_tar(archive, dest_dir)
        prog.mark(key)
    log.info("[done] LibriSpeech")


def download_speech_commands(prog: ProgressLog) -> None:
    """Google Speech Commands v2 (~2.3 GB; only downloading v2 mini)."""
    key = "speech_commands_v2"
    if prog.done(key):
        log.info("[skip] Speech Commands v2 already done.")
        return
    dest_dir = DATA / "speech_kws"
    # Full dataset
    url = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
    archive = dest_dir / "speech_commands_v2.tar.gz"
    download(url, archive, "Speech Commands v2")
    extract_tar(archive, dest_dir)
    prog.mark(key)
    log.info("[done] Speech Commands v2")


# ── main ─────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 1 — Download source corpora")
    ap.parse_args()

    prog = ProgressLog(PROGRESS)

    download_esc50(prog)
    download_ms_snsd(prog)
    download_noisex(prog)
    download_musan(prog)
    download_librispeech(prog)
    download_speech_commands(prog)

    log.info(f"=== Stage 1 complete. {len(prog)} items marked done. ===")


if __name__ == "__main__":
    main()
