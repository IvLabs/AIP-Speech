#!/usr/bin/env python3
"""
04_extract_descriptors.py — Stage 4: acoustic descriptors of the backgrounds.

Characterises every curated background in data/bg/ by the nine descriptors
analysed in the paper (Appendix B, Tables 2 and 3):
  - eight librosa descriptors, averaged over frames: zero crossing rate,
    spectral centroid, rolloff, bandwidth, flux (onset strength), flatness,
    contrast, and the variance of the frame RMS energy;
  - the Silero VAD speech fraction: the fraction of samples inside the speech
    segments detected by Silero VAD (snakers4/silero-vad via PyTorch Hub,
    published default threshold 0.5).

Produces:
  descriptors/acoustic_descriptors.csv   one row per bg_id — read by 09_make_figures.py

Usage:
  python scripts/04_extract_descriptors.py
  python scripts/04_extract_descriptors.py --sensitivity 0.5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, DATA, get_logger, load_audio

log = get_logger("04_extract_descriptors")

BG_DIR  = DATA / "bg"
OUT_CSV = ROOT / "descriptors" / "acoustic_descriptors.csv"

# published default, not tuned on this project's data
SILERO_SENSITIVITY = 0.50


def _spectral_descriptors(wav_path: Path) -> dict:
    """The eight librosa descriptors of one recording, each a mean over frames."""
    y, sr = sf.read(wav_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=16000)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=16000)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=16000)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))

    onset_env = librosa.onset.onset_strength(y=y, sr=16000)
    flux = float(np.mean(onset_env))

    contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=16000)))
    rms_var = float(np.var(librosa.feature.rms(y=y)[0]))

    return {
        "Zero Crossing Rate (ZCR)": round(zcr, 4),
        "Spectral Centroid (Hz)":   round(centroid, 2),
        "Spectral Rolloff (Hz)":    round(rolloff, 2),
        "Spectral Bandwidth (Hz)":  round(bandwidth, 2),
        "Spectral Flux":            round(flux, 4),
        "Spectral Flatness":        round(flatness, 4),
        "Spectral Contrast (dB)":   round(contrast, 4),
        "RMS Energy Variance":      round(rms_var, 6),
    }


def _load_silero():
    """Load Silero VAD from torch.hub (cached locally after first download)."""
    import torch
    import warnings
    log.info(f"[Silero] Loading model (cache: {torch.hub.get_dir()}) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            verbose=False,
        )
    (get_speech_timestamps, *_) = utils
    log.info("[Silero] Loaded successfully.")
    return model, get_speech_timestamps


def _speech_fraction(wav_path: Path, model, get_speech_timestamps,
                     sensitivity: float) -> float:
    """Fraction of samples inside Silero speech segments (NaN if Silero fails)."""
    import torch
    try:
        audio = load_audio(wav_path, sr=16_000)
        tensor = torch.from_numpy(audio)
        model.reset_states()
        timestamps = get_speech_timestamps(
            tensor, model,
            sampling_rate=16_000,
            window_size_samples=512,
            threshold=sensitivity,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
            return_seconds=False,
        )
        total = len(tensor)
        speech = sum(s["end"] - s["start"] for s in timestamps)
        return round(float(speech / total) if total > 0 else 0.0, 4)
    except Exception as e:
        log.warning(f"  [Silero error] {wav_path.name}: {e}")
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 4 — Extract acoustic descriptors")
    ap.add_argument(
        "--sensitivity",
        type=float, default=SILERO_SENSITIVITY,
        help=(
            f"Silero VAD per-frame speech probability threshold "
            f"(default: {SILERO_SENSITIVITY} — Silero's published default)."
        ),
    )
    args = ap.parse_args()

    wav_files = sorted(BG_DIR.glob("*.wav"))
    if not wav_files:
        log.error(f"No WAV files found in {BG_DIR}. Run scripts/03_curate_backgrounds.py first.")
        sys.exit(1)

    model, get_speech_timestamps = _load_silero()
    records = []
    for i, wav_path in enumerate(wav_files, 1):
        frac = _speech_fraction(wav_path, model, get_speech_timestamps, args.sensitivity)
        log.info(f"[{i}/{len(wav_files)}] {wav_path.stem}: speech fraction {frac:.4f}")
        records.append({
            "bg_id": wav_path.stem,
            **_spectral_descriptors(wav_path),
            "Silero VAD Speech Fraction": frac,
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).set_index("bg_id").to_csv(OUT_CSV)
    log.info(f"[save] {len(records)} rows → {OUT_CSV}")
    log.info("=== Stage 4 complete. ===")


if __name__ == "__main__":
    main()
