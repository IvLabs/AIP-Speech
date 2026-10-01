#!/usr/bin/env python3
"""
03_curate_backgrounds.py — Stage 3: curate the 20 background recordings.

Each source recording is converted to mono 16 kHz, trimmed/looped to 30 s,
loudness-normalised and saved under data/bg/.

Produces:
  data/bg/<bg_id>.wav           the curated backgrounds
  descriptors/battery.csv       bg_id, source_file, wav — read by Stages 5 to 7

The acoustic descriptors analysed in the paper are computed from data/bg/ by
Stage 4 (04_extract_descriptors.py).

Usage:
  python scripts/03_curate_backgrounds.py            # full run (resumable)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, SR, BG_LUFS,
                   get_logger, ProgressLog, load_audio, loudness_normalize)

log = get_logger("03_curate_backgrounds")
PROGRESS = ROOT / "checks" / "battery_progress.json"
BG_DIR   = DATA / "bg"
BATTERY  = ROOT / "descriptors" / "battery.csv"

# (bg_id, glob pattern relative to DATA). Keep this order: Stage 5 assigns
# mixing seeds in battery order, so reordering changes every stimulus.
BATTERY_SPEC: list[tuple[str, str]] = [
    # ESC-50 (category IDs verified against data/esc50/meta/esc50.csv)
    ("esc_rain",       "esc50/**/audio/1-*-A-10.wav"),   # rain          → ID 10
    ("esc_seawave",    "esc50/**/audio/1-*-A-11.wav"),   # sea_waves     → ID 11
    ("esc_engine",     "esc50/**/audio/1-*-A-44.wav"),   # engine        → ID 44
    ("esc_vacuum",     "esc50/**/audio/1-*-A-36.wav"),   # vacuum_cleaner→ ID 36
    ("esc_footsteps",  "esc50/**/audio/1-*-A-25.wav"),   # footsteps     → ID 25
    ("esc_fire",       "esc50/**/audio/1-*-A-12.wav"),   # crackling_fire→ ID 12
    ("esc_helicopter", "esc50/**/audio/1-*-A-40.wav"),   # helicopter    → ID 40
    ("esc_clock",      "esc50/**/audio/1-*-A-38.wav"),   # clock_tick    → ID 38
    ("esc_keyboard",   "esc50/**/audio/1-*-A-32.wav"),   # keyboard_typing→ ID 32
    ("esc_dog",        "esc50/**/audio/1-*-A-0.wav"),    # dog           → ID 0
    ("esc_rooster",    "esc50/**/audio/1-*-A-1.wav"),    # rooster       → ID 1
    ("esc_wind",       "esc50/**/audio/1-*-A-16.wav"),   # wind          → ID 16
    # MS-SNSD
    ("snsd_cafeteria", "ms_snsd/noise_train/CafeTeria_1.wav"),
    ("snsd_restaurant","ms_snsd/noise_train/Restaurant_1.wav"),
    ("snsd_square",    "ms_snsd/noise_train/Square_1.wav"),
    ("snsd_airport",   "ms_snsd/noise_train/AirportAnnouncements_7.wav"),
    # NOISEX-92
    ("noisex_babble",  "noisex/babble.wav"),
    # MUSAN (music; "hubbub" is a LibriVox read-speech recording)
    ("musan_music",    "musan/**/music/fma/music-fma-0062.wav"),
    ("musan_hubbub",   "musan/**/speech/librivox/speech-librivox-0001.*"),
    # MS-SNSD
    ("snsd_airconditioner", "ms_snsd/noise_train/AirConditioner_1.wav"),
]

CLIP_LEN_S = 30  # seconds to trim/loop each background to


def _prepare_bg(src_path: Path, out_path: Path) -> np.ndarray | None:
    """Load, mono, resample, trim/loop to CLIP_LEN_S, loudness-normalise, save."""
    if not src_path.exists():
        log.warning(f"  [missing] {src_path}")
        return None
    x = load_audio(src_path)
    target_len = CLIP_LEN_S * SR
    if len(x) < target_len:
        x = np.tile(x, int(np.ceil(target_len / len(x))))
    x = x[:target_len]
    x = loudness_normalize(x, BG_LUFS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), x, SR)
    return x


def curate_battery(prog: ProgressLog) -> None:
    records: list[dict] = []

    for bg_id, glob_pattern in BATTERY_SPEC:
        key = f"curate_{bg_id}"
        out = BG_DIR / f"{bg_id}.wav"
        src_candidates = sorted(DATA.glob(glob_pattern))

        if prog.done(key):
            if out.exists():
                log.info(f"[reload] {bg_id} already curated.")
                records.append({
                    "bg_id": bg_id,
                    "source_file": str(src_candidates[0].relative_to(DATA)) if src_candidates else "",
                    "wav": str(out.relative_to(ROOT)),
                })
            else:
                log.warning(f"[skip] {bg_id} marked done but WAV missing at {out}. Re-run to fix.")
            continue

        if not src_candidates:
            log.warning(f"[missing src] {bg_id}: no files match data/{glob_pattern}")
            continue
        src = src_candidates[0]

        log.info(f"[curate] {bg_id} ← {src.name}")
        if _prepare_bg(src, out) is None:
            continue
        records.append({
            "bg_id": bg_id,
            "source_file": str(src.relative_to(DATA)),
            "wav": str(out.relative_to(ROOT)),
        })
        prog.mark(key)

    if records:
        import pandas as pd
        BATTERY.parent.mkdir(parents=True, exist_ok=True)
        combined = pd.DataFrame(records).drop_duplicates("bg_id")
        combined.to_csv(BATTERY, index=False)
        log.info(f"[battery] {len(combined)} backgrounds → {BATTERY}")
    else:
        log.warning("[battery] No records collected — battery.csv not written.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 3 — Curate the background battery")
    ap.parse_args()

    prog = ProgressLog(PROGRESS)
    curate_battery(prog)
    log.info("=== Stage 3 complete. ===")


if __name__ == "__main__":
    main()
