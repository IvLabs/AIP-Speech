#!/usr/bin/env python3
"""
05_build_manifests.py — Stage 5: build the stimulus manifests.

Pairs every item-bank entry (Stage 2) with the clean condition and with each
curated background (Stage 3) at every SNR in SNR_GRID, and gives each stimulus
its own mixing seed.

Produces (read by Stages 6 and 7):
  manifests/asr.csv   ASR stimuli, seeds counting up from 0
  manifests/kws.csv   KWS stimuli, seeds counting up from 10000

An existing manifest is kept as is; delete it to rebuild.

Usage:
  python scripts/05_build_manifests.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, SNR_GRID, get_logger, load_battery, jsonl_read

log = get_logger("05_build_manifests")

MANIFESTS = ROOT / "manifests"


def build_manifests() -> None:
    """Generate manifests/asr.csv and manifests/kws.csv if not present."""
    import pandas as pd

    battery = load_battery()
    if not battery:
        log.warning("[manifest] Battery not found. Run scripts/03_curate_backgrounds.py first.")
        return

    asr_path = MANIFESTS / "asr.csv"
    if not asr_path.exists():
        items = jsonl_read(ROOT / "itembanks" / "asr.jsonl")
        rows = []
        seed = 0
        for item in items:
            rows.append({
                "id": f"{item['id']}_clean",
                "speech_id": item["id"],
                "background_id": "clean",
                "snr_db": 99,
                "condition": "clean",
                "seed": seed,
                "speech_path": item["wav"],
                "bg_path": "",
                "accent": item.get("accent", ""),
                "gender": item.get("gender", ""),
                "transcript": item.get("transcript", ""),
            })
            seed += 1
            for bg in battery:
                for snr in SNR_GRID:
                    rows.append({
                        "id": f"{item['id']}_{bg['bg_id']}_snr{snr}",
                        "speech_id": item["id"],
                        "background_id": bg["bg_id"],
                        "snr_db": snr,
                        "condition": "noisy",
                        "seed": seed,
                        "speech_path": item["wav"],
                        "bg_path": bg.get("wav", ""),
                        "accent": item.get("accent", ""),
                        "gender": item.get("gender", ""),
                        "transcript": item.get("transcript", ""),
                    })
                    seed += 1
        MANIFESTS.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(asr_path, index=False)
        log.info(f"[manifest] asr.csv: {len(rows)} rows → {asr_path}")

    kws_path = MANIFESTS / "kws.csv"
    if not kws_path.exists():
        items = jsonl_read(ROOT / "itembanks" / "kws.jsonl")
        rows = []
        seed = 10000
        for item in items:
            if item.get("probe_id"):
                continue  # held-out probe clips are not part of the KWS set
            rows.append({
                "id": f"{item['id']}_clean",
                "speech_id": item["id"],
                "background_id": "clean",
                "snr_db": 99,
                "condition": "clean",
                "seed": seed,
                "speech_path": item["wav"],
                "bg_path": "",
                "keyword": item.get("keyword", ""),
                "is_target": item.get("is_target", False),
            })
            seed += 1
            for bg in battery:
                for snr in SNR_GRID:
                    rows.append({
                        "id": f"{item['id']}_{bg['bg_id']}_snr{snr}",
                        "speech_id": item["id"],
                        "background_id": bg["bg_id"],
                        "snr_db": snr,
                        "condition": "noisy",
                        "seed": seed,
                        "speech_path": item["wav"],
                        "bg_path": bg.get("wav", ""),
                        "keyword": item.get("keyword", ""),
                        "is_target": item.get("is_target", False),
                    })
                    seed += 1
        pd.DataFrame(rows).to_csv(kws_path, index=False)
        log.info(f"[manifest] kws.csv: {len(rows)} rows → {kws_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 5 — Build the stimulus manifests")
    ap.parse_args()

    build_manifests()
    log.info("=== Stage 5 complete. ===")


if __name__ == "__main__":
    main()
