#!/usr/bin/env python3
"""
08_score_outputs.py — Stage 8: score every inference run used in the paper.

  ASR  WER per stimulus on normalized text (lower-cased, punctuation removed)
       and the paired ΔWER = WER(noisy) − WER(clean) for the same model and
       utterance.
  KWS  closed-set accuracy over the ten Speech Commands keywords — a target
       clip needs an exact keyword match, a non-target clip is correct only if
       no target keyword is predicted — and the paired Δacc = acc(noisy) −
       acc(clean).

Scored tasks: asr and kws (baseline prompts), asr_steer_p5 and kws_steer_p5
(steering prompt). Each task gets one CSV with one row per stimulus:

  results/scores/<task>.csv   → read by 09_make_figures.py

Usage:
  python scripts/08_score_outputs.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, get_logger, jsonl_read, wer, kws_predicted, kws_correct

log = get_logger("08_score_outputs")

INFER  = ROOT / "inference"
SCORES = ROOT / "results" / "scores"

ASR_TASKS = ["asr", "asr_steer_p5"]
KWS_TASKS = ["kws", "kws_steer_p5"]


def _all_models() -> list[str]:
    if not INFER.exists():
        return []
    return sorted(d.name for d in INFER.iterdir() if d.is_dir())


def _load_inference(model_id: str, task: str) -> pd.DataFrame:
    """Inference rows for one run, minus rows whose generation raised."""
    rows = jsonl_read(INFER / model_id / f"{task}.jsonl")
    return pd.DataFrame([r for r in rows if r.get("raw") != "__ERROR__"])


def _pair_with_clean(df: pd.DataFrame, score: str, clean_col: str, delta: str) -> pd.DataFrame:
    """Attach each item's clean-condition score and the noisy − clean delta."""
    clean = (df[df["condition"] == "clean"][["model", "speech_id", score]]
             .rename(columns={score: clean_col}))
    df = df.merge(clean, on=["model", "speech_id"], how="left")
    df[delta] = df[score] - df[clean_col]
    return df


def score_asr(task: str) -> pd.DataFrame:
    records = []
    for mid in _all_models():
        for _, row in _load_inference(mid, task).iterrows():
            records.append({
                "model":     mid,
                "speech_id": row.get("speech_id", ""),
                "bg_id":     row.get("background_id", ""),
                "snr_db":    row.get("snr_db", ""),
                "condition": row.get("condition", ""),
                "accent":    row.get("accent", ""),
                "gender":    row.get("gender", ""),
                "wer":       wer(row.get("transcript", ""), row.get("raw", "")),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = _pair_with_clean(df, "wer", "wer_clean", "dwer")
    return df


def score_kws(task: str) -> pd.DataFrame:
    records = []
    for mid in _all_models():
        for _, row in _load_inference(mid, task).iterrows():
            raw = row.get("raw", "")
            is_target = bool(row.get("is_target", False))
            records.append({
                "model":     mid,
                "speech_id": row.get("speech_id", ""),
                "bg_id":     row.get("background_id", ""),
                "snr_db":    row.get("snr_db", ""),
                "condition": row.get("condition", ""),
                "keyword":   str(row.get("keyword", "")).lower().strip(),
                "is_target": is_target,
                "predicted": kws_predicted(raw),
                "correct":   int(kws_correct(row.get("keyword", ""), raw, is_target)),
            })
    df = pd.DataFrame(records)
    if not df.empty:
        df = _pair_with_clean(df, "correct", "acc_clean", "dacc")
    return df


def _save(df: pd.DataFrame, name: str) -> None:
    out = SCORES / name
    if df.empty:
        # don't leave a stale file behind from an earlier run
        out.unlink(missing_ok=True)
        log.info(f"  → {name}: no inference outputs found, skipped")
        return
    df.to_csv(out, index=False)
    log.info(f"  → {out}  ({len(df)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 8 — Score inference outputs")
    ap.parse_args()

    SCORES.mkdir(parents=True, exist_ok=True)
    for task in ASR_TASKS + KWS_TASKS:
        log.info(f"[{task}] scoring ...")
        df = score_asr(task) if task in ASR_TASKS else score_kws(task)
        _save(df, f"{task}.csv")

    log.info("=== Stage 8 complete. Scores in results/scores/ ===")


if __name__ == "__main__":
    main()
