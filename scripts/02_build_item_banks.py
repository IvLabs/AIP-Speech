#!/usr/bin/env python3
"""
02_build_item_banks.py — Stage 2: build foreground item banks.

Reads downloaded corpora and produces:
  itembanks/asr.jsonl  : {id, wav, transcript, source, accent, gender, age}
  itembanks/kws.jsonl  : {id, wav, keyword, is_target, probe_id?}

All audio paths are RELATIVE to ROOT so the project is portable.
Common Voice is streamed; only the clips we keep are saved, under
data/speech_asr/common_voice/.

Usage:
  python scripts/02_build_item_banks.py
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, DATA, CACHE, KWS_TARGETS, get_logger, ProgressLog,
                   jsonl_append, jsonl_ids)

log = get_logger("02_build_item_banks")

ASR_BANK   = ROOT / "itembanks" / "asr.jsonl"
KWS_BANK   = ROOT / "itembanks" / "kws.jsonl"
PROGRESS   = ROOT / "checks" / "bank_progress.json"

RANDOM_SEED = 42

# Target counts (per proposal: ~100 ASR items, ~120 KWS items)
ASR_N   = 100
CV_N    = 80
KWS_N   = 120

# Common Voice 17, streamed through its loading script (needs datasets < 4.0)
CV_DATASET = "fsicoli/common_voice_17_0"


def build_asr_bank(prog: ProgressLog) -> None:
    key = "asr_bank"
    if prog.done(key):
        log.info("[skip] ASR bank already built.")
        return

    done_ids = jsonl_ids(ASR_BANK)
    items: list[dict] = []

    # gender comes from SPEAKERS.TXT, not the transcript files
    speakers_file = DATA / "speech_asr" / "LibriSpeech" / "SPEAKERS.TXT"
    speaker_gender = {}
    if speakers_file.exists():
        with open(speakers_file) as f:
            for line in f:
                if line.startswith(";"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    spk_id = parts[0]
                    gender = "female" if parts[1] == "F" else "male" if parts[1] == "M" else "unknown"
                    speaker_gender[spk_id] = gender

    for split in ["test-clean", "test-other"]:
        trans_files = sorted(
            (DATA / "speech_asr" / "LibriSpeech" / split).rglob("*.trans.txt")
        )
        for tf in trans_files:
            with open(tf) as f:
                for line in f:
                    parts = line.strip().split(" ", 1)
                    if len(parts) != 2:
                        continue
                    utt_id, transcript = parts
                    wav = tf.parent / f"{utt_id}.flac"
                    if not wav.exists():
                        continue
                    spk_id = utt_id.split("-")[0]
                    items.append({
                        "id": utt_id,
                        "wav": str(wav.relative_to(ROOT)),
                        "transcript": transcript,
                        "source": f"librispeech_{split}",
                        "accent": "american_english",
                        "gender": speaker_gender.get(spk_id, "unknown"),
                        "age": "unknown",
                    })

    # shuffle first so both splits survive the cap at ASR_N
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(items)
    for item in items[:ASR_N]:
        if item["id"] not in done_ids:
            jsonl_append(ASR_BANK, item)
            done_ids.add(item["id"])

    # LibriSpeech alone gives no accent spread — top up from Common Voice
    _augment_asr_with_cv(done_ids, len(done_ids) + CV_N, prog)
    prog.mark(key)
    log.info(f"[done] ASR bank: {len(jsonl_ids(ASR_BANK))} items.")


def _cv_error(reason: str) -> str:
    """Actionable message for a Common Voice stream that did not deliver CV_N clips."""
    import os
    from importlib.metadata import PackageNotFoundError, version
    try:
        found = version("datasets")
    except PackageNotFoundError:
        found = "not installed"
    # datasets names a lock file after the whole cache path, which can pass
    # Windows' 260-character limit
    windows = (" On Windows, a FileNotFoundError on a .lock file means the path is too "
               "long: enable long paths or use a shorter repository path.") if os.name == "nt" else ""
    return (f"[CV] Could not stream the ASR bank's {CV_N} accent-labelled Common Voice clips "
            f"from {CV_DATASET} (config 'en', split 'validation'): {reason}. "
            f"This dataset is a loading script: it needs datasets < 4.0 and not 3.6.0 "
            f"(requirements.txt pins 3.5.1; installed: {found}). Also check that huggingface.co "
            f"is reachable and that {CV_DATASET} is still on the Hub, then re-run "
            f"scripts/02_build_item_banks.py.{windows}")


def _augment_asr_with_cv(done_ids: set, target: int, prog: ProgressLog) -> None:
    existing = len(done_ids)
    if existing >= target:
        return
    needed = target - existing
    log.info(f"[CV] Streaming Common Voice to add {needed} diverse items...")
    # rows are appended only once all clips are in: the ids depend on the stream
    # position, so a partial set would shift the ids of a re-run
    rows: list[dict] = []
    added = 0
    try:
        from datasets import load_dataset
        cv = load_dataset(
            CV_DATASET,
            "en",
            split="validation",
            streaming=True,
            trust_remote_code=True,
            cache_dir=str(CACHE / "datasets"),
        )
        for ex in cv:
            if added >= needed:
                break

            accent = ex.get("accent") or ""
            gender = ex.get("gender") or ""
            if not accent or not gender or accent.lower() == "unknown" or gender.lower() == "unknown":
                continue

            utt_id = f"cv_{ex['client_id'][:8]}_{added}"
            if utt_id in done_ids:
                continue
            # the stream is not replayable at inference time, so keep the clip
            out = DATA / "speech_asr" / "common_voice" / f"{utt_id}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                import soundfile as sf
                audio = ex["audio"]
                sf.write(str(out), audio["array"], audio["sampling_rate"])
            rows.append({
                "id": utt_id,
                "wav": str(out.relative_to(ROOT)),
                "transcript": ex.get("sentence", ""),
                "source": "common_voice_17",
                "accent": accent,
                "gender": gender,
                "age":    ex.get("age",    "unknown") or "unknown",
            })
            done_ids.add(utt_id)
            added += 1
    except Exception as e:
        raise RuntimeError(_cv_error(f"{type(e).__name__}: {e}")) from e
    if added < needed:
        raise RuntimeError(_cv_error(f"the stream ended after {added} of {needed} clips"))
    for row in rows:
        jsonl_append(ASR_BANK, row)
    log.info(f"[CV] Added {added} Common Voice items.")


def build_kws_bank(prog: ProgressLog) -> None:
    key = "kws_bank"
    if prog.done(key):
        log.info("[skip] KWS bank already built.")
        return

    done_ids = jsonl_ids(KWS_BANK)
    kws_root = DATA / "speech_kws"
    rng = random.Random(RANDOM_SEED)

    items: list[dict] = []
    for kw in KWS_TARGETS:
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        rng.shuffle(wav_files)
        for w in wav_files[:KWS_N // len(KWS_TARGETS) + 2]:
            items.append({
                "id": f"kws_{kw}_{w.stem}",
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": True,
                "probe_id": None,
            })

    # Non-target (background / silence)
    for kw in ["_background_noise_", "silence"]:
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        rng.shuffle(wav_files)
        for w in wav_files[:KWS_N // 4]:
            items.append({
                "id": f"kws_nt_{kw}_{w.stem}",
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": False,
                "probe_id": None,
            })

    # Held-out clips that say a target word (the injection probes of the
    # original design). 05_build_manifests.py leaves them out of the KWS
    # manifest, but they stay in the pool so the shuffle below selects the
    # same foreground clips as the paper's runs.
    for kw in KWS_TARGETS[:4]:
        wav_files = sorted((kws_root / kw).glob("*.wav")) if (kws_root / kw).exists() else []
        probes = wav_files[-3:] if len(wav_files) >= 3 else wav_files
        for w in probes:
            pid = f"probe_{kw}_{w.stem}"
            items.append({
                "id": pid,
                "wav": str(w.relative_to(ROOT)),
                "keyword": kw,
                "is_target": False,
                "probe_id": pid,
            })

    rng.shuffle(items)
    for item in items[:KWS_N]:
        if item["id"] not in done_ids:
            jsonl_append(KWS_BANK, item)
            done_ids.add(item["id"])

    prog.mark(key)
    log.info(f"[done] KWS bank: {len(jsonl_ids(KWS_BANK))} items.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 2 — Build foreground item banks")
    ap.parse_args()

    prog = ProgressLog(PROGRESS)

    build_asr_bank(prog)
    build_kws_bank(prog)

    log.info("=== Stage 2 complete. ===")


if __name__ == "__main__":
    main()
