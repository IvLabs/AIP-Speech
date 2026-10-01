"""
utils.py — shared helpers for AIP-Speech.
All scripts import from here to avoid duplication.
"""
from __future__ import annotations

import os, sys, json, csv, logging
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent   # repository root
DATA = ROOT / "data"
CACHE = ROOT / "models" / "cache"

# keep HuggingFace / torch downloads inside the project directory
os.environ.setdefault("HF_HOME", str(CACHE))
os.environ.setdefault("TORCH_HOME", str(CACHE))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "hub"))
os.environ.setdefault("HF_DATASETS_CACHE", str(CACHE / "datasets"))

SR = 16_000          # project-wide sample rate (Hz)
SPEECH_LUFS = -23.0  # speech loudness target
BG_LUFS = -23.0      # bg loudness target before SNR scaling
SNR_GRID = [10, 5, 0]

# the ten Google Speech Commands v2 keywords of the closed-set KWS protocol
KWS_TARGETS = ["yes", "no", "up", "down", "left", "right",
               "on", "off", "stop", "go"]

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        # UTF-8 stream, else Windows cp1252 consoles raise UnicodeEncodeError
        import io
        stream = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace") \
                 if hasattr(sys.stdout, "buffer") else sys.stdout
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                         datefmt="%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(logging.INFO)
    return log

def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def jsonl_read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8", errors="replace") as f:
        return [json.loads(l) for l in f if l.strip()]

def jsonl_ids(path: Path, key: str = "id") -> set:
    return {r[key] for r in jsonl_read(path) if key in r and r.get("raw") != "__ERROR__"}

def csv_append(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    fields = fieldnames or list(row.keys())
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)

def load_audio(path: Path | str, sr: int = SR) -> np.ndarray:
    """Load mono float32 at project SR."""
    import soundfile as sf
    import librosa
    x, r = sf.read(str(path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if r != sr:
        x = librosa.resample(x.astype(np.float32), orig_sr=r, target_sr=sr)
    return x.astype(np.float32)

def loudness_normalize(x: np.ndarray, target_lufs: float) -> np.ndarray:
    import pyloudnorm as pyln
    meter = pyln.Meter(SR)
    measured = meter.integrated_loudness(x)
    if np.isinf(measured):   # silence / near-silence
        return x
    return pyln.normalize.loudness(x, measured, target_lufs).astype(np.float32)

class ProgressLog:
    """
    Tiny key-value store backed by a JSON file.
    Records completed IDs so a script can resume after interruption.
    """
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict = json.loads(path.read_text()) if path.exists() else {}

    def done(self, key: str) -> bool:
        return self._data.get(key) == "done"

    def mark(self, key: str) -> None:
        self._data[key] = "done"
        self.path.write_text(json.dumps(self._data, indent=2))

    def __len__(self) -> int:
        return len(self._data)

def load_battery() -> list[dict]:
    """Curated backgrounds from Stage 3, one dict per row, in battery order."""
    bat = ROOT / "descriptors" / "battery.csv"
    if not bat.exists():
        return []
    import pandas as pd
    # CSV rather than parquet: reading parquet after transformers is imported
    # can segfault on some pyarrow/torch builds, and Stage 7 needs both
    return pd.read_csv(bat, dtype=str, keep_default_na=False).to_dict("records")

def normalize_text(s) -> str:
    """Lower-case, turn punctuation into spaces, collapse whitespace."""
    import re, string
    s = str(s).lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    return " ".join(s.split())

def wer(ref, hyp) -> float:
    """WER on normalized text; NaN if jiwer rejects the pair (empty reference)."""
    import jiwer
    try:
        return float(jiwer.wer(normalize_text(ref), normalize_text(hyp)))
    except Exception:
        return float("nan")

def kws_predicted(raw) -> str:
    """The keyword a KWS response names: its first token, punctuation stripped."""
    import string
    hyp = str(raw).lower().strip()
    return hyp.split()[0].strip(string.punctuation) if hyp else "silence"

def kws_correct(keyword, raw, is_target: bool) -> bool:
    """Closed-set KWS: a target clip needs an exact keyword match; a non-target
    clip is correct only if no target keyword was predicted."""
    predicted = kws_predicted(raw)
    if is_target:
        return predicted == str(keyword).lower().strip()
    return predicted not in KWS_TARGETS
