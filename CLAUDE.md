# CLAUDE.md — AIP-Speech Progress Log

> **Project:** AIP-Speech — Acoustic Interference Profiling for Speech-LLMs
> **Target venue:** IMPACT-SPEECH @ EMNLP 2026 (4-page short, 15 Jul 2026 AoE)
> **Compute:** Single 24 GB GPU, inference-only, ~10 days

---

## Stage Status

| Stage | Script | Status | Notes |
|-------|--------|--------|-------|
| 0 | `setup.py` | ✅ written | Cross-platform: create/reuse `.venv`, install deps via pip |
| 1 | `scripts/01_download.py` | ✅ written | Download ESC-50, MS-SNSD, MUSAN, NOISEX, LibriSpeech, SpeechCommands; resumable |
| 2 | `scripts/02_build_banks.py` | ✅ written | Build ASR/KWS item banks; CV streamed |
| 3 | `scripts/03_curate_battery.py` | ✅ written | Curate ~20 backgrounds, all descriptors, freeze prereg |
| 4 | `scripts/04_manipulation_checks.py` | ✅ written | bg_presence, wer_constancy, determinism; materialise inspection sample |
| 5 | `scripts/05_inference.py` | ✅ written | All **6** model adapters (+ Qwen2.5-Omni-7B); 8-bit/fp16 precision; one-at-a-time GPU; row-by-row JSONL; resumable |
| 6 | `scripts/06_score_metrics.py` | ✅ written | WER/CER/FAR/Miss/TIR/DRI/RER scoring for all experiments |
| 7 | `scoring/analyse.py` | ✅ written | C-PROFILE OLS regression + bootstrap CI + VIF; C-FAIR permutation test; C-INJECT |
| 8 | `scoring/figures.py` | ✅ written | Figs 1–3 (main paper) + Figs A1–A5 (appendix) as PDF+PNG |

---

## Commands

### Environment setup
One command, same on every platform. `setup.py` creates (or reuses) `.venv`
itself and installs `requirements.txt` into it — no separate venv step needed:

```bash
python setup.py
```

Then activate `.venv` for subsequent commands:

**Linux / macOS:** `source .venv/bin/activate`
**Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`

### Stage 1 — Download (smoke test)
```bash
python scripts/01_download.py --smoke-test
python scripts/01_download.py          # full run (resumable)
```

### Stage 2 — Build item banks (smoke test)
```bash
python scripts/02_build_banks.py --smoke-test
python scripts/02_build_banks.py
```

### Stage 3 — Curate battery (smoke test)
```bash
python scripts/03_curate_battery.py --smoke-test
python scripts/03_curate_battery.py
```

### Stage 4 — Manipulation checks (smoke test)
```bash
python scripts/04_manipulation_checks.py --smoke-test
python scripts/04_manipulation_checks.py
```

### Stage 5 — Inference (smoke test)
```bash
python scripts/05_inference.py --model qwen25_omni_3b --task asr --smoke-test
python scripts/05_inference.py --model qwen25_omni_3b --task asr   # one model/task
python scripts/05_inference.py --model qwen25_omni_7b --task asr   # 7B within-family contrast
python scripts/05_inference.py --model all --task all               # full sweep (6 models)
```

### Stage 6 — Scoring
```bash
python scripts/06_score_metrics.py --smoke-test
python scripts/06_score_metrics.py
```

### Stage 7 — Analysis
```bash
python scoring/analyse.py --smoke-test
python scoring/analyse.py
```

### Stage 8 — Figures
```bash
python scoring/figures.py
```

---

## Change Log

| Date | What |
|------|------|
| 2026-06-23 | **Full scaffold written.** All 8 stages scripted from proposal + implementation plan. `utils.py`, `mixing/mix.py`, `scripts/{01–05}`, `scoring/{score_all,analyse,figures}.py`, `setup.sh`, `.gitignore`, `CLAUDE.md`, `results.md` |
| 2026-06-24 | **MS-SNSD & ESC-50 fixes.** Migrated background noise pipeline from DEMAND to MS-SNSD. Cleaned up DEMAND references, code, and raw data. Fixed ESC-50 category mapping IDs. Reran Stage 2, 3, and 4 to verify correct sound curation. |
| 2026-06-24 | **Stage 5 Inference fixes.** Fixed device placement/VRAM OOM with BitsAndBytes 4-bit quantization and monkey-patching of `caching_allocator_warmup` to bypass pre-allocation. Handled the tuple return format of Qwen2.5-Omni generator. Successfully completed the Qwen2.5-Omni ASR smoke test. |
| 2026-06-28 | **Inference & Metrics scoring update.** Added audio duration clipping (30s max) and recovery from CUDA OOM in `05_inference.py`. Replaced `scoring/score_all.py` with `scripts/06_score_metrics.py` to evaluate E1-E4. |
| 2026-06-28 | **24 GB upgrade — precision & model roster.** Switched all models from 4-bit NF4 → fp16/8-bit (per `AIP_Speech_24GB_Updates.md`). Added `Qwen2.5-Omni-7B` (8-bit, Thinker-only) as within-family capacity contrast. Fixed `03_curate_battery.py` bug where `battery.parquet` was not written on re-runs (descriptors now reloaded from existing WAVs). Fixed smoke-test progress contamination via separate `*_smoke.json` progress files. Cross-platform setup via `setup.py` + `requirements.txt`. |
| 2026-09-15 | **Removed Speech Accent Archive (SAA).** SAA download, item-bank building, inference task, and E3-SAA scoring/plots were never actually run for the reported results (few speakers per accent made per-accent ΔWER unstable — flagged by review). Removed `download_saa`, `build_saa_bank`, the `saa` inference task/manifest, and `_score_e3_saa` from `01_download.py`, `02_build_banks.py`, `05_inference.py`, `06_score_metrics.py`, and `scoring/score_all.py`; dropped `data/speech_saa` from `setup.py`/`setup.sh`. Accent fairness now runs on Common Voice only. **Consolidated to one setup file.** Folded `setup.sh`'s venv-creation into `setup.py` (stdlib `venv` module) and deleted `setup.sh` — `python setup.py` now works identically on Windows/Linux/macOS. |

---

## Known Issues / Decisions

- All data, models, and caches are stored **inside `aip-speech/`** (never system-wide).
- `HF_HOME`, `TORCH_HOME` env vars are set in each script to `./models/cache`.
- Stimulus set is **never persisted** — use `mixing/mix.py::materialize()` to regenerate any clip.
- Common Voice is **streamed** (not downloaded) to stay under 3 GB.
- Day-6 gate: if E1 shows no descriptor law and no disparity signal, pivot to robustness-null framing.
