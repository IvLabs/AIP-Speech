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
| 5 | `scripts/05_inference.py` | ✅ written | All **5** model adapters; 8-bit/fp16 precision; one-at-a-time GPU; row-by-row JSONL; resumable |
| 6 | `scripts/06_score_metrics.py` | ✅ written | WER/CER/FAR/Miss/TIR/DRI/RER scoring for all experiments |
| 7–8 | `scripts/generate_plots_new.py`, `scripts/generate_results_new_2.py` | ✅ written | Actual analysis/figure pipeline that produced the paper: Spearman descriptor correlations, top-3 descriptor selection, accent/RER/SNR figures. (`scoring/{score_all,analyse,figures}.py` — the original Stage 6–8 scaffold — were removed 2026-09-15: fully superseded, dead outputs, and `score_all.py`'s KWS scoring used a different/incorrect non-closed-set formula.) |

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

### Stage 1 — Download
```bash
python scripts/01_download.py          # full run (resumable)
```

### Stage 2 — Build item banks
```bash
python scripts/02_build_banks.py
```

### Stage 3 — Curate battery
```bash
python scripts/03_curate_battery.py
```

### Stage 4 — Manipulation checks
```bash
python scripts/04_manipulation_checks.py
```

### Stage 5 — Inference
```bash
python scripts/05_inference.py --model qwen25_omni_3b --task asr   # one model/task
python scripts/05_inference.py --model qwen25_omni_7b --task asr   # 7B within-family contrast
python scripts/05_inference.py --model all --task all               # full sweep (5 models)
```

### Stage 6 — Scoring
```bash
python scripts/06_score_metrics.py
```

### Stage 7–8 — Analysis & Figures
```bash
python scripts/generate_plots_new.py       # builds results/plots_new/{asr,kws}_results.csv
python scripts/generate_results_new_2.py   # descriptor correlations + paper figures → results_new/
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
| 2026-09-15 | **Removed Spoken Question Answering (SQA).** Dropped from the paper at the last moment; most of the SQA pipeline was dead code (`sqa_results.csv` was read but never written by anything) or Modal-only. Deleted 7 SQA-only scripts, removed `build_sqa_bank`, the `sqa` task/prompt/manifest, `score_e1_sqa` + its 4 helpers, and SQA columns/rows from every figure-generating script — including `generate_results_new_2.py`'s descriptor-correlation ranking, which was silently including SQA despite the paper being ASR+KWS-only. **Removed all `--smoke-test` flags.** Stripped `--smoke-test`/`smoke` plumbing from every script (`01_download.py`–`06_score_metrics.py`, `scoring/{analyse,score_all}.py`, `vad_classify_bg.py`, both `run_*_modal_pipeline.py` wrappers) and `ProgressLog`'s smoke-progress-file redirect in `utils.py`; dropped `smoke_mode` from the frozen `prereg/prereg.json`. |
| 2026-09-15 | **Removed the superseded old scoring pipeline.** Deleted `scoring/score_all.py`, `scoring/analyse.py`, `scoring/figures.py` (original Stage 6–8 scaffold) — confirmed via file-dependency tracing that nothing they uniquely produce (`e4_steerability.csv`, `profile_regression.csv`, `fairness_summary.csv`, `fairness_permutation.json`, `injection_summary.csv`) is read by any script in the actual pipeline that produced the paper (`generate_plots_new.py`, `generate_results_new_2.py`). `score_all.py`'s KWS scoring also used a different, non-closed-set correctness formula (`hyp_kw != ref_kw` instead of `hyp_kw not in KWS_TARGETS`) than every other implementation in the repo — an overlapping-but-divergent duplicate, not just dead code. Also deleted `compute_rer_fast.py`/`compute_rer_instant.py`: earlier iterations of the RER computation now in `generate_results_new_2.py::generate_prompt_rer_plot()`; both only printed to stdout, so nothing could have depended on them. Deleting `scoring/analyse.py` incidentally also removed the pre-existing `IndentationError` bug flagged earlier. |
| 2026-09-15 | **Removed 5 more redundant/superseded scripts, verified against the actual paper PDF.** `generate_new_results.py` (writes to `result_new_1/`, read by nothing — not even the same filenames as the paper's Fig 1–7). `generate_snr_plot_fast.py` and `fix_fig4_perfect.py` — diffed line-by-line against `generate_results_new_2.py`'s own `generate_snr_performance_side_by_side_plot()`/`generate_accent_degradation_plot()`: functionally identical (same data pipeline, same styling, same output path); rendered the actual PDF pages for Figures 5 and 7 to confirm the embedded figures match this exact styling. `fix_fig4.py` — an earlier, visibly different draft (confirmed not to match the paper's styling). `run_kws_steer_modal_pipeline_duplicate.py` — diff showed only pinned-vs-unpinned dependency versions and whitespace vs. the original. |
| 2026-09-15 | **Internal-duplication pass across every script.** Deduped `05_inference.py`'s 5 byte-identical `unload()` methods into the `SpeechLLM` base class. Extracted `06_score_metrics.py`'s duplicated KWS correctness logic (`score_e1_kws`/`score_e4_kws`) into `_score_kws_row()`. Deleted `generate_results_new_2.py::generate_figure2_heatmap()` — a second, wrong-orientation Spearman heatmap function that doesn't match Figure 3 in the submitted PDF (verified via high-DPI render: paper has a single-line title, 2 rows × 9 cols, matching only `generate_figure2_heatmap_horizontal()`). **Major open finding, not acted on:** every output of `06_score_metrics.py` is read only by itself — none of it reaches `generate_plots_new.py`/`generate_results_new_2.py` (the scripts that actually produce the paper). Its unique metrics (TIR, DRI, robustness gap) don't appear anywhere in the paper's text either. It's still invoked by `run_kws_steer_modal_pipeline.py` though, so this is a much bigger, higher-stakes call than anything removed so far — see `camera-ready.md` for full evidence; needs a decision before touching. |

---

## Known Issues / Decisions

- All data, models, and caches are stored **inside `aip-speech/`** (never system-wide).
- `HF_HOME`, `TORCH_HOME` env vars are set in each script to `./models/cache`.
- Stimulus set is **never persisted** — use `mixing/mix.py::materialize()` to regenerate any clip.
- Common Voice is **streamed** (not downloaded) to stay under 3 GB.
- Day-6 gate: if E1 shows no descriptor law and no disparity signal, pivot to robustness-null framing.
