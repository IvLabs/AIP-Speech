# AIP-Speech

Code artifact for the paper *AIP-Speech: Auditing Speech LLM Robustness to
Background Interference* ([`acl_latex 1.pdf`](./acl_latex%201.pdf)).

**Models:** `Qwen2.5-Omni-3B`, `Qwen2.5-Omni-7B`, `Qwen2-Audio-7B-Instruct`,
`Phi-4-multimodal-instruct`, `gemma-3n-E4B-it`.
**Tasks:** ASR (LibriSpeech `test-clean`/`test-other`) and closed-set KWS
(Google Speech Commands v2), each replayed clean and mixed with 20 curated
real-world backgrounds (ESC-50, MS-SNSD, MUSAN, NOISEX-92) at 10/5/0 dB SNR.
**Fairness:** accent-subgroup ΔWER on Mozilla Common Voice (streamed, never
downloaded in bulk). No corpora or model weights are redistributed here —
Stages 1–2 below download/stream everything into the project directory.

---

## Setup

```bash
python setup.py
```

Python 3.10+. Creates (or reuses) `.venv`, installs `requirements.txt` into
it, and creates the `data/`, `models/`, `results/`, etc. directories the
pipeline expects. `requirements.txt` itself is **not tracked in this repo**
(gitignored) — supply your own next to `setup.py`; at minimum you need
`torch`, `transformers`, `accelerate`, `bitsandbytes`, `librosa`,
`soundfile`, `pyloudnorm`, `jiwer`, `pandas`, `numpy`, `scipy`,
`statsmodels`, `seaborn`, `matplotlib`, `datasets`, `huggingface_hub`.

Activate before running anything else:
- Linux/macOS: `source .venv/bin/activate`
- Windows: `.\.venv\Scripts\Activate.ps1`

Everything — downloaded corpora, model cache, results — lives inside the
project directory (`HF_HOME`/`TORCH_HOME` are redirected to
`models/cache/` by every script); nothing is written system-wide. Stage 5
(inference) needs a GPU; every other stage runs on CPU.

---

## Pipeline

```bash
python scripts/01_download.py              # Stage 1 — download ESC-50 / MS-SNSD / MUSAN / NOISEX / LibriSpeech / Speech Commands
python scripts/02_build_banks.py           # Stage 2 — build itembanks/{asr,kws}.jsonl (Common Voice streamed for accent metadata)
python scripts/03_curate_battery.py        # Stage 3 — curate data/bg/, descriptors/battery.parquet, freeze prereg/prereg.json
python scripts/04_manipulation_checks.py   # Stage 4 — background-presence / WER-constancy / determinism checks
python scripts/05_inference.py --model all --task all   # Stage 5 — run every model over every manifest (GPU)
python scripts/06_score_metrics.py         # Stage 6 — WER/CER/accuracy/TIR/DRI/RER → results/*.csv
```

`05_inference.py` also takes a single `--model` (`qwen25_omni_3b`,
`qwen25_omni_7b`, `qwen2_audio_7b`, `phi4_multimodal`, `gemma3n_e4b`) and
`--task` (`asr`, `kws`, `asr_steer[_p1..p5]`, `kws_steer[_p1..p5]`) instead
of `all`/`all`.

### Paper figures

```bash
python scripts/generate_plots_new.py       # aggregates inference/*.jsonl → results/plots_new/{asr,kws}_results.csv
python scripts/generate_results_new_2.py   # descriptor↔degradation correlations + Figures 1–7 → results_new/
```

This is the path that produced the figures in `acl_latex 1.pdf`. It reads
`inference/*.jsonl` directly and does not depend on Stage 6's output.
Internal function names (e.g. `generate_figure2_heatmap_horizontal`) don't
all line up 1:1 with the paper's final figure numbers — the `results_new/fig1_...`
through `fig7_...` filenames are the authoritative mapping.

### Additional scoring (`scoring/`)

A second, independent analysis path over Stage 6's output tables:

```bash
python scoring/score_all.py    # standalone Stage-6 scorer (E1–E4), writes the same results/*.csv filenames as 06_score_metrics.py
python scoring/analyse.py      # Stage 7 — descriptor OLS regression + VIF, fairness permutation test, injection summary
python scoring/figures.py      # Stage 8 — earlier figure/appendix set, predates the camera-ready descriptor set
```

Neither this path nor Stage 6 feeds the paper-figure path above.

---

## Repository layout

### `scripts/`

| File | Role |
|------|------|
| `01_download.py` | Stage 1 — resumable download of all source corpora into `data/`. |
| `02_build_banks.py` | Stage 2 — builds `itembanks/asr.jsonl` and `itembanks/kws.jsonl` from the downloaded corpora. |
| `03_curate_battery.py` | Stage 3 — curates the ~20-recording background battery, extracts descriptors, freezes `prereg/prereg.json`. |
| `04_manipulation_checks.py` | Stage 4 — background-presence, WER-constancy, and determinism-floor checks; materializes the fixed inspection sample. |
| `05_inference.py` | Stage 5 — `SpeechLLM` base class + one adapter per model; writes `inference/<model>/<task>.jsonl`. |
| `06_score_metrics.py` | Stage 6 — WER/CER/ΔWER/accuracy/FAR/miss/TIR/DRI/RER from inference JSONL → `results/*.csv`. |
| `generate_plots_new.py` | Aggregates `inference/*.jsonl` into `results/plots_new/{asr,kws}_results.csv`. |
| `generate_results_new_2.py` | Descriptor↔degradation Spearman correlations; writes the paper's Figures 1–7 to `results_new/`. |
| `vad_classify_bg.py` | Silero VAD + WebRTC VAD majority-vote classification of background recordings. |
| `utils.py` | Shared paths, logging, JSONL/CSV I/O, and loudness-normalization helpers used by every script. |
| `run_modal_inference.py` | Modal cloud-GPU entry point running Stage 5 over a full model/task sweep. |
| `run_kws_steer_modal_pipeline.py` | Modal cloud-GPU entry point running Stage 5 (`kws_steer_p5` only) + Stage 6 remotely. |
| `replace_bg_results.py`, `update_inference_gender.py` | One-off local patch scripts for specific inference runs (hardcoded local paths — not part of the regular pipeline). |

### `scoring/`

| File | Role |
|------|------|
| `score_all.py` | Standalone Stage-6 scorer (E1–E4) — parallel implementation to `scripts/06_score_metrics.py`, same output filenames. |
| `analyse.py` | Stage 7 — descriptor OLS regression + VIF (`profile_regression.csv`), fairness permutation test, injection summary. |
| `figures.py` | Stage 8 — earlier figure/appendix set (Figs 1–3, A1–A5) built on the pre-registration's original descriptor set; superseded by `generate_results_new_2.py` for the camera-ready figures. |

### `mixing/`, `prereg/`

| File | Role |
|------|------|
| `mixing/mix.py` | On-the-fly SNR mixing. `materialize(row)` regenerates any stimulus bit-for-bit from its manifest row; nothing is persisted in bulk except the Stage-4 inspection sample and `checks/mix_diagnostics.csv`. |
| `prereg/prereg.json` | Frozen pre-registration (background IDs, SNR grid, metric definitions), written by Stage 3. |

---

## Known issues

- **`scripts/05_inference.py`** — `Qwen25Omni3B`, `Qwen25Omni7B`,
  `Qwen2Audio7B`, and `Gemma3nE4B` each call
  `self.model.load_adapter(MODEL_ID, ...)`, but `MODEL_ID` is only defined
  inside `Phi4Multimodal.__init__`. Only `Phi4Multimodal` runs as committed.
- **`scoring/analyse.py`** — `_load_battery()` has an incomplete
  `if "category" in df.columns:` with no body (`IndentationError`); does
  not run as committed.
