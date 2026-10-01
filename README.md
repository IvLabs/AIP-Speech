# AIP-Speech

Code for the paper **AIP-Speech: Auditing Speech LLM Robustness to Background
Interference** ([arXiv:XXXX.XXXXX](https://arxiv.org/pdf/XXXX.XXXXX)).
<!-- TODO: replace XXXX.XXXXX with the paper's arXiv id -->

AIP-Speech is an inference-only benchmark for auditing how real-world
background sound degrades Speech LLMs. Clean utterances are mixed with 20
background recordings at 10, 5 and 0 dB SNR. Five open Speech LLMs run speech
recognition (ASR) and keyword spotting (KWS) on the clean and the noisy
versions, and each clean–noisy pair gives the degradation caused by a
background. Every background is described by nine interpretable acoustic
descriptors, and the main analysis relates them to degradation by Spearman
correlation. The benchmark also compares degradation across accent groups and
tests whether a steering prompt reduces it. This repository holds the whole
pipeline, from downloading the corpora to the paper's figures and tables. It
redistributes no corpora or model weights: the stages download them.

---

## Models and tasks

| `--model` | Hugging Face model | Precision |
|---|---|---|
| `qwen25_omni_3b` | `Qwen/Qwen2.5-Omni-3B` | fp16 |
| `qwen25_omni_7b` | `Qwen/Qwen2.5-Omni-7B` | 8-bit (bitsandbytes) |
| `qwen2_audio_7b` | `Qwen/Qwen2-Audio-7B-Instruct` | 8-bit (bitsandbytes) |
| `phi4_multimodal` | `microsoft/phi-4-multimodal-instruct` | bf16; remote code and its speech LoRA adapter |
| `gemma3n_e4b` | `google/gemma-3n-E4B-it` (gated) | bf16 |

Decoding is greedy, and audio is capped at 30 s. Responses are limited to 64
new tokens (`max_new_tokens=64`), except for the two Qwen2.5-Omni models:
their `generate()` ignores `max_new_tokens` and applies its own text limit,
`thinker_max_new_tokens` (1024 by default). The paper's runs had the same
limits, and Stage 7 keeps them.

The prompts are `PROMPTS` in `scripts/07_run_inference.py`. Each task has a
baseline prompt (`asr`, `kws`) and a steering prompt that tells the model to
attend only to the foreground speaker (`asr_steer_p5`, `kws_steer_p5`). The
paper's runs used the code's steering prompts; Appendix F prints condensed
versions.

- **ASR**: 180 utterances, 100 from LibriSpeech test-clean/test-other and 80
  accent-labelled English clips from Mozilla Common Voice 17 (streamed from
  `fsicoli/common_voice_17_0`; only the kept clips are saved). WER is computed
  on normalised text, and ΔWER = WER(noisy) − WER(clean) for the same
  utterance. The accent analysis (Figure 5, Table 4) uses the 80 Common Voice
  clips, which cover 11 accent groups.
- **KWS**: 113 Google Speech Commands v2 clips, scored closed-set over the ten
  keywords (yes, no, up, down, left, right, on, off, stop, go). The first word
  of the response must be the keyword on a target clip, and must not be any of
  the ten on a non-target clip. Degradation = accuracy(clean) − accuracy(noisy).
- **Backgrounds** (Table 1): 12 ESC-50 clips, 5 MS-SNSD recordings, NOISEX-92
  babble, and from MUSAN a music track and a LibriVox read-speech recording
  (`musan_hubbub`). Each is converted to mono 16 kHz, cut or looped to 30 s and
  loudness-normalised. Every item is run clean and with each background at
  each SNR, which gives 61 stimuli per item.
- **Descriptors**: eight librosa descriptors summarised over the frames of each
  recording (zero crossing rate; spectral centroid, rolloff, bandwidth, flux,
  flatness and contrast; RMS energy variance) plus the Silero VAD speech
  fraction. Degradation is averaged per background over the SNRs (and over the
  models for the pooled values) and rank-correlated with each descriptor
  across the 20 backgrounds.
- **Prompt steering**: the Residual Effect Ratio, RER = Δ(steering) /
  Δ(baseline), is computed per model and task. Values below 1 mean the
  steering prompt reduced degradation. A KWS RER is reported as 1.0 when both
  accuracy drops are below 0.01.

Responses are scored as generated, with no model-specific clean-up.

---

## Setup

You need Python 3.10–3.12 (the paper's runs used 3.10), `git` (Stage 1 clones
MS-SNSD), network access and an NVIDIA GPU for Stage 7. You also need plenty
of disk space: Stage 1 keeps the downloaded archives next to the extracted
corpora (MUSAN alone is about 10 GB extracted), and Stage 7 downloads all five
models.

PyTorch is a prerequisite and is not in `requirements.txt`. Install a CUDA
build of torch ≥ 2.4 and its matching torchvision for your GPU and driver
([pytorch.org](https://pytorch.org/get-started/locally/)); plain
`pip install torch` gives a CPU-only build on Windows, and torchaudio is not
needed. Then run `setup.py` in the same environment:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .\.venv\Scripts\Activate.ps1
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
python setup.py
```

torch 2.5.1 is the release the paper's Phi-4 runs used; pick the build that
suits your GPU and driver. `setup.py` uses the active virtual environment (or
creates `./.venv` if none is active) and creates the folders the stages write
to. It stops with instructions if torch is missing, CPU-only or older than
2.4. It then installs the pinned `requirements.txt` without letting pip
replace torch or torchvision, and exits with an error if PyTorch sees no GPU.
`requirements.txt` has comments that explain the less obvious pins
(transformers 5.12.1, datasets 3.5.1 for the Common Voice loading script, …).

Gemma 3n is gated. Accept its licence on its
[model page](https://huggingface.co/google/gemma-3n-E4B-it), then run
`hf auth login` (or set `HF_TOKEN`) before Stage 7.

**Where files go.**
- `data/`: the corpora (Stage 1), the saved Common Voice clips (Stage 2) and
  the curated backgrounds (Stage 3)
- `models/cache/hub/`: the model weights (Stage 7) and Silero VAD (Stage 4,
  through PyTorch Hub)
- `models/cache/datasets/`: the Common Voice stream cache (Stage 2)
- `models/cache/whisper/`: Whisper `base` (Stage 6)

`scripts/utils.py` sets `HF_HOME` and `TORCH_HOME` to `models/cache/` and
`HF_DATASETS_CACHE` to `models/cache/datasets/`, but only when they are not
already set, so a value you export first takes precedence. The weights, the
Common Voice cache and Whisper get explicit paths, so they stay in
`models/cache/` whatever you export. Stage 7 imports transformers before
`utils.py`, and Hugging Face reads its default locations at that import,
before `utils.py` sets `HF_HOME`. Stage 7 therefore sets `HF_HUB_CACHE` to
`models/cache/hub/` just before that import (again only if you have not
exported it), so the two files it fetches without an explicit path, Phi-4's
speech LoRA adapter and Qwen2.5-Omni's `spk_dict.pt`, are stored there as
well. The login token and the module cache for Phi-4's remote code still use
Hugging Face's default location (`~/.cache/huggingface`, or your `HF_HOME` if
you export one).

On Windows, a Stage 2 `FileNotFoundError` on a `.lock` file means the path is
too long for the datasets cache. Enable long paths or move the repository to a
shorter path.

---

## Pipeline

Run the stages in order:

```bash
python scripts/01_download_data.py
python scripts/02_build_item_banks.py
python scripts/03_curate_backgrounds.py
python scripts/04_extract_descriptors.py
python scripts/05_build_manifests.py
python scripts/06_manipulation_checks.py
python scripts/07_run_inference.py --model all --task all
python scripts/08_score_outputs.py
python scripts/09_make_figures.py
```

| Stage | Reads | Writes | GPU |
|---|---|---|---|
| `01_download_data.py` | — | `data/`: ESC-50, MS-SNSD, NOISEX-92, MUSAN, LibriSpeech test-clean + test-other, Speech Commands v2 | — |
| `02_build_item_banks.py` | LibriSpeech and Speech Commands in `data/`, Common Voice 17 (streamed) | `itembanks/asr.jsonl` (100 LibriSpeech + 80 Common Voice), `itembanks/kws.jsonl`, `data/speech_asr/common_voice/` | — |
| `03_curate_backgrounds.py` | ESC-50, MS-SNSD, NOISEX-92 and MUSAN in `data/` | `data/bg/<bg_id>.wav` (20 × 30 s), `descriptors/battery.csv` | — |
| `04_extract_descriptors.py` | `data/bg/` | `descriptors/acoustic_descriptors.csv` (the 9 descriptors per background) | — |
| `05_build_manifests.py` | `itembanks/`, `descriptors/battery.csv` | `manifests/asr.csv`, `manifests/kws.csv` (61 stimuli per item, each with its own mixing seed) | — |
| `06_manipulation_checks.py` | `itembanks/asr.jsonl`, `descriptors/battery.csv`, `manifests/asr.csv` | `checks/inspection/` (60 mixed clips for listening), `checks/bg_presence.csv`, `checks/wer_constancy.csv`, `checks/determinism.json` | if available |
| `07_run_inference.py` | `manifests/`, `descriptors/battery.csv`, the audio in `data/` | `inference/<model>/<task>.jsonl`, `checks/mix_diagnostics.csv` | **required** |
| `08_score_outputs.py` | `inference/` | `results/scores/<task>.csv`: per stimulus, WER and ΔWER, or correctness and Δacc | — |
| `09_make_figures.py` | `results/scores/`, `descriptors/acoustic_descriptors.csv` | `results/figures/`, `results/tables/` (see [Outputs](#outputs-and-the-paper)) | — |

Stimuli are not stored in bulk. Stage 7 mixes each one on the fly from its
manifest row with `mixing/mix.py`. Stage 6 runs sanity checks on the mixed
stimuli before inference, two of them with Whisper `base`; no reported number
depends on it. Each stage script takes `--help`. The only options are Stage
4's `--sensitivity` (Silero VAD threshold, default 0.5) and Stage 7's `--model`
and `--task`.

Re-running a stage is safe. Stages 1–3 and 6 skip the steps recorded in
`checks/*_progress.json`, and Stage 5 keeps existing manifests. Stage 7
appends each result as it goes, and on a re-run it only runs the stimuli
missing from its output file. To redo a stage, delete its outputs and its
progress file.

### One model or one task

```bash
python scripts/07_run_inference.py --model gemma3n_e4b --task asr
python scripts/07_run_inference.py --model phi4_multimodal --task all
python scripts/07_run_inference.py --model all --task kws_steer_p5
```

`--model` takes a model id from the table above, or `all`. `--task` takes
`asr`, `kws`, `asr_steer_p5`, `kws_steer_p5` or `all`. Without options, Stage 7
runs `qwen25_omni_3b` on `asr`. Each model is loaded once for all its tasks
and freed before the next one. If a model fails to load (for example Gemma 3n
without access), Stage 7 logs the error and continues with the next model.

Stages 8 and 9 use every model and task found in `inference/`. Stage 9 needs
at least the `asr` and `kws` scores, and Figure 6 and Table 5 also need the two
steering tasks. The paper's numbers use all five models on all four tasks.

---

## Outputs and the paper

| Paper | Output | Content |
|---|---|---|
| Figure 1 | `assets/figure1.png` | pipeline diagram (static image) |
| Figure 2 | `results/figures/figure2_descriptor_regressions.png` | degradation against the 3 descriptors with the largest mean absolute correlation |
| Figure 3 | `results/figures/figure3_descriptor_correlations.png` | Spearman correlations, tasks × descriptors |
| Figure 4 | `results/figures/figure4_background_ranking.png` | backgrounds ranked by mean ASR/KWS degradation |
| Figure 5 | `results/figures/figure5_accent_degradation.png` | mean ΔWER per Common Voice accent group |
| Figure 6 | `results/figures/figure6_prompt_steering_rer.png` | RER per model and task |
| Figure 7 | `results/figures/figure7_snr_performance.png` | WER and KWS accuracy per SNR; as in the paper, panel (b) leaves out Qwen2-Audio |
| Table 1 | `BATTERY_SPEC` in `scripts/03_curate_backgrounds.py` | the 20 background recordings |
| Table 2 | `results/tables/table2_descriptor_correlations.csv` | Spearman correlations, pooled over the models and per model |
| Table 3 | `results/tables/table3_descriptor_values.csv` | descriptor values per background, in Figure 4 order |
| Table 4 | `results/tables/table4_accent_dwer.csv` | mean ΔWER per model for the 11 accent groups of Figure 5 |
| Table 5 | `results/tables/table5_prompt_steering_rer.csv` | RER for ASR and KWS per model |

---

## Repository layout

```
scripts/01_…09_*.py   the nine stages above
scripts/utils.py      shared paths and cache settings, logging, JSONL/CSV and audio I/O,
                      loudness normalisation, WER and closed-set KWS scoring
mixing/mix.py         SNR mixer; materialize() writes the exact stimulus of a manifest row
setup.py              environment setup (see Setup)
requirements.txt      pinned dependencies (PyTorch excluded)
index.html            project page
assets/               Figure 1 (figure1.png)
```

The generated folders (`data/`, `models/`, `itembanks/`, `descriptors/`,
`manifests/`, `checks/`, `inference/`, `results/`) and `.venv/` are git-ignored.

---

## Notes

- **Phi-4 remote code.** `microsoft/phi-4-multimodal-instruct` ships modeling
  code written for transformers 4.48. When Stage 7 loads Phi-4, it patches the
  downloaded files in `models/cache/hub/`: `eager` attention in `config.json`,
  one speech-encoder tensor created on the CPU, `lm_head` tied to the
  embeddings through the `_tied_weights_keys` dict that transformers 5
  expects, and `GenerationMixin` added to `Phi4MMModel`. It also restores the
  tokenizer's pre-tokenizer from `tokenizer.json` and passes
  `num_logits_to_keep=1` to `generate`. With these changes Phi-4 runs on
  transformers 5.12.1 as it did on 4.48.2, the version the paper's Phi-4 runs
  pinned. Each of the other patches at the top of the script has a comment.
- **Existing inference outputs.** Stage 7 writes the same files as the code
  that produced the paper's runs: the same paths, row fields, stimulus ids and
  task names (the steering tasks keep the names `asr_steer_p5` and
  `kws_steer_p5`). Existing `inference/<model>/<task>.jsonl` files therefore
  remain valid. Stage 7 runs only the missing stimuli, and Stages 8–9 score the
  files without a GPU. A row whose generation failed is saved with `raw` set to
  `__ERROR__`. The next Stage 7 run retries it, and Stage 8 ignores it.
- **Figure 1** is a static image (`assets/figure1.png`); no script draws it.
- **Unpinned sources.** The five models and Silero VAD (`snakers4/silero-vad`
  through PyTorch Hub) are fetched from the default branch of their
  repositories; no revision is pinned.

---

## Citation

```bibtex
@inproceedings{dsouza2026aipspeech,
  title     = {{AIP-Speech}: Auditing Speech {LLM} Robustness to Background Interference},
  author    = {Dsouza, Aiden and Mishra, Shreyansh and Suneja, Yashraj and Swain, Rohan and Borkar, Samyak and Jaishankar, Arjun and Urade, Tanmay},
  booktitle = {Proceedings of the 2026 Impact Speech Workshop: Identifying, Measuring, Preventing, and Assessing Consequences of Bias in Speech {LLM}s ({IMPACT-SPEECH})},
  year      = {2026},
  publisher = {Association for Computational Linguistics},
  note      = {To appear},
}
```
