# results.md — AIP-Speech Experimental Results

> This table was drafted from the original 8-stage plan and was never
> populated by the pipeline that actually produced the paper's results
> (`scripts/generate_plots_new.py` + `scripts/generate_results_new_2.py`,
> which write to `results/plots_new/` and `results_new/` as CSVs/figures
> instead). Kept here as a historical record of the planned experiment
> structure.

---

## Manipulation Checks (`checks/`)

| Check | Status | Notes |
|-------|--------|-------|
| Background presence | ⬜ pending | |
| WER constancy | ⬜ pending | |

| Determinism floor | ⬜ pending | |

---

## E1 — Interference Profiling

### T1.1 ASR (LibriSpeech)

| Model | Clean WER | SNR+10 WER | SNR+5 WER | SNR0 WER | Worst-bg ΔWER |
|-------|-----------|------------|-----------|----------|---------------|
| Qwen2.5-Omni-3B | — | — | — | — | — |
| Qwen2-Audio-7B | — | — | — | — | — |
| Phi-4-multimodal | — | — | — | — | — |
| Gemma-3n-E4B | — | — | — | — | — |

### T1.2 KWS (Google Speech Commands v2)

| Model | Clean Acc | FAR | Miss Rate | Worst-bg ΔACC |
|-------|-----------|-----|-----------|----------------|
| Qwen2.5-Omni-3B | — | — | — | — |
| Qwen2-Audio-7B | — | — | — | — |
| Phi-4-multimodal | — | — | — | — |
| Gemma-3n-E4B | — | — | — | — |

---

## E2 — Semantic vs Energetic

### T2.1 ASR: semantic gap @ 0 dB

| Background | ΔWER | Gap (semantic) |
|------------|------|----------------|
| PCAFETER | — | — |
| PRESTO | — | — |
| SPSQUARE | — | — |
| OMEETING | — | — |
| NOISEX-babble | — | — |

### T2.2 KWS: trigger injection

| Background | TIR | TIR (generic) |
|------------|------------|-----------------|---------------|
| Injection probes | — | — | — |

---

## E3 — Disparate Robustness

### T3.1 Ecological (Common Voice, accent/gender)

| Subgroup | Clean WER | ΔWER speech-like | ΔWER non-speech | Robustness Gap | DRI |
|----------|-----------|------------------|-----------------|----------------|-----|
| — | — | — | — | — | — |

---

## E4 — Steerability (stretch)

| Model | Effect w/o instruction | Effect w/ instruction | RER |
|-------|------------------------|----------------------|-----|
| — | — | — | — |

---

## C-PROFILE Regression (E1)

| Descriptor | ASR coeff (std) | KWS coeff (std) | p-value | VIF |
|------------|-----------------|-----------------|---------|-----|
| speech_likeness | — | — | — | — |
| linguistic_content | — | — | — | — |
| mod_2to8Hz | — | — | — | — |
| spectral_overlap | — | — | — | — |
| stationarity | — | — | — | — |
| onset_density | — | — | — | — |
| loudness | — | — | — | — |
| SNR | — | — | — | — |

---

## Day-6 Go/No-Go Decision

- [ ] Descriptor law found beyond SNR?
- [ ] Disparity signal found?
- [ ] Decision: **proceed** / **pivot to robustness-null**
