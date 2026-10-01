#!/usr/bin/env python3
"""
07_run_inference.py — Stage 7: run Speech-LLM models over all manifests.

Reads manifests/asr.csv and manifests/kws.csv (Stage 5) and produces:
  inference/<model>/<task>.jsonl   (one row per stimulus)

Needs a CUDA GPU. All models are loaded one at a time, then freed
(del model; cuda.empty_cache()).
Output is written row-by-row → safe to resume after interruption.

Usage:
  python scripts/07_run_inference.py --model qwen2_audio_7b --task kws
  python scripts/07_run_inference.py --model all --task all   # full sweep
"""
from __future__ import annotations

import os
# read when torch initialises CUDA; an exported value takes precedence
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# huggingface_hub fixes its default cache at the transformers import below, before
# utils sets HF_HOME: make it _cache()'s folder, so files fetched without cache_dir
# (Phi-4's speech LoRA via load_adapter, Qwen2.5-Omni's spk_dict.pt) land there
# too. Not HF_HOME, which also locates the `hf auth login` token
os.environ.setdefault("HF_HUB_CACHE", os.path.join(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "models", "cache", "hub"))
import transformers.modeling_utils
# from_pretrained would first reserve GPU memory for the whole checkpoint (a
# loading-speed optimisation); the paper's runs loaded without that warm-up
transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None

# Shims for the transformers version pinned in requirements.txt.
# Phi-4's remote attention code calls DynamicCache.get_usable_length, removed in
# transformers 4.54. The value only chooses LongRoPE's short or long factors
# (long above 4096 tokens); returning the new length, as the paper's runs did,
# keeps the short factors, since 30 s of audio plus the longest prompt stays far
# below 2048 tokens
try:
    import transformers.cache_utils
    def _get_usable_length(self, seq_length: int, *args, **kwargs):
        return seq_length
    transformers.cache_utils.DynamicCache.get_usable_length = _get_usable_length
except (ImportError, AttributeError):
    pass

# transformers refuses torch.load on torch < 2.6 (CVE-2025-32434); Qwen2.5-Omni
# loads its speaker dictionary (spk_dict.pt) that way. No effect on torch >= 2.6
try:
    import transformers.utils.import_utils
    import transformers.utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
    transformers.utils.check_torch_load_is_safe = lambda *args, **kwargs: None
except (ImportError, AttributeError):
    pass


import argparse
import gc
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (ROOT, SR, get_logger, load_battery,
                   jsonl_ids, jsonl_append, load_audio)

sys.path.insert(0, str(ROOT / "mixing"))
from mix import mix, diagnostics

log = get_logger("07_run_inference")

MANIFESTS  = ROOT / "manifests"
INFER_DIR  = ROOT / "inference"

PROMPTS = {
    "asr":  "Transcribe the speech in this audio clip exactly as spoken. "
            "Return only the transcription text.",
    "kws":  "Listen to this audio clip. "
            "Respond with ONLY the spoken word (one word). "
            "If no clear word is spoken, respond with SILENCE.",
    # the steering prompts of the paper's prompt-steering experiment; the _p5
    # task names are kept so existing inference/<model>/*_steer_p5.jsonl
    # outputs stay valid
    "asr_steer_p5":
            "You are a highly reliable automatic speech recognition (ASR) system designed to accurately transcribe speech in noisy real-world environments. "
            "The provided audio may contain background conversations, multiple speakers, music, television, radio, announcements, traffic, machinery, environmental sounds, or other acoustic interference. "
            "Your task is to identify the intended foreground speaker (the speaker closest to the microphone or the dominant primary speaker) and transcribe ONLY that speaker's speech. "
            "Ignore all background voices, overlapping conversations, music, environmental noises, sound effects, and any speech that does not belong to the intended foreground speaker. "
            "Do not transcribe words originating from background speakers, even if they are clearly audible. Do not infer, hallucinate, or guess words that are masked by interference. If a foreground word is unintelligible due to noise, omit it rather than guessing. "
            "Internally determine which speaker is the primary foreground speaker before producing the transcription, and mentally separate foreground speech from all competing sounds. "
            "Your objective is to produce the same transcript that would have been obtained if the foreground speaker had been recorded in a completely silent environment. "
            "Output only the final transcript without explanations, notes, confidence scores, or any additional text.",
    "kws_steer_p5":
            "You are a highly reliable keyword spotting (KWS) system designed for noisy real-world environments. "
            "The provided audio may contain background conversations, multiple speakers, music, television, radio, announcements, traffic, machinery, environmental sounds, or other acoustic interference. "
            "Identify the intended foreground speaker and respond with ONLY the single keyword spoken by that speaker. "
            "Ignore all background voices, overlapping conversations, music, and environmental noises. "
            "If a foreground keyword is unintelligible due to noise, respond with SILENCE rather than guessing. "
            "Output only the final keyword without explanations, notes, or any additional text.",
}


# the paper's evaluation caps every stimulus at 30 s
MAX_AUDIO_SAMPLES = SR * 30   # 30 seconds

class SpeechLLM:
    """Abstract adapter. Subclasses load in __init__ and implement generate()."""
    model_id: str

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        import torch
        del self.model, self.processor
        gc.collect()
        torch.cuda.empty_cache()


def _cache() -> str:
    return str(ROOT / "models" / "cache" / "hub")


def _clip_audio(wav: np.ndarray) -> np.ndarray:
    """Truncate audio to MAX_AUDIO_SAMPLES (the 30 s cap)."""
    if len(wav) > MAX_AUDIO_SAMPLES:
        log.warning(f"[clip] Truncating audio from {len(wav)/SR:.1f}s to {MAX_AUDIO_SAMPLES/SR:.0f}s")
        return wav[:MAX_AUDIO_SAMPLES]
    return wav


class Qwen25Omni3B(SpeechLLM):
    model_id = "qwen25_omni_3b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B", cache_dir=_cache(), trust_remote_code=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-3B",
            cache_dir=_cache(),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text",  "text": task_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        wav = _clip_audio(wav)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, return_audio=False)
        if isinstance(out_ids, tuple):
            out_ids = out_ids[0]
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Qwen25Omni7B(SpeechLLM):
    model_id = "qwen25_omni_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5OmniForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-Omni-7B",
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio"},
            {"type": "text",  "text": task_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        wav = _clip_audio(wav)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, return_audio=False)
        if isinstance(out_ids, tuple):
            out_ids = out_ids[0]
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Qwen2Audio7B(SpeechLLM):
    model_id = "qwen2_audio_7b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration, BitsAndBytesConfig
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", cache_dir=_cache(), trust_remote_code=True)
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct",
            cache_dir=_cache(),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio_url": "__local__"},
            {"type": "text",  "text": task_prompt},
        ]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        wav = _clip_audio(wav)
        inputs = self.processor(text=text, audio=wav, sampling_rate=SR,
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


class Phi4Multimodal(SpeechLLM):
    model_id = "phi4_multimodal"

    def __init__(self):
        import torch, json, os, re
        from transformers import AutoProcessor, AutoModelForCausalLM
        from huggingface_hub import snapshot_download

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        MODEL_ID = "microsoft/phi-4-multimodal-instruct"

        model_dir = snapshot_download(
            MODEL_ID,
            cache_dir=_cache(),
        )

        # phi4's __init__ reads _attn_implementation straight off the config file,
        # and the baked-in "flash_attention_2" is a value its own code rejects
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            cfg_json = json.load(f)
        if cfg_json.get("_attn_implementation") not in (None, "eager", "sdpa"):
            cfg_json["_attn_implementation"] = "eager"
            cfg_json.pop("_attn_implementation_autoset", None)
            with open(config_path, "w") as f:
                json.dump(cfg_json, f, indent=2)

        # transformers builds the model on the meta device, where int() on this
        # encoder tensor throws — pin it to CPU instead
        encoder_path = os.path.join(model_dir, "speech_conformer_encoder.py")
        if os.path.exists(encoder_path):
            with open(encoder_path, "r") as f:
                encoder_code = f.read()
            if "torch.tensor(feat_in, dtype=torch.float)" in encoder_code:
                encoder_code = encoder_code.replace(
                    "torch.tensor(feat_in, dtype=torch.float)",
                    "torch.tensor(feat_in, dtype=torch.float, device='cpu')"
                )
                with open(encoder_path, "w") as f:
                    f.write(encoder_code)

        # The checkpoint has no lm_head.weight (tie_word_embeddings: lm_head
        # reuses the embeddings); transformers 5 ties it only through a
        # {target: source} _tied_weights_keys dict, not phi4's list. The regex
        # also restores the line earlier versions of this script commented out.
        # PreTrainedModel stopped inheriting GenerationMixin in 4.52, but phi4
        # wraps its inner Phi4MMModel in a CAUSAL_LM peft model, which reads
        # prepare_inputs_for_generation from it
        modeling_path = os.path.join(model_dir, "modeling_phi4mm.py")
        if os.path.exists(modeling_path):
            with open(modeling_path, "r") as f:
                mod_code = f.read()
            patched = re.sub(
                r'^(\s*)(?:# )?_tied_weights_keys = \["lm_head\.weight"\]$',
                r'\1_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}',
                mod_code, flags=re.M,
            ).replace(
                "class Phi4MMModel(Phi4MMPreTrainedModel):",
                "class Phi4MMModel(Phi4MMPreTrainedModel, GenerationMixin):"
            )
            if patched != mod_code:
                with open(modeling_path, "w") as f:
                    f.write(patched)

        # load by model id, not by path — the trust_remote_code loader needs
        # that to resolve the symlinked .py files in the Hugging Face cache
        self.processor = AutoProcessor.from_pretrained(
            MODEL_ID, cache_dir=_cache(), trust_remote_code=True)
        # transformers 5 rebuilds phi4's GPT-2-style tokenizer with GPT-2's own
        # pre-tokenizer instead of the one in its tokenizer.json, which the
        # paper's runs (transformers 4.48) used; without this, words such as
        # "real-world" in the steering prompts are split into other tokens
        from tokenizers import Tokenizer
        self.processor.tokenizer.backend_tokenizer.pre_tokenizer = Tokenizer.from_file(
            os.path.join(model_dir, "tokenizer.json")).pre_tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            cache_dir=_cache(),
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(self.device)
        self.model.load_adapter(MODEL_ID, adapter_name="speech", adapter_kwargs={"subfolder": "speech-lora"})
        self.model.set_adapter("speech")
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": f"<|audio_1|>{task_prompt}"}]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        wav = _clip_audio(wav)
        inputs = self.processor(text=prompt, audios=[(wav, SR)],
                                return_tensors="pt").to(self.device)
        with torch.no_grad():
            # transformers 4.48 generate() passed num_logits_to_keep=1 itself;
            # since 4.49 it sets logits_to_keep, and phi4's forward fails on None
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False, num_logits_to_keep=1)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.tokenizer.decode(out[0], skip_special_tokens=True).strip()


class Gemma3nE4B(SpeechLLM):
    model_id = "gemma3n_e4b"

    def __init__(self):
        import torch
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(
            "google/gemma-3n-E4B-it", cache_dir=_cache(), trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            "google/gemma-3n-E4B-it",
            cache_dir=_cache(),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, wav: np.ndarray, task_prompt: str,
                 system_prompt: str | None = None, max_new_tokens: int = 64) -> str:
        import torch
        messages = [{"role": "user", "content": [
            {"type": "audio", "audio": _clip_audio(wav)},
            {"type": "text",  "text": task_prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                          do_sample=False)
        out = out_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.decode(out[0], skip_special_tokens=True).strip()


MODEL_CLASSES: dict[str, type[SpeechLLM]] = {
    "qwen25_omni_3b":  Qwen25Omni3B,
    "qwen25_omni_7b":  Qwen25Omni7B,
    "qwen2_audio_7b":  Qwen2Audio7B,
    "phi4_multimodal": Phi4Multimodal,
    "gemma3n_e4b":     Gemma3nE4B,
}
ALL_MODELS = list(MODEL_CLASSES.keys())
ALL_TASKS  = ["asr", "kws", "asr_steer_p5", "kws_steer_p5"]


def _base_task(task: str) -> str:
    """Steering runs replay the manifest of their baseline task."""
    return "kws" if task.startswith("kws") else "asr"


def run_inference_with_model(model: SpeechLLM, model_id: str, task: str) -> None:
    run_key = f"{model_id}_{task}"

    import pandas as pd
    manifest = pd.read_csv(MANIFESTS / f"{_base_task(task)}.csv")

    out_path = INFER_DIR / model_id / f"{task}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = jsonl_ids(out_path)
    remaining = manifest[~manifest["id"].isin(done)]

    if remaining.empty:
        log.info(f"[skip] {run_key} already complete ({len(done)} rows).")
        return

    log.info(f"[infer] {model_id} / {task}: {len(remaining)} rows remaining.")

    prompt = PROMPTS[task]
    battery = {b["bg_id"]: b for b in load_battery()}

    total_remaining = len(remaining)
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        try:
            sp_path_str = str(row["speech_path"]).replace("\\", "/")
            sp = load_audio(ROOT / sp_path_str)
            bg_id = row["background_id"]
            if row["condition"] == "clean" or bg_id == "clean":
                bg = None
            else:
                bg_wav_rel = battery.get(bg_id, {}).get("wav", "")
                if bg_wav_rel:
                    bg_path_str = str(bg_wav_rel).replace("\\", "/")
                    bg = load_audio(ROOT / bg_path_str)
                else:
                    bg = None

            wav = mix(sp, bg, float(row["snr_db"]), int(row["seed"]))

            diag_row = dict(row)
            diag_row["id"] = row["id"]
            diag_row["condition"] = row.get("condition", "noisy")
            diagnostics(diag_row, wav, sp)

            raw = model.generate(wav, prompt)
            log.info(f"[save] {model_id} / {task} ({i}/{total_remaining}) - {row['id']} -> '{raw}' saved to {out_path.relative_to(ROOT)}")
            out_row = {
                **{k: row[k] for k in row.index},
                "raw": raw,
                "model": model_id,
                "task": task,
            }
            jsonl_append(out_path, out_row)
        except Exception as e:
            import traceback
            log.warning(f"[error] {model_id} / {task} - {row['id']}: {e}\n{traceback.format_exc()}")
            jsonl_append(out_path, {
                "id": row["id"], "raw": "__ERROR__", "error": str(e),
                "model": model_id, "task": task,
            })
            # without this, one OOM cascades into every remaining row
            if "CUDA out of memory" in str(e) or "OutOfMemoryError" in type(e).__name__:
                import torch
                gc.collect()
                torch.cuda.empty_cache()
                log.info(f"[recovery] Cleared CUDA cache after OOM on {row['id']}")

    log.info(f"[done] {run_key} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage 7 — Inference")
    ap.add_argument("--model", default="qwen25_omni_3b",
                    choices=ALL_MODELS + ["all"], help="Model to run")
    ap.add_argument("--task",  default="asr",
                    choices=ALL_TASKS + ["all"], help="Task to run")
    args = ap.parse_args()

    models = ALL_MODELS if args.model == "all" else [args.model]
    tasks  = ALL_TASKS  if args.task  == "all" else [args.task]

    for task in tasks:
        manifest_file = MANIFESTS / f"{_base_task(task)}.csv"
        if not manifest_file.exists():
            log.error(f"[infer] Manifest not found: {manifest_file}. "
                      "Run scripts/05_build_manifests.py first.")
            sys.exit(1)

    for model_id in models:
        # skip loading the model at all if nothing is left to run
        tasks_to_run = []
        for task in tasks:
            import pandas as pd
            manifest = pd.read_csv(MANIFESTS / f"{_base_task(task)}.csv")
            out_path = INFER_DIR / model_id / f"{task}.jsonl"
            done = jsonl_ids(out_path)
            remaining = manifest[~manifest["id"].isin(done)]

            if not remaining.empty:
                tasks_to_run.append(task)

        if not tasks_to_run:
            log.info(f"[skip] {model_id}: all tasks already complete.")
            continue

        import torch
        if not torch.cuda.is_available():
            log.error("[infer] PyTorch sees no CUDA GPU, and the models run on one. "
                      "Run `python setup.py` to check the PyTorch/CUDA install.")
            sys.exit(1)

        log.info(f"=== Loading model: {model_id} ===")
        cls = MODEL_CLASSES.get(model_id)
        if cls is None:
            log.error(f"Unknown model: {model_id}")
            continue
        try:
            model = cls()
            log.info(f"[model] {model_id} loaded.")
        except Exception as e:
            import traceback
            log.error(f"[model] Failed to load {model_id}: {e}")
            log.error(traceback.format_exc())
            continue

        try:
            for task in tasks_to_run:
                log.info(f"=== {model_id} / {task} ===")
                run_inference_with_model(model, model_id, task)
        finally:
            model.unload()
            log.info(f"[infer] {model_id} unloaded.")

    log.info("=== Stage 7 complete. ===")


if __name__ == "__main__":
    main()
