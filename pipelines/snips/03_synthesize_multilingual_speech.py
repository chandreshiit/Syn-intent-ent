#!/usr/bin/env python3
"""
Synthesize speech for SNIPS smart-lights multilingual commands using Facebook MMS-TTS.

Mirrors data/multiatis_multilingual_pipeline/03_synthesize_multilingual_speech.py,
but only supports English and French (the two languages in scope for the SNIPS
multilingual experiment). Single fixed voice per language: MMS-TTS is single-speaker
per checkpoint; the multilingual claim does not depend on speaker variability.

Output: 16 kHz mono WAV per utterance (matches original SNIPS close-field recordings)
plus audio_metadata.json.
"""

import json
import os
import argparse
import torch
import soundfile as sf
import scipy.io.wavfile
from transformers import VitsModel, AutoTokenizer, set_seed
from tqdm import tqdm


TORCH_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MMS_TTS_MODELS = {
    "en": "facebook/mms-tts-eng",
    "fr": "facebook/mms-tts-fra",
}

LANGUAGE_NAME_TO_CODE = {
    "english": "en",
    "french": "fr",
}

TARGET_SR = 16000  # SNIPS close-field standard


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_mms_model(language_code, token=None):
    model_name = MMS_TTS_MODELS.get(language_code)
    if not model_name:
        print(f"Warning: No MMS-TTS model for language code '{language_code}'")
        return None, None
    print(f"Loading MMS-TTS model for {language_code}: {model_name}")
    try:
        model = VitsModel.from_pretrained(model_name, token=token).to(TORCH_DEVICE)
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    except Exception as e:
        print(f"*** Failed to load model for {language_code} ({model_name}): {e}")
        return None, None
    return model, tokenizer


def generate_speech(model, tokenizer, text, output_file):
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {
        k: (v.to(TORCH_DEVICE).long() if v.dtype == torch.float32 and k == "input_ids" else v.to(TORCH_DEVICE))
        for k, v in inputs.items()
    }
    with torch.no_grad():
        output = model(**inputs).waveform
    audio_np = output.cpu().to(torch.float32).numpy().squeeze()

    src_sr = model.config.sampling_rate
    if src_sr != TARGET_SR:
        # MMS-TTS produces 16 kHz natively, but downsample/resample if a different
        # checkpoint is ever swapped in (defensive).
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(src_sr, TARGET_SR)
        up = TARGET_SR // g
        down = src_sr // g
        audio_np = resample_poly(audio_np, up=up, down=down)

    try:
        sf.write(output_file, audio_np, TARGET_SR)
    except Exception as e:
        print(f"sf.write failed: {e}; falling back to scipy.io.wavfile")
        if audio_np.ndim == 1:
            audio_np = audio_np.reshape(-1, 1)
        scipy.io.wavfile.write(output_file, rate=TARGET_SR, data=audio_np)

    return {
        "audio_file": output_file,
        "text": text,
        "sampling_rate": TARGET_SR,
        "audio_length": float(len(audio_np)) / TARGET_SR,
    }


def main(jsonl_path, languages, output_dir, resume=False, token=None):
    print(f"Starting MMS-TTS generation. Device: {TORCH_DEVICE}")
    os.makedirs(output_dir, exist_ok=True)

    data = load_jsonl(jsonl_path)
    print(f"Loaded {len(data)} entries from {jsonl_path}")

    lang_codes = [LANGUAGE_NAME_TO_CODE.get(l.lower(), l.lower()) for l in languages]
    print(f"Languages requested: {languages} -> {lang_codes}")

    filtered = []
    for entry in data:
        entry_lang = entry.get("language", "").lower()
        entry_code = entry.get("language_code", "")
        lang_code = LANGUAGE_NAME_TO_CODE.get(entry_lang, entry_code)
        if lang_code in lang_codes:
            entry["_resolved_lang_code"] = lang_code
            filtered.append(entry)
    print(f"Processing {len(filtered)} entries from {len(data)} total")

    by_lang = {}
    for entry in filtered:
        by_lang.setdefault(entry["_resolved_lang_code"], []).append(entry)

    metadata = []
    for lang_code, entries in by_lang.items():
        print(f"\n{'=' * 60}\nProcessing {lang_code} ({len(entries)} entries)\n{'=' * 60}")
        model, tokenizer = load_mms_model(lang_code, token=token)
        if model is None:
            print(f"Skipping {lang_code} - no model available")
            continue

        skipped = 0
        errors = 0
        bar = tqdm(total=len(entries), desc=f"Generating {lang_code} speech", unit="audio")
        for i, entry in enumerate(entries):
            command_id = f"cmd_{i:04d}_{lang_code}"
            text = entry.get("translated_command", "")
            english_cmd = entry.get("english_command", f"cmd_{i}")
            if not text:
                bar.update(1)
                continue

            output_file = os.path.join(output_dir, f"{command_id}.wav")
            if resume and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                skipped += 1
                bar.update(1)
                continue

            try:
                info = generate_speech(model, tokenizer, text, output_file)
                metadata.append({
                    "command_id": command_id,
                    "original_command": english_cmd,
                    "language": lang_code,
                    "text": text,
                    "intent": entry.get("intent", ""),
                    "category": entry.get("category", ""),
                    "audio_file": os.path.basename(output_file),
                    "audio_length": info["audio_length"],
                    "sampling_rate": info["sampling_rate"],
                    "tts_engine": "mms-tts",
                })
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"Error generating speech for entry {i} ({lang_code}): {e}")
                elif errors == 6:
                    print(f"  ... suppressing further errors for {lang_code}")
            bar.update(1)
        bar.close()

        if resume and skipped > 0:
            print(f"  Resumed: skipped {skipped} already-generated files for {lang_code}")
        if errors > 0:
            print(f"  Errors: {errors} entries failed for {lang_code}")

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metadata_file = os.path.join(output_dir, "audio_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"\nGenerated {len(metadata)} audio files. Metadata saved to {metadata_file}")

    lang_counts = {}
    for entry in metadata:
        lang = entry["language"]
        lang_counts[lang] = lang_counts.get(lang, 0) + 1
    print("\nLanguage breakdown:")
    for lang, count in sorted(lang_counts.items()):
        print(f"  {lang}: {count} audio files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SNIPS smart-lights multilingual TTS via MMS-TTS")
    parser.add_argument("--jsonl", type=str,
                        default="data/snips_multilingual_pipeline/snips_bio_all_languages.json",
                        help="Path to input JSON file from step 02")
    parser.add_argument("--output_dir", type=str,
                        default="data/snips_multilingual_pipeline/generated_audio",
                        help="Directory to save generated audio files")
    parser.add_argument("--languages", type=str, nargs="+", default=["english", "french"],
                        help="Languages to generate speech for")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-generated audio files")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace access token (if any model is gated)")
    args = parser.parse_args()

    set_seed(args.seed)
    main(args.jsonl, args.languages, args.output_dir, resume=args.resume, token=args.token)
