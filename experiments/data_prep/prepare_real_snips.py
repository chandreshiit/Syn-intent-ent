#!/usr/bin/env python3
"""
Prepare real SNIPS smart-lights (close-field) for our 5-fold CV baselines.

Reads:
  SNIPS/smart-lights-en-close-field/dataset.json     (Snips NLU schema)
  SNIPS/smart-lights-en-close-field/speech_corpus/   (audio + metadata.json)

Writes:
  data/snips_real_close/
    bio/seq.in
    bio/seq.out
    bio/label
    audio_index.json    {bio_row_index: "/abs/path/to/wav"}
    dataset.json        (copy of original, kept for Snips NLU evaluation)
    summary.json
"""

import argparse
import json
import os
import re
import shutil
import sys

SNIPS_ROOT = "SNIPS/smart-lights-en-close-field"
OUT_DIR = "data/snips_real_close"


def tokenize(text):
    # Whitespace + simple punctuation split, matching how SNIPS NLU tokenizes
    # for slot extraction. Apostrophes preserved (don't -> don't).
    return re.findall(r"\w+(?:'\w+)?|[^\w\s]", text, flags=re.UNICODE)


def utterance_to_bio(u):
    """Convert one dataset.json utterance dict to (tokens, BIO tags, text)."""
    tokens = []
    tags = []
    full_text_parts = []
    for chunk in u.get("data", []):
        chunk_text = chunk.get("text", "")
        full_text_parts.append(chunk_text)
        chunk_tokens = tokenize(chunk_text)
        if not chunk_tokens:
            continue
        slot = chunk.get("slot_name")
        if slot:
            tokens.extend(chunk_tokens)
            tags.append(f"B-{slot}")
            tags.extend([f"I-{slot}"] * (len(chunk_tokens) - 1))
        else:
            tokens.extend(chunk_tokens)
            tags.extend(["O"] * len(chunk_tokens))
    return tokens, tags, "".join(full_text_parts).strip()


def build_audio_index(text_to_audiopath):
    """Build a fast lookup keyed by canonicalized text."""
    return {canonicalize(k): v for k, v in text_to_audiopath.items()}


def canonicalize(s):
    # Lower, collapse whitespace, strip trailing punctuation for matching.
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[\.\?!]+$", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snips-root", default=SNIPS_ROOT)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--max-warnings", type=int, default=20)
    args = ap.parse_args()

    dataset_path = os.path.join(args.snips_root, "dataset.json")
    metadata_path = os.path.join(args.snips_root, "speech_corpus", "metadata.json")
    audio_dir = os.path.join(args.snips_root, "speech_corpus", "audio")

    if not os.path.exists(dataset_path):
        sys.exit(f"dataset.json missing: {dataset_path}")
    if not os.path.exists(metadata_path):
        sys.exit(f"metadata.json missing: {metadata_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        ds = json.load(f)
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Map: canonical text -> absolute audio path
    audio_lookup = {}
    for key, entry in meta.items():
        wav_name = entry.get("filename") or entry.get("file") or f"{key}.wav"
        wav_path = os.path.abspath(os.path.join(audio_dir, wav_name))
        # The dataset and metadata both use "text" field
        for txt_field in ("text", "sentence"):
            txt = entry.get(txt_field)
            if txt:
                audio_lookup.setdefault(canonicalize(txt), wav_path)

    out_bio = os.path.join(args.out_dir, "bio")
    os.makedirs(out_bio, exist_ok=True)

    seq_in_lines = []
    seq_out_lines = []
    label_lines = []
    audio_index = {}
    warnings = 0
    n_per_intent = {}
    n_audio_missing = 0
    audio_missing_examples = []

    intents_order = list(ds["intents"].keys())  # deterministic order
    for intent_name in intents_order:
        utts = ds["intents"][intent_name]["utterances"]
        n_per_intent[intent_name] = len(utts)
        for u in utts:
            tokens, tags, full_text = utterance_to_bio(u)
            if not tokens:
                if warnings < args.max_warnings:
                    print(f"  WARN empty utterance under {intent_name}: {u}")
                warnings += 1
                continue
            row = len(seq_in_lines)
            seq_in_lines.append(" ".join(tokens))
            seq_out_lines.append(" ".join(tags))
            # Match our existing synthetic intent label convention: lowercase, no underscores.
            label_lines.append(intent_name.lower())
            wav_path = audio_lookup.get(canonicalize(full_text))
            if wav_path and os.path.exists(wav_path):
                audio_index[str(row)] = wav_path
            else:
                n_audio_missing += 1
                if len(audio_missing_examples) < 10:
                    audio_missing_examples.append(full_text)

    with open(os.path.join(out_bio, "seq.in"), "w", encoding="utf-8") as f:
        f.write("\n".join(seq_in_lines) + "\n")
    with open(os.path.join(out_bio, "seq.out"), "w", encoding="utf-8") as f:
        f.write("\n".join(seq_out_lines) + "\n")
    with open(os.path.join(out_bio, "label"), "w", encoding="utf-8") as f:
        f.write("\n".join(label_lines) + "\n")

    with open(os.path.join(args.out_dir, "audio_index.json"), "w", encoding="utf-8") as f:
        json.dump(audio_index, f, indent=2)

    # Copy the original dataset.json for Snips NLU eval
    shutil.copy(dataset_path, os.path.join(args.out_dir, "dataset.json"))

    summary = {
        "n_utterances": len(seq_in_lines),
        "n_per_intent": n_per_intent,
        "n_audio_indexed": len(audio_index),
        "n_audio_missing": n_audio_missing,
        "audio_missing_examples": audio_missing_examples,
        "warnings": warnings,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
