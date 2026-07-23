#!/usr/bin/env python3
"""
Process SNIPS smart-lights multilingual BIO-tagged data into the BIO format
expected by the JointBERT and Whisper training pipelines.

For the multilingual SNIPS experiment we do NOT pre-bake a train/dev/test split,
because the reviewer-requested 5-fold cross-validation slicing happens in the
baseline kfold script ([baselines/kfold_evaluation.py]). This step instead:
  1. Emits per-language and combined BIO directories (seq.in, seq.out, label)
  2. Emits audio metadata files (one per language + combined) for Whisper
  3. Writes intent_label.txt and slot_label.txt (full slot ontology: room, color, brightness)
  4. Writes a dataset_statistics.txt summary
"""

import json
import os
import re
import argparse
from collections import defaultdict, Counter


def load_language_config(config_path):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_word_tokenize(text):
    return re.findall(r"\w+(?:'\w+)?", text)


def tokenize_by_language(text, language_code, lang_config):
    tokenization = "whitespace"
    for _, lang_info in lang_config.get("languages", {}).items():
        if lang_info.get("code") == language_code:
            tokenization = lang_info.get("tokenization", "whitespace")
            break
    if tokenization == "character":
        return [c for c in text if c.strip()]
    return simple_word_tokenize(text.lower())


def write_bio_split(split_dir, data, lang_config):
    """Write seq.in, seq.out, label files for the given list of entries."""
    os.makedirs(split_dir, exist_ok=True)
    with open(os.path.join(split_dir, "seq.in"), "w", encoding="utf-8") as fin, \
         open(os.path.join(split_dir, "seq.out"), "w", encoding="utf-8") as fout, \
         open(os.path.join(split_dir, "label"), "w", encoding="utf-8") as flabel:
        for item in data:
            tokens = item.get("tokens")
            if not tokens:
                tokens = tokenize_by_language(item.get("translated_command", ""), item.get("language_code", "en"), lang_config)
            bio_tags = item.get("bio_tags", "")
            if isinstance(bio_tags, list):
                bio_tags = " ".join(bio_tags)
            if not bio_tags:
                bio_tags = " ".join(["O"] * len(tokens))
            tag_list = bio_tags.split()
            if len(tag_list) < len(tokens):
                tag_list = tag_list + ["O"] * (len(tokens) - len(tag_list))
            elif len(tag_list) > len(tokens):
                tag_list = tag_list[:len(tokens)]
            fin.write(" ".join(tokens) + "\n")
            fout.write(" ".join(tag_list) + "\n")
            flabel.write(str(item.get("intent", "UNK")).lower() + "\n")


def write_label_files(output_dir, data):
    intents = sorted(set(str(item.get("intent", "UNK")).lower() for item in data))
    intents = ["UNK"] + intents
    with open(os.path.join(output_dir, "intent_label.txt"), "w", encoding="utf-8") as f:
        for intent in intents:
            f.write(intent + "\n")

    all_tags = []
    for item in data:
        bio_tags = item.get("bio_tags", "O")
        if isinstance(bio_tags, list):
            bio_tags = " ".join(bio_tags)
        all_tags.extend(bio_tags.split())
    unique_tags = sorted(set(all_tags))
    unique_tags = ["PAD", "UNK"] + unique_tags
    with open(os.path.join(output_dir, "slot_label.txt"), "w", encoding="utf-8") as f:
        for tag in unique_tags:
            f.write(tag + "\n")


def write_audio_metadata(output_dir, data, audio_dir, lang_code=None):
    """Write audio_metadata.json suitable for Whisper consumption."""
    metadata = []
    for i, item in enumerate(data):
        lang = item.get("language_code", "en").lower()
        if lang_code is not None and lang != lang_code:
            continue
        cmd_id = f"cmd_{i:04d}_{lang}"
        metadata.append({
            "command_id": cmd_id,
            "original_command": item.get("english_command", ""),
            "language": lang,
            "text": item.get("translated_command", ""),
            "intent": item.get("intent", ""),
            "category": item.get("category", ""),
            "audio_file": f"{cmd_id}.wav",
            "audio_dir": audio_dir,
            "sampling_rate": 16000,
        })
    return metadata


def write_statistics(data, output_dir, lang_config):
    stats_file = os.path.join(output_dir, "dataset_statistics.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("SNIPS Smart-Lights Synthetic Multilingual Dataset Statistics\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total entries: {len(data)}\n\n")

        f.write("Per-Language Statistics:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Language':<15} {'Count':<8} {'Intents':<10} {'Slots':<8}\n")
        f.write("-" * 70 + "\n")
        lang_codes = sorted(set(item.get("language_code", "en") for item in data))
        for lang_code in lang_codes:
            lang_data = [item for item in data if item.get("language_code", "") == lang_code]
            unique_intents = len(set(str(item.get("intent", "")).lower() for item in lang_data))
            all_tags = set()
            for item in lang_data:
                bio_tags = item.get("bio_tags", "")
                if isinstance(bio_tags, list):
                    bio_tags = " ".join(bio_tags)
                for tag in bio_tags.split():
                    if tag.startswith("B-") or tag.startswith("I-"):
                        all_tags.add(tag[2:])
            lang_name = lang_code
            for _, info in lang_config.get("languages", {}).items():
                if info.get("code") == lang_code:
                    lang_name = info.get("name", lang_code)
                    break
            f.write(f"{lang_name:<15} {len(lang_data):<8} {unique_intents:<10} {len(all_tags):<8}\n")
        f.write("\n")

        intent_counts = Counter(str(item.get("intent", "")).lower() for item in data)
        f.write("Intent Distribution:\n")
        f.write("-" * 40 + "\n")
        for intent, count in sorted(intent_counts.items()):
            f.write(f"  {intent:<25} {count:>6}\n")
        f.write("\n")

        slot_counts = Counter()
        for item in data:
            bio_tags = item.get("bio_tags", "")
            if isinstance(bio_tags, list):
                bio_tags = " ".join(bio_tags)
            for tag in bio_tags.split():
                if tag.startswith("B-"):
                    slot_counts[tag[2:]] += 1
        f.write("Slot Type Distribution (B- tag counts):\n")
        f.write("-" * 40 + "\n")
        for slot, count in sorted(slot_counts.items()):
            f.write(f"  {slot:<15} {count:>6}\n")
        f.write("\n")

        multislot = sum(
            1 for item in data
            if sum(1 for t in (item.get("bio_tags", "") if isinstance(item.get("bio_tags", ""), str) else " ".join(item.get("bio_tags", []))).split()
                   if t.startswith("B-")) >= 2
        )
        f.write(f"Multi-slot utterances (>=2 slots): {multislot}\n")

    print(f"Statistics written to {stats_file}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process SNIPS multilingual BIO-tagged data")
    parser.add_argument("--input", type=str,
                        default="data/snips_multilingual_pipeline/snips_bio_all_languages.json",
                        help="Input BIO JSON from step 02")
    parser.add_argument("--output_dir", type=str,
                        default="data/snips_multilingual_pipeline/processed_data",
                        help="Output directory")
    parser.add_argument("--audio_dir", type=str,
                        default="data/snips_multilingual_pipeline/generated_audio",
                        help="Audio directory (path stored in metadata)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "language_config.json"),
                        help="Language configuration file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    lang_config = load_language_config(args.config)

    print(f"Loading BIO data from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries")

    os.makedirs(args.output_dir, exist_ok=True)

    # Per-language BIO directories
    by_lang = defaultdict(list)
    for item in data:
        by_lang[item.get("language_code", "en")].append(item)

    for lang_code, lang_data in by_lang.items():
        lang_dir = os.path.join(args.output_dir, lang_code)
        write_bio_split(os.path.join(lang_dir, "all"), lang_data, lang_config)

    # Combined BIO directory (all languages mixed)
    write_bio_split(os.path.join(args.output_dir, "combined", "all"), data, lang_config)

    # Label files
    write_label_files(args.output_dir, data)

    # Audio metadata
    combined_meta = write_audio_metadata(args.output_dir, data, args.audio_dir)
    with open(os.path.join(args.output_dir, "audio_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(combined_meta, f, ensure_ascii=False, indent=2)

    for lang_code in by_lang:
        lang_meta = write_audio_metadata(args.output_dir, data, args.audio_dir, lang_code=lang_code)
        with open(os.path.join(args.output_dir, f"audio_metadata_{lang_code}.json"), "w", encoding="utf-8") as f:
            json.dump(lang_meta, f, ensure_ascii=False, indent=2)

    write_statistics(data, args.output_dir, lang_config)

    print(f"\nDone. Output in {args.output_dir}")
    print("  - {lang_code}/all/ (per-language BIO: seq.in, seq.out, label)")
    print("  - combined/all/ (all languages mixed)")
    print("  - intent_label.txt, slot_label.txt")
    print("  - audio_metadata.json + audio_metadata_{lang_code}.json")
    print("  - dataset_statistics.txt")


if __name__ == "__main__":
    main()
