#!/usr/bin/env python3
"""
Cross-lingual slot projection for SNIPS smart-lights multilingual dataset.

Logic is a thin subset of data/multiatis_multilingual_pipeline/02_project_slots_crosslingual.py:
- Whitespace tokenization (EN and FR both use whitespace; no CJK in scope)
- Multi-slot projection per utterance (room, color, brightness)
- BIO tags generated using the slot_translations field from step 01

For English entries (identity translation), the bio_tags from step 00 are kept
as-is so source-stage tagging is preserved.
"""

import json
import os
import re
import unicodedata
import argparse
from collections import Counter
from tqdm import tqdm


def load_language_config(config_path):
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def strip_accents(text):
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def tokenize_by_language(text, language_code, lang_config):
    """Whitespace tokenization for SNIPS multilingual (EN, FR)."""
    tokenization = "whitespace"
    for _, lang_info in lang_config.get("languages", {}).items():
        if lang_info.get("code") == language_code:
            tokenization = lang_info.get("tokenization", "whitespace")
            break
    if tokenization == "character":
        return [c for c in text if c.strip()]
    return re.findall(r"\w+(?:'\w+)?", text.lower())


def find_value_in_tokens(tokens, value):
    """Return (start_idx, end_idx) of value within tokens, or None.

    Match strategies in order:
      1. Exact accent-normalized token-sequence match
      2. Single-token equality on accent-normalized form
      3. Substring search inside apostrophe-fused tokens (l'X, d'X, m'X, etc.)
      4. Hyphenated value -> split hyphens and retry sequence match
      5. Reverse-order match (for cases like 'arrière-cour' vs 'cour arrière')
    """
    if not value or value.lower() == "none":
        return None
    value_tokens = re.findall(r"\w+(?:'\w+)?", value.lower())
    if not value_tokens:
        return None

    norm_tokens = [strip_accents(t) for t in tokens]
    norm_value_tokens = [strip_accents(v) for v in value_tokens]

    # 1. Sequence match
    for i in range(len(norm_tokens) - len(norm_value_tokens) + 1):
        if norm_tokens[i:i + len(norm_value_tokens)] == norm_value_tokens:
            return (i, i + len(norm_value_tokens))

    # 2. Single-token equality
    if len(norm_value_tokens) == 1:
        for i, token in enumerate(norm_tokens):
            if token == norm_value_tokens[0]:
                return (i, i + 1)
        # 3. Apostrophe-fused token: e.g. "l'entree" contains "entree"
        for i, token in enumerate(norm_tokens):
            if "'" in token:
                parts = token.split("'")
                if norm_value_tokens[0] in parts:
                    return (i, i + 1)

    # 4. Hyphenated value: split hyphens and retry
    if any("-" in v for v in value_tokens):
        flat_value = " ".join(v.replace("-", " ") for v in value_tokens)
        sub_tokens = re.findall(r"\w+", flat_value.lower())
        norm_sub = [strip_accents(t) for t in sub_tokens]
        if norm_sub and norm_sub != norm_value_tokens:
            for i in range(len(norm_tokens) - len(norm_sub) + 1):
                if norm_tokens[i:i + len(norm_sub)] == norm_sub:
                    return (i, i + len(norm_sub))
            # 5. Reverse-order (e.g. arriere-cour -> cour arriere)
            rev_sub = list(reversed(norm_sub))
            for i in range(len(norm_tokens) - len(rev_sub) + 1):
                if norm_tokens[i:i + len(rev_sub)] == rev_sub:
                    return (i, i + len(rev_sub))

    return None


def create_bio_tags_multislot(item, lang_config):
    """Project slots cross-lingually and generate BIO tags."""
    language_code = item.get("language_code", "en")
    command = item.get("translated_command", item.get("english_command", ""))
    slot_translations = item.get("slot_translations", [])

    if not command or not command.strip():
        return "O", []

    tokens = tokenize_by_language(command, language_code, lang_config)
    if not tokens:
        return "O", tokens

    bio_tags = ["O"] * len(tokens)
    tagged_positions = set()

    original_slots = item.get("slots", [])
    trans_lookup = {}
    for st in slot_translations:
        orig_val = (st.get("original") or "").lower().strip()
        trans_val = (st.get("translated") or "").strip()
        if orig_val and trans_val:
            trans_lookup[orig_val] = trans_val

    for slot in original_slots:
        slot_type = slot.get("slot_type", "")
        original_value = slot.get("value", "")
        if not slot_type or not original_value:
            continue

        translated_value = trans_lookup.get(original_value.lower().strip(), original_value)
        match = find_value_in_tokens(tokens, translated_value)
        if match is None and translated_value != original_value:
            match = find_value_in_tokens(tokens, original_value)
        if match is None:
            continue

        start_idx, end_idx = match
        positions = set(range(start_idx, end_idx))
        if positions & tagged_positions:
            continue

        bio_tags[start_idx] = f"B-{slot_type}"
        for j in range(start_idx + 1, end_idx):
            bio_tags[j] = f"I-{slot_type}"
        tagged_positions.update(positions)

    return " ".join(bio_tags), tokens


def parse_arguments():
    parser = argparse.ArgumentParser(description="Project slots cross-lingually and generate BIO tags for SNIPS multilingual")
    parser.add_argument("--input", "-i", type=str,
                        default="data/snips_multilingual_pipeline/snips_translated_all_languages.json",
                        help="Input translated JSON file")
    parser.add_argument("--output", "-o", type=str,
                        default="data/snips_multilingual_pipeline/snips_bio_all_languages.json",
                        help="Output file with BIO tags")
    parser.add_argument("--config", "-c", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "language_config.json"),
                        help="Language configuration file")
    return parser.parse_args()


def main():
    args = parse_arguments()
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    lang_config = load_language_config(args.config)

    print(f"Loading translated data from {args.input}...")
    try:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found!")
        return

    print("Generating cross-lingual BIO tags...")
    alignment_stats = Counter()
    slot_type_counts = Counter()

    for item in tqdm(data, desc="Projecting slots"):
        lang_code = item.get("language_code", "en")
        # English entries already have source-stage BIO tags from step 00. Keep them.
        if lang_code == "en" and item.get("bio_tags"):
            tokens = item.get("tokens") or tokenize_by_language(item.get("english_command", ""), "en", lang_config)
            item["tokens"] = tokens
            item["token_count"] = len(tokens)
        else:
            bio_tags, tokens = create_bio_tags_multislot(item, lang_config)
            item["bio_tags"] = bio_tags
            item["tokens"] = tokens
            item["token_count"] = len(tokens)

        bio_tags_str = item["bio_tags"]
        if isinstance(bio_tags_str, list):
            bio_tags_str = " ".join(bio_tags_str)
            item["bio_tags"] = bio_tags_str

        # Alignment stats
        lang = item.get("language", "unknown")
        total_slots = len(item.get("slot_translations", []))
        found_slots = sum(1 for t in bio_tags_str.split() if t.startswith("B-"))
        alignment_stats[f"{lang}_total"] += 1
        alignment_stats[f"{lang}_slots_expected"] += total_slots
        alignment_stats[f"{lang}_slots_found"] += found_slots
        if found_slots > 0:
            alignment_stats[f"{lang}_has_slots"] += 1

        for t in bio_tags_str.split():
            if t.startswith("B-"):
                slot_type_counts[t[2:]] += 1

        # Validate token count matches tag count
        tag_count = len(bio_tags_str.split())
        if tag_count != len(item["tokens"]):
            if tag_count < len(item["tokens"]):
                item["bio_tags"] = bio_tags_str + " " + " ".join(["O"] * (len(item["tokens"]) - tag_count))
            else:
                item["bio_tags"] = " ".join(bio_tags_str.split()[:len(item["tokens"])])

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nBIO-tagged data saved to {args.output}")
    print(f"Total entries: {len(data)}")
    print(f"Unique slot types used: {len(slot_type_counts)} -> {dict(slot_type_counts)}")

    languages_seen = sorted(set(item.get("language", "unknown") for item in data))
    print("\nSlot alignment statistics per language:")
    for lang in languages_seen:
        total = alignment_stats.get(f"{lang}_total", 0)
        has_slots = alignment_stats.get(f"{lang}_has_slots", 0)
        expected = alignment_stats.get(f"{lang}_slots_expected", 0)
        found = alignment_stats.get(f"{lang}_slots_found", 0)
        if total > 0:
            pct = found / expected * 100 if expected > 0 else 0
            print(f"  {lang}: {has_slots}/{total} entries with slots, "
                  f"{found}/{expected} individual slots found ({pct:.1f}%)")


if __name__ == "__main__":
    main()
