#!/usr/bin/env python3
"""
Build EN -> target-language slot-value dictionaries for the SNIPS smart-lights
pipeline using Google Translate (via deep-translator).

Produces config/slot_translations_<lang>.json with structure:
  {
    "room":       {"office": "bureau", "kitchen": "cuisine", ...},
    "color":      {"red": "rouge", "blue": "bleu", ...},
    "brightness": {"twenty": "vingt", "thirty two": "trente-deux", ...}
  }

Digit forms (e.g., "25") and the canonical color synonyms (e.g., "pink" as
synonym of "red") are mapped explicitly; everything else goes through Google
Translate.

Run once per target language:
    python build_slot_translations.py --target-lang fr
"""

import argparse
import json
import os
import time
from deep_translator import GoogleTranslator
from tqdm import tqdm


TARGET_NAME_BY_CODE = {
    "fr": "French",
    "es": "Spanish",
    "de": "German",
    "pt": "Portuguese",
}


def load_taxonomy(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def translate_list(values, target_code, delay=0.05):
    """Translate a list of EN values to target_code via Google Translate."""
    out = {}
    translator = GoogleTranslator(source="en", target=target_code)
    for v in tqdm(values, desc=f"EN->{target_code}"):
        try:
            t = translator.translate(v)
        except Exception as e:
            print(f"  WARN: failed to translate '{v}': {e}")
            t = v  # fall back to original
        out[v] = (t or "").lower().strip()
        time.sleep(delay)
    return out


def main():
    parser = argparse.ArgumentParser(description="Build slot-value translation dictionary for SNIPS multilingual")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "snips_slot_taxonomy.json"),
                        help="Path to slot taxonomy JSON")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config"),
                        help="Directory to write slot_translations_<lang>.json")
    parser.add_argument("--target-lang", type=str, default="fr",
                        help="Target language code (e.g. fr, es, de)")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.config)
    target_code = args.target_lang.lower()
    print(f"Building EN -> {target_code} slot dictionary")

    room_values = taxonomy["slot_types"]["room"]["values"]
    color_values = taxonomy["slot_types"]["color"]["values"]
    color_synonyms = taxonomy["slot_types"]["color"].get("synonyms", {})
    brightness_word_values = taxonomy["slot_types"]["brightness"]["word_values"]
    brightness_digit_values = taxonomy["slot_types"]["brightness"]["digit_values"]

    print(f"  Rooms: {len(room_values)}")
    print(f"  Colors: {len(color_values)} (+ synonyms)")
    print(f"  Brightness words: {len(brightness_word_values)}")
    print(f"  Brightness digits: {len(brightness_digit_values)} (kept as-is)")

    room_map = translate_list(room_values, target_code)
    color_map = translate_list(color_values, target_code)

    # Map synonyms to the canonical translation
    for canonical, syns in color_synonyms.items():
        if canonical in color_map:
            for syn in syns:
                color_map[syn] = color_map[canonical]

    brightness_word_map = translate_list(brightness_word_values, target_code)
    brightness_digit_map = {d: d for d in brightness_digit_values}  # digits unchanged
    brightness_map = {**brightness_word_map, **brightness_digit_map}

    out = {
        "room": room_map,
        "color": color_map,
        "brightness": brightness_map,
        "_metadata": {
            "target_lang": target_code,
            "target_name": TARGET_NAME_BY_CODE.get(target_code, target_code),
            "source_lang": "en",
            "engine": "deep-translator GoogleTranslator",
        },
    }

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"slot_translations_{target_code}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved dictionary to: {output_file}")
    print(f"  rooms: {len(room_map)}")
    print(f"  colors (incl synonyms): {len(color_map)}")
    print(f"  brightness: {len(brightness_map)}")


if __name__ == "__main__":
    main()
