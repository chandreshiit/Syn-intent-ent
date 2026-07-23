#!/usr/bin/env python3
"""
Generate SNIPS smart-lights English source commands with full BIO annotations.
Uses template-based generation with value pools so:
  - all 3 slot types (room, color, brightness) are exercised
  - multi-slot intents (SetLightBrightness, SetLightColor) carry both slots
  - intent distribution matches the original SNIPS smart-lights subset exactly
    (DecreaseBrightness=296, IncreaseBrightness=296, SetLightBrightness=296,
     SetLightColor=300, SwitchLightOff=299, SwitchLightOn=278; total=1765)

This mirrors data/multiatis_multilingual_pipeline/00_generate_source_commands.py
in structure but specialised to SNIPS smart-lights ontology.

Output: snips_commands_v1.json
"""

import json
import os
import re
import random
import argparse
from collections import Counter


# Templates per intent. Placeholders {room}, {color}, {brightness} are substituted
# from value pools loaded from config/snips_slot_taxonomy.json. SNIPS Voice paper
# style: mix of imperative ("Turn down the lights in the kitchen") and question
# forms ("Can you turn off the bedroom lights?"); some short, some longer.
TEMPLATES = {
    "DecreaseBrightness": [
        "dim the lights in the {room}",
        "turn down the lights in the {room}",
        "lower the brightness in the {room}",
        "make the {room} lights dimmer",
        "decrease the brightness in the {room}",
        "can you dim the {room} lights",
        "i want to lower the lights in the {room}",
        "reduce the brightness in the {room}",
        "turn the lights down in the {room}",
        "dim the lights in my {room} please",
    ],
    "IncreaseBrightness": [
        "brighten the lights in the {room}",
        "turn up the lights in the {room}",
        "increase the brightness in the {room}",
        "make the {room} brighter",
        "raise the brightness in the {room}",
        "can you brighten the {room}",
        "turn the lights up in the {room}",
        "i want more light in the {room}",
        "brighten the {room} please",
        "make the {room} lights brighter",
    ],
    "SetLightBrightness": [
        "set the brightness to {brightness} in the {room}",
        "set the {room} lights to {brightness}",
        "adjust the {room} brightness to {brightness}",
        "change the {room} lights to {brightness}",
        "make the {room} brightness {brightness}",
        "turn the {room} lights to {brightness}",
        "i want the brightness in the {room} at {brightness}",
        "can you set the brightness in the {room} to {brightness}",
        "adjust the lights in the {room} to {brightness}",
        "set lights to level {brightness} in the {room}",
    ],
    "SetLightColor": [
        "set the {room} lights to {color}",
        "change the {room} lights to {color}",
        "make the {room} lights {color}",
        "turn the {room} lights {color}",
        "set the color of the {room} lights to {color}",
        "i want the {room} lights to be {color}",
        "can you change the {room} lights to {color}",
        "switch the {room} lights to {color}",
        "make the lights in the {room} {color}",
        "turn the lights {color} in the {room}",
    ],
    "SwitchLightOff": [
        "turn off the lights in the {room}",
        "switch off the {room} lights",
        "turn the {room} lights off",
        "lights off in the {room}",
        "can you turn off the {room} lights",
        "please turn off the lights in the {room}",
        "shut off the lights in the {room}",
        "i want the {room} lights off",
        "make sure no lights are on in the {room}",
        "turn off the {room} lights please",
    ],
    "SwitchLightOn": [
        "turn on the lights in the {room}",
        "switch on the {room} lights",
        "turn the {room} lights on",
        "lights on in the {room}",
        "can you turn on the {room} lights",
        "please turn on the lights in the {room}",
        "i want the {room} lights on",
        "light up the {room}",
        "make the lights come on in the {room}",
        "switch on the lights in my {room}",
    ],
}

# Slot placeholder -> slot_type mapping (for BIO annotation)
PLACEHOLDER_TO_SLOT_TYPE = {
    "room": "room",
    "color": "color",
    "brightness": "brightness",
}


def load_taxonomy(config_path):
    """Load slot taxonomy configuration."""
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def expand_color_values(taxonomy):
    """Return color values plus their synonyms (red -> pink synonym, etc.)."""
    color_def = taxonomy["slot_types"]["color"]
    values = list(color_def["values"])
    for canonical, syns in color_def.get("synonyms", {}).items():
        values.extend(syns)
    return values


def sample_brightness_value(taxonomy, rng):
    """Return a brightness value sampled from words + digits per the configured ratio."""
    bdef = taxonomy["slot_types"]["brightness"]
    if rng.random() < float(bdef.get("word_digit_mix_ratio", 0.5)):
        return rng.choice(bdef["word_values"])
    return rng.choice(bdef["digit_values"])


def whitespace_tokens(text):
    """Tokenize on whitespace + simple punctuation rules used downstream."""
    return re.findall(r"\w+(?:'\w+)?", text.lower())


def find_value_span(tokens, value):
    """Return (start_idx, end_idx) of `value` inside `tokens`, or None."""
    value_tokens = whitespace_tokens(value)
    if not value_tokens:
        return None
    lower_tokens = [t.lower() for t in tokens]
    for i in range(len(lower_tokens) - len(value_tokens) + 1):
        if lower_tokens[i:i + len(value_tokens)] == value_tokens:
            return (i, i + len(value_tokens))
    return None


def render_template(template, slot_types, taxonomy, rng):
    """Substitute placeholders in `template` with sampled values.

    Returns (rendered_text, slot_records) where slot_records is a list of
    {slot_type, value} dicts in template order.
    """
    rendered = template
    slot_records = []
    room_pool = taxonomy["slot_types"]["room"]["values"]
    color_pool = expand_color_values(taxonomy)

    for slot_type in slot_types:
        if slot_type == "room":
            value = rng.choice(room_pool)
        elif slot_type == "color":
            value = rng.choice(color_pool)
        elif slot_type == "brightness":
            value = sample_brightness_value(taxonomy, rng)
        else:
            raise ValueError(f"Unknown slot type: {slot_type}")
        rendered = rendered.replace("{" + slot_type + "}", value, 1)
        slot_records.append({"slot_type": slot_type, "value": value})

    return rendered, slot_records


def build_bio_tags(text, slot_records):
    """Tokenize text and produce BIO tags + per-slot spans."""
    tokens = whitespace_tokens(text)
    bio_tags = ["O"] * len(tokens)
    tagged = set()
    slots_with_spans = []
    for slot in slot_records:
        match = find_value_span(tokens, slot["value"])
        if match is None:
            slots_with_spans.append({**slot, "start": -1, "end": -1})
            continue
        start, end = match
        positions = set(range(start, end))
        if positions & tagged:
            slots_with_spans.append({**slot, "start": -1, "end": -1})
            continue
        bio_tags[start] = f"B-{slot['slot_type']}"
        for j in range(start + 1, end):
            bio_tags[j] = f"I-{slot['slot_type']}"
        tagged.update(positions)
        slots_with_spans.append({**slot, "start": start, "end": end})
    return tokens, bio_tags, slots_with_spans


def generate_for_intent(intent_name, count, taxonomy, rng):
    """Generate `count` unique-ish utterances for `intent_name`."""
    slot_types = taxonomy["intent_slots"][intent_name]
    templates = TEMPLATES[intent_name]
    category = taxonomy["intent_categories"][intent_name]

    results = []
    seen = set()
    max_attempts = count * 6
    attempts = 0
    while len(results) < count and attempts < max_attempts:
        attempts += 1
        template = rng.choice(templates)
        text, slot_records = render_template(template, slot_types, taxonomy, rng)
        if text in seen:
            continue
        seen.add(text)
        tokens, bio_tags, slots_with_spans = build_bio_tags(text, slot_records)
        results.append({
            "command": text,
            "intent": intent_name,
            "category": category,
            "tokens": tokens,
            "bio_tags": " ".join(bio_tags),
            "slots": slots_with_spans,
            "token_count": len(tokens),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate SNIPS smart-lights source commands with BIO tags")
    parser.add_argument("--config", type=str,
                        default=os.path.join(os.path.dirname(__file__), "config", "snips_slot_taxonomy.json"),
                        help="Path to slot taxonomy JSON")
    parser.add_argument("--output", type=str, default="snips_commands_v1.json",
                        help="Output JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--counts-scale", type=float, default=1.0,
                        help="Multiplier for per-intent counts (default 1.0 = original SNIPS sizes)")
    args = parser.parse_args()

    taxonomy = load_taxonomy(args.config)
    rng = random.Random(args.seed)

    all_records = []
    print("Generating source commands per intent...")
    for intent_name, base_count in taxonomy["intent_counts"].items():
        target = max(1, int(round(base_count * args.counts_scale)))
        print(f"  {intent_name}: target {target} utterances")
        records = generate_for_intent(intent_name, target, taxonomy, rng)
        if len(records) < target:
            print(f"    WARNING: only generated {len(records)} (insufficient template/value variety)")
        all_records.extend(records)

    print(f"\nWriting {len(all_records)} entries to {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    print("\nIntent distribution:")
    c = Counter(r["intent"] for r in all_records)
    for intent, count in sorted(c.items()):
        print(f"  {intent}: {count}")

    print("\nSlot type usage (B- tag counts):")
    slot_counts = Counter()
    for r in all_records:
        for tag in r["bio_tags"].split():
            if tag.startswith("B-"):
                slot_counts[tag[2:]] += 1
    for slot_type, n in sorted(slot_counts.items()):
        print(f"  {slot_type}: {n}")

    n_with_2 = sum(
        1 for r in all_records
        if sum(1 for t in r["bio_tags"].split() if t.startswith("B-")) >= 2
    )
    print(f"\nUtterances with >=2 slots (multi-slot): {n_with_2}")


if __name__ == "__main__":
    main()
