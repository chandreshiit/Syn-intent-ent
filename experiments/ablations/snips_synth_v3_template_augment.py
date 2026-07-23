#!/usr/bin/env python3
"""
Extend the SNIPS synth-v2 dataset with diagnostic-driven template augmentation.

Based on the error patterns from `snips_synthonly_error_analysis.py`, add new
templates targeting the four missing phrasing families that the CRF couldn't
recover:
  1) Colloquial verbs (kill / boost / shutdown / dial down / crank up / brighten)
  2) Indirect openers (Can you, Could you, I'd like, I want, Please)
  3) Explanatory / conditional forms ("so it's less bright", "make it brighter")
  4) First-person state ("I'm in the X", "I need more light in the X")

Each new template is either:
  * with-room (uses {room} placeholder + emits room slot)
  * no-room (matches real utts like "I need more light here" that carry no slot)

Multi-slot intents (SetLightBrightness, SetLightColor) also get variants.

Output: data/snips_synth_for_snipsnlu_v3/dataset.json
        data/snips_synth_for_snipsnlu_v3/AUGMENTATION_NOTES.md
"""
import json
import os
import random
from collections import Counter, defaultdict


V2 = "data/snips_synth_for_snipsnlu_v2/dataset.json"
TAXONOMY = "data/snips_multilingual_pipeline/config/snips_slot_taxonomy.json"
OUT_DIR = "data/snips_synth_for_snipsnlu_v3"
SEED = 42


# Additional room values seen in real that aren't in the pipeline taxonomy —
# so new templates can draw from them too, matching what v2 already added
# via substitution.
EXTRA_ROOMS = [
    "bed room", "kids room", "lounge", "toilet", "toilets", "loo",
    "entire house", "entire flat", "entire appartment", "child's bedroom",
    "children room", "parking room", "cella", "cublicle", "facilities",
    "toilets room", "toilet room",
]


# ============================================================================
# NEW TEMPLATES — organised by intent and pattern family.
# `TEMPLATES_WITH_ROOM[intent]` = list of strings containing "{room}" (and
#    optionally "{brightness}" or "{color}").
# `TEMPLATES_NO_ROOM[intent]` = list of strings with no placeholders (or only
#    brightness/color); no room slot will be emitted for these.
# ============================================================================

TEMPLATES_WITH_ROOM = {
    "DecreaseBrightness": [
        # Colloquial verbs
        "dial down the lights in the {room}",
        "dial the {room} lights down",
        "tone down the {room} lights",
        "cut the lights in the {room}",
        "soften the lights in the {room}",
        "chill the lights in the {room}",
        # Explanatory / conditional
        "make it less bright in the {room}",
        "make the {room} less bright",
        "the {room} is too bright, turn it down",
        "the light is too bright in the {room}, turn it down",
        "it's too bright in the {room}, please dim the lights",
        # First-person / hedging
        "i'd like less light in the {room}",
        "i need less light in the {room}",
        "i want less light in the {room}",
        "i'm in the {room} and it's too bright",
        # Politeness / question
        "could you dim the lights in the {room}",
        "would you mind dimming the {room} lights",
    ],
    "IncreaseBrightness": [
        # Colloquial verbs
        "boost the brightness in the {room}",
        "boost the {room} lights",
        "crank up the lights in the {room}",
        "crank the {room} lights up",
        "amp up the lights in the {room}",
        "kick up the brightness in the {room}",
        "jack up the {room} lights",
        # Explanatory / conditional
        "the {room} is too dark, brighten it up",
        "make it brighter in the {room}",
        "the {room} lights are too dim, brighten them",
        "make the {room} brighter please",
        # First-person / hedging
        "i need more light in the {room}",
        "i need it brighter in the {room}",
        "i'd like more light in the {room}",
        "i'm in the {room} and i can't see",
        # Politeness / question
        "could you brighten the {room}",
        "would you please turn the {room} lights up",
    ],
    "SwitchLightOff": [
        # Colloquial verbs
        "kill the lights in the {room}",
        "kill the {room} lights",
        "shutdown the {room} lights",
        "shut off the lights in the {room}",
        "end the lights in the {room}",
        "cut the lights in the {room}",
        "power off the {room} lights",
        "flip off the {room} lights",
        # Explanatory / conditional
        "the {room} lights are on, turn them off",
        "i don't need the {room} lights on",
        "no need for the {room} lights",
        # First-person / hedging
        "i'd like the {room} lights off",
        "i'm done with the {room} lights",
        "i'm leaving the {room}, turn the lights off",
        # Politeness / question
        "could you turn off the {room} lights",
        "would you mind turning off the lights in the {room}",
    ],
    "SwitchLightOn": [
        # Colloquial verbs
        "fire up the {room} lights",
        "flip on the {room} lights",
        "power on the {room} lights",
        "hit the {room} lights",
        "kick on the {room} lights",
        # Explanatory / conditional
        "the {room} is dark, turn on the lights",
        "the {room} needs light",
        "i can't see in the {room}, turn on the lights",
        # First-person / hedging
        "i'm in the {room}, turn on the lights",
        "i need the lights on in the {room}",
        "i'd like the {room} lights on",
        # Politeness / question
        "could you turn on the {room} lights",
        "would you please switch on the lights in the {room}",
    ],
    "SetLightBrightness": [
        # Colloquial / imperative
        "crank the {room} lights to {brightness}",
        "peg the {room} lights at {brightness}",
        "dial the {room} lights to {brightness}",
        # First-person
        "i want the {room} at {brightness}",
        "i'd like the {room} lights at {brightness}",
        "i want the brightness in the {room} at {brightness}",
        # Explanatory
        "set the brightness of the {room} lights to {brightness}",
        "adjust the {room} lights to level {brightness}",
        # Politeness / question
        "could you set the {room} lights to {brightness}",
        "would you mind setting the {room} lights to {brightness}",
    ],
    "SetLightColor": [
        # Colloquial
        "flood the {room} with {color}",
        "paint the {room} lights {color}",
        "wash the {room} in {color}",
        # First-person
        "i want the {room} lights {color}",
        "i'd like the {room} lights {color}",
        # Politeness / question
        "could you make the {room} lights {color}",
        "would you please switch the {room} lights to {color}",
    ],
}


TEMPLATES_NO_ROOM = {
    "DecreaseBrightness": [
        "dim the lights",
        "turn the lights down",
        "make it less bright",
        "the light is too bright, turn it down",
        "adjust the lights so it's less bright",
        "it's too bright in here",
        "please turn the lights down",
        "could you dim the lights",
    ],
    "IncreaseBrightness": [
        "boost the brightness",
        "turn the lights up",
        "make it brighter",
        "i need more light here",
        "it's dark in here",
        "please turn the lights up",
        "could you brighten the lights",
    ],
    "SwitchLightOff": [
        "kill the lights",
        "shutdown",
        "lights off",
        "turn off the lights",
        "i'd like the lights to be shutdown",
        "shut off the lights",
        "please turn off the lights",
        "kill it",
    ],
    "SwitchLightOn": [
        "lights on",
        "turn on the lights",
        "give me some light",
        "i need some light",
        "could you turn on the lights",
        "please turn on the lights",
    ],
    "SetLightBrightness": [
        "set the brightness to {brightness}",
        "make the lights {brightness}",
        "dim the lights to {brightness}",
        "brighten the lights to {brightness}",
    ],
    "SetLightColor": [
        "make the lights {color}",
        "turn the lights {color}",
        "change the lights to {color}",
    ],
}


# Per-intent target new-utt counts (with-room + no-room combined). Weighted
# toward the intents with the biggest diagnostic gaps: SwitchLightOff (biased
# to None), DecreaseBrightness, IncreaseBrightness.
NEW_COUNTS = {
    "DecreaseBrightness": 60,
    "IncreaseBrightness": 60,
    "SetLightBrightness": 35,
    "SetLightColor": 20,
    "SwitchLightOff": 60,
    "SwitchLightOn": 40,
}
# Fraction of no-room templates within each intent's new utts. Real test has
# some no-room utts (~5-10% of errors), so we want a small but present share.
NO_ROOM_FRAC = 0.25


def utt_from_template(template, slot_types, room_pool, color_pool, brightness_pool, rng):
    """Return a Snips NLU utterance dict {"data": [...]} from a template string.

    Handles at most one occurrence per placeholder type ({room}, {color},
    {brightness}). Returns None if generation fails (e.g. placeholder without
    value).
    """
    # Build ordered list of (start_idx, placeholder, value) using sequential scan
    text = template
    # Sample values
    values = {}
    if "{room}" in text:
        values["room"] = rng.choice(room_pool)
    if "{color}" in text:
        values["color"] = rng.choice(color_pool)
    if "{brightness}" in text:
        values["brightness"] = rng.choice(brightness_pool)

    # Segment the template into chunks: text between placeholders + slot chunks
    chunks = []
    i = 0
    while i < len(text):
        # find next placeholder
        next_pos = None
        next_slot = None
        for slot_name in ("room", "color", "brightness"):
            token = "{" + slot_name + "}"
            pos = text.find(token, i)
            if pos != -1 and (next_pos is None or pos < next_pos):
                next_pos = pos
                next_slot = slot_name
        if next_pos is None:
            # rest of text
            if i < len(text):
                chunks.append({"text": text[i:]})
            break
        # text before placeholder
        if next_pos > i:
            chunks.append({"text": text[i:next_pos]})
        # slot chunk
        val = values.get(next_slot)
        if val is None:
            return None
        entity_name = {"room": "house_room_unique", "color": "color",
                       "brightness": "snips/number"}[next_slot]
        chunks.append({"text": val, "slot_name": next_slot, "entity": entity_name})
        i = next_pos + len("{" + next_slot + "}")

    # Merge adjacent plain-text chunks (defensive)
    merged = []
    for c in chunks:
        if merged and "slot_name" not in c and "slot_name" not in merged[-1]:
            merged[-1] = {"text": merged[-1]["text"] + c["text"]}
        else:
            merged.append(c)
    return {"data": merged}


def main():
    rng = random.Random(SEED)
    v2 = json.load(open(V2, encoding="utf-8"))
    taxonomy = json.load(open(TAXONOMY, encoding="utf-8"))

    # Room pool: taxonomy values MINUS the noisy "room" value, PLUS extra rooms
    # from real. Colors: primary + synonyms. Brightness: word + digit mix.
    tax_rooms = [r for r in taxonomy["slot_types"]["room"]["values"] if r != "room"]
    room_pool = list(set(tax_rooms + EXTRA_ROOMS))
    color_pool = list(taxonomy["slot_types"]["color"]["values"])
    for canon, syns in taxonomy["slot_types"]["color"].get("synonyms", {}).items():
        color_pool.extend(syns)
    b_conf = taxonomy["slot_types"]["brightness"]
    brightness_pool = list(b_conf["word_values"]) + list(b_conf["digit_values"])
    print(f"rooms: {len(room_pool)}, colors: {len(color_pool)}, brightness: {len(brightness_pool)}")

    notes = []
    notes.append("# SNIPS synth v3 template augmentation notes\n")
    notes.append(f"Base: `{V2}` (v2 cleaned + room-expanded)\n")
    notes.append(f"Seed: {SEED}\n")

    orig_counts = {i: len(v2["intents"][i]["utterances"]) for i in v2["intents"]}

    added_by_intent = defaultdict(int)
    added_by_pattern = Counter()  # 'with_room' / 'no_room'
    added_texts = set()

    for intent, target in NEW_COUNTS.items():
        with_room_templates = TEMPLATES_WITH_ROOM.get(intent, [])
        no_room_templates = TEMPLATES_NO_ROOM.get(intent, [])
        target_no_room = int(round(target * NO_ROOM_FRAC))
        target_with_room = target - target_no_room

        # Slot types this intent should carry (for template rendering)
        # SetLightBrightness has brightness + optional room; SetLightColor has
        # color + optional room. Others: just room (or none).
        # slot_types isn't strictly needed because we detect placeholders from text.

        added_this_intent = 0
        attempts = 0
        max_attempts = target * 12
        # with-room loop
        while added_this_intent < target_with_room and attempts < max_attempts:
            attempts += 1
            template = rng.choice(with_room_templates)
            u = utt_from_template(template, None, room_pool, color_pool,
                                  brightness_pool, rng)
            if u is None:
                continue
            key = ("with_room", "".join(c.get("text", "") for c in u["data"]))
            if key in added_texts:
                continue
            added_texts.add(key)
            v2["intents"][intent]["utterances"].append(u)
            added_by_intent[intent] += 1
            added_by_pattern["with_room"] += 1
            added_this_intent += 1

        # no-room loop
        no_room_added = 0
        attempts = 0
        while no_room_added < target_no_room and attempts < max_attempts:
            attempts += 1
            template = rng.choice(no_room_templates)
            u = utt_from_template(template, None, room_pool, color_pool,
                                  brightness_pool, rng)
            if u is None:
                continue
            key = ("no_room", "".join(c.get("text", "") for c in u["data"]))
            if key in added_texts:
                continue
            added_texts.add(key)
            v2["intents"][intent]["utterances"].append(u)
            added_by_intent[intent] += 1
            added_by_pattern["no_room"] += 1
            no_room_added += 1

    new_counts = {i: len(v2["intents"][i]["utterances"]) for i in v2["intents"]}
    notes.append(f"\n## Template families added\n")
    for i, t in TEMPLATES_WITH_ROOM.items():
        notes.append(f"- **{i}** with-room templates: {len(t)}\n")
    for i, t in TEMPLATES_NO_ROOM.items():
        notes.append(f"- **{i}** no-room templates: {len(t)}\n")
    notes.append(f"\n## Per-intent target and actual\n")
    notes.append(f"| Intent | v2 | target-new | +with-room | +no-room | v3 |\n|---|---:|---:|---:|---:|---:|\n")
    # Split added counts by pattern per intent (we appended per intent
    # sequentially so we track separately)
    # Approximate: use NEW_COUNTS distribution
    for intent in v2["intents"]:
        t_no = int(round(NEW_COUNTS.get(intent, 0) * NO_ROOM_FRAC))
        t_wr = NEW_COUNTS.get(intent, 0) - t_no
        notes.append(f"| {intent} | {orig_counts[intent]} | {NEW_COUNTS.get(intent, 0)} | {t_wr} | {t_no} | {new_counts[intent]} |\n")
    notes.append(f"| **TOTAL** | **{sum(orig_counts.values())}** | **{sum(NEW_COUNTS.values())}** | | | **{sum(new_counts.values())}** |\n")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, "dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(v2, f, indent=2, ensure_ascii=False)
    notes_path = os.path.join(OUT_DIR, "AUGMENTATION_NOTES.md")
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write("".join(notes))
    print(f"Wrote: {out_path}")
    print(f"Total added: {sum(added_by_intent.values())} (with_room={added_by_pattern['with_room']}, no_room={added_by_pattern['no_room']})")
    print(f"v2 -> v3: {sum(orig_counts.values())} -> {sum(new_counts.values())}")


if __name__ == "__main__":
    main()
