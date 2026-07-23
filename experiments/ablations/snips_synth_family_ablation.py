#!/usr/bin/env python3
"""
Per-family ablation of the v3 template augmentation.

The v3 augmentation added four *register families* of phrasing on top of the
cleaned v2 synth corpus:
  - colloquial   : slangy/idiomatic verbs (kill, boost, shutdown, crank up, ...)
  - indirect     : politeness / question openers (could you, would you mind, ...)
  - explanatory  : conditional / stateful justifications ("the room is too
                   bright, turn it down", "it's dark in here")
  - first_person : speaker-state framing ("i'm in the {room}", "i need light")

This script generates, for each family, a dataset = v2 + ONLY that family, with
an EQUAL utterance budget (so the comparison isolates register *quality*, not
quantity). It also emits a v2-baseline copy and a v3-all copy for anchors.

Each dataset is written to:
  data/snips_family_ablation/{name}/dataset.json

Then evaluate synth-only (ratio 0.0) on each in WSL:
  for d in v2_baseline colloquial indirect explanatory first_person v3_all; do
    .venv_wsl/bin/python phase3/snips_domain_tuning_snipsnlu.py \
        --synth data/snips_family_ablation/$d/dataset.json \
        --ratios 0.0 --skip-real-only \
        --out-name family_$d.json
  done
"""
import copy
import json
import os
import random

V2 = "data/snips_synth_for_snipsnlu_v2/dataset.json"
TAXONOMY = "data/snips_multilingual_pipeline/config/snips_slot_taxonomy.json"
OUT_ROOT = "data/snips_family_ablation"
SEED = 42
BUDGET_PER_FAMILY = 60  # equal utterances added per family
NO_ROOM_FRAC = 0.25

EXTRA_ROOMS = [
    "bed room", "kids room", "lounge", "toilet", "toilets", "loo",
    "entire house", "entire flat", "entire appartment", "child's bedroom",
    "children room", "parking room", "cella", "cublicle", "facilities",
]

# ---------------------------------------------------------------------------
# Templates organised by FAMILY -> intent -> {with_room, no_room}
# (Re-tagged from snips_synth_v3_template_augment.py's comment groups.)
# ---------------------------------------------------------------------------
FAMILIES = {
    "colloquial": {
        "DecreaseBrightness": {
            "with_room": ["dial down the lights in the {room}", "dial the {room} lights down",
                          "tone down the {room} lights", "cut the lights in the {room}",
                          "soften the lights in the {room}", "chill the lights in the {room}"],
            "no_room": ["dim the lights", "turn the lights down"],
        },
        "IncreaseBrightness": {
            "with_room": ["boost the brightness in the {room}", "boost the {room} lights",
                          "crank up the lights in the {room}", "crank the {room} lights up",
                          "amp up the lights in the {room}", "kick up the brightness in the {room}",
                          "jack up the {room} lights"],
            "no_room": ["boost the brightness", "turn the lights up"],
        },
        "SwitchLightOff": {
            "with_room": ["kill the lights in the {room}", "kill the {room} lights",
                          "shutdown the {room} lights", "shut off the lights in the {room}",
                          "end the lights in the {room}", "cut the lights in the {room}",
                          "power off the {room} lights", "flip off the {room} lights"],
            "no_room": ["kill the lights", "shutdown", "lights off", "kill it"],
        },
        "SwitchLightOn": {
            "with_room": ["fire up the {room} lights", "flip on the {room} lights",
                          "power on the {room} lights", "hit the {room} lights",
                          "kick on the {room} lights"],
            "no_room": ["lights on"],
        },
        "SetLightBrightness": {
            "with_room": ["crank the {room} lights to {brightness}",
                          "peg the {room} lights at {brightness}",
                          "dial the {room} lights to {brightness}"],
            "no_room": ["make the lights {brightness}"],
        },
        "SetLightColor": {
            "with_room": ["flood the {room} with {color}", "paint the {room} lights {color}",
                          "wash the {room} in {color}"],
            "no_room": ["make the lights {color}"],
        },
    },
    "indirect": {
        "DecreaseBrightness": {
            "with_room": ["could you dim the lights in the {room}",
                          "would you mind dimming the {room} lights"],
            "no_room": ["please turn the lights down", "could you dim the lights"],
        },
        "IncreaseBrightness": {
            "with_room": ["could you brighten the {room}",
                          "would you please turn the {room} lights up"],
            "no_room": ["please turn the lights up", "could you brighten the lights"],
        },
        "SwitchLightOff": {
            "with_room": ["could you turn off the {room} lights",
                          "would you mind turning off the lights in the {room}"],
            "no_room": ["please turn off the lights"],
        },
        "SwitchLightOn": {
            "with_room": ["could you turn on the {room} lights",
                          "would you please switch on the lights in the {room}"],
            "no_room": ["could you turn on the lights", "please turn on the lights"],
        },
        "SetLightBrightness": {
            "with_room": ["could you set the {room} lights to {brightness}",
                          "would you mind setting the {room} lights to {brightness}"],
            "no_room": [],
        },
        "SetLightColor": {
            "with_room": ["could you make the {room} lights {color}",
                          "would you please switch the {room} lights to {color}"],
            "no_room": [],
        },
    },
    "explanatory": {
        "DecreaseBrightness": {
            "with_room": ["make it less bright in the {room}", "make the {room} less bright",
                          "the {room} is too bright, turn it down",
                          "the light is too bright in the {room}, turn it down",
                          "it's too bright in the {room}, please dim the lights"],
            "no_room": ["make it less bright", "the light is too bright, turn it down",
                        "adjust the lights so it's less bright", "it's too bright in here"],
        },
        "IncreaseBrightness": {
            "with_room": ["the {room} is too dark, brighten it up", "make it brighter in the {room}",
                          "the {room} lights are too dim, brighten them",
                          "make the {room} brighter please"],
            "no_room": ["make it brighter", "it's dark in here"],
        },
        "SwitchLightOff": {
            "with_room": ["the {room} lights are on, turn them off",
                          "i don't need the {room} lights on", "no need for the {room} lights"],
            "no_room": [],
        },
        "SwitchLightOn": {
            "with_room": ["the {room} is dark, turn on the lights", "the {room} needs light",
                          "i can't see in the {room}, turn on the lights"],
            "no_room": ["give me some light"],
        },
        "SetLightBrightness": {
            "with_room": ["set the brightness of the {room} lights to {brightness}",
                          "adjust the {room} lights to level {brightness}"],
            "no_room": ["set the brightness to {brightness}", "dim the lights to {brightness}"],
        },
        "SetLightColor": {
            "with_room": [],
            "no_room": ["change the lights to {color}"],
        },
    },
    "first_person": {
        "DecreaseBrightness": {
            "with_room": ["i'd like less light in the {room}", "i need less light in the {room}",
                          "i want less light in the {room}", "i'm in the {room} and it's too bright"],
            "no_room": [],
        },
        "IncreaseBrightness": {
            "with_room": ["i need more light in the {room}", "i need it brighter in the {room}",
                          "i'd like more light in the {room}", "i'm in the {room} and i can't see"],
            "no_room": ["i need more light here"],
        },
        "SwitchLightOff": {
            "with_room": ["i'd like the {room} lights off", "i'm done with the {room} lights",
                          "i'm leaving the {room}, turn the lights off"],
            "no_room": ["i'd like the lights to be shutdown"],
        },
        "SwitchLightOn": {
            "with_room": ["i'm in the {room}, turn on the lights", "i need the lights on in the {room}",
                          "i'd like the {room} lights on"],
            "no_room": ["i need some light"],
        },
        "SetLightBrightness": {
            "with_room": ["i want the {room} at {brightness}", "i'd like the {room} lights at {brightness}",
                          "i want the brightness in the {room} at {brightness}"],
            "no_room": [],
        },
        "SetLightColor": {
            "with_room": ["i want the {room} lights {color}", "i'd like the {room} lights {color}"],
            "no_room": [],
        },
    },
}


def utt_from_template(template, room_pool, color_pool, brightness_pool, rng):
    text = template
    values = {}
    if "{room}" in text:
        values["room"] = rng.choice(room_pool)
    if "{color}" in text:
        values["color"] = rng.choice(color_pool)
    if "{brightness}" in text:
        values["brightness"] = rng.choice(brightness_pool)
    chunks = []
    i = 0
    while i < len(text):
        next_pos, next_slot = None, None
        for slot_name in ("room", "color", "brightness"):
            token = "{" + slot_name + "}"
            pos = text.find(token, i)
            if pos != -1 and (next_pos is None or pos < next_pos):
                next_pos, next_slot = pos, slot_name
        if next_pos is None:
            if i < len(text):
                chunks.append({"text": text[i:]})
            break
        if next_pos > i:
            chunks.append({"text": text[i:next_pos]})
        val = values.get(next_slot)
        if val is None:
            return None
        entity = {"room": "house_room_unique", "color": "color",
                  "brightness": "snips/number"}[next_slot]
        chunks.append({"text": val, "slot_name": next_slot, "entity": entity})
        i = next_pos + len("{" + next_slot + "}")
    merged = []
    for c in chunks:
        if merged and "slot_name" not in c and "slot_name" not in merged[-1]:
            merged[-1] = {"text": merged[-1]["text"] + c["text"]}
        else:
            merged.append(c)
    return {"data": merged}


def gen_family_dataset(family_name, base_v2, room_pool, color_pool, brightness_pool, budget, rng):
    ds = copy.deepcopy(base_v2)
    fam = FAMILIES[family_name]
    # Flatten (intent, pattern, template) options
    with_room_opts, no_room_opts = [], []
    for intent, groups in fam.items():
        for t in groups.get("with_room", []):
            with_room_opts.append((intent, t))
        for t in groups.get("no_room", []):
            no_room_opts.append((intent, t))
    n_no_room = int(round(budget * NO_ROOM_FRAC))
    n_with_room = budget - n_no_room
    added = 0
    seen = set()

    def add_from(opts, n):
        nonlocal added
        got, attempts = 0, 0
        while got < n and attempts < n * 40 and opts:
            attempts += 1
            intent, tmpl = rng.choice(opts)
            u = utt_from_template(tmpl, room_pool, color_pool, brightness_pool, rng)
            if u is None:
                continue
            key = (intent, "".join(c.get("text", "") for c in u["data"]))
            if key in seen:
                continue
            seen.add(key)
            ds["intents"][intent]["utterances"].append(u)
            got += 1
            added += 1
        return got

    add_from(with_room_opts, n_with_room)
    add_from(no_room_opts, n_no_room)
    return ds, added


def main():
    rng = random.Random(SEED)
    v2 = json.load(open(V2, encoding="utf-8"))
    tax = json.load(open(TAXONOMY, encoding="utf-8"))
    tax_rooms = [r for r in tax["slot_types"]["room"]["values"] if r != "room"]
    room_pool = sorted(set(tax_rooms + EXTRA_ROOMS))
    color_pool = list(tax["slot_types"]["color"]["values"])
    for canon, syns in tax["slot_types"]["color"].get("synonyms", {}).items():
        color_pool.extend(syns)
    b = tax["slot_types"]["brightness"]
    brightness_pool = list(b["word_values"]) + list(b["digit_values"])

    os.makedirs(OUT_ROOT, exist_ok=True)

    # v2 baseline (unchanged)
    d = os.path.join(OUT_ROOT, "v2_baseline")
    os.makedirs(d, exist_ok=True)
    json.dump(v2, open(os.path.join(d, "dataset.json"), "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    v2_total = sum(len(v2["intents"][i]["utterances"]) for i in v2["intents"])
    print(f"v2_baseline: {v2_total} utts")

    for fam in FAMILIES:
        ds, added = gen_family_dataset(fam, v2, room_pool, color_pool,
                                       brightness_pool, BUDGET_PER_FAMILY, rng)
        d = os.path.join(OUT_ROOT, fam)
        os.makedirs(d, exist_ok=True)
        json.dump(ds, open(os.path.join(d, "dataset.json"), "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        total = sum(len(ds["intents"][i]["utterances"]) for i in ds["intents"])
        print(f"{fam}: v2 + {added} utts = {total}")


if __name__ == "__main__":
    main()
