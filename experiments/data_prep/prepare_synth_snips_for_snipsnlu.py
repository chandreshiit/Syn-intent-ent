#!/usr/bin/env python3
"""
Convert our synthetic SNIPS BIO data into a Snips NLU dataset.json so it
can be trained + evaluated with the original Snips NLU library.

Input:
  data/snips_multilingual_pipeline/processed_data/en/all/
    seq.in     (tokenized utterance per line)
    seq.out    (BIO tags per line)
    label      (lowercase intent per line)

Output:
  data/snips_synth_for_snipsnlu/dataset.json   (Snips NLU schema)
  data/snips_synth_for_snipsnlu/bio/           (copy of BIO so the
                                                 BIO-based 5-fold CV can
                                                 work from the same dir)
  data/snips_synth_for_snipsnlu/index_map.json  (row_index -> indices in
                                                 the original BIO; used to
                                                 keep CV folds aligned)

Notes:
* Entity name <-> slot name mapping mirrors the real Snips smart-lights
  dataset:
    slot 'room'       -> entity 'house_room_unique'
    slot 'color'      -> entity 'color'
    slot 'brightness' -> entity 'snips/number'  (builtin)
* Intent labels in the synthetic BIO are lowercase ('switchlighton').
  We map them back to the original CamelCase ('SwitchLightOn').
* If --downsample is set, we stratified-sample N rows (deterministic).
"""

import argparse
import json
import os
import random
from collections import defaultdict

INTENT_LC2CAMEL = {
    "decreasebrightness": "DecreaseBrightness",
    "increasebrightness": "IncreaseBrightness",
    "setlightbrightness": "SetLightBrightness",
    "setlightcolor": "SetLightColor",
    "switchlightoff": "SwitchLightOff",
    "switchlighton": "SwitchLightOn",
}

SLOT_TO_ENTITY = {
    "room": "house_room_unique",
    "color": "color",
    "brightness": "snips/number",
}


def load_bio(base):
    with open(os.path.join(base, "seq.in"), "r", encoding="utf-8") as f:
        utts = [line.strip().split() for line in f if line.strip()]
    with open(os.path.join(base, "seq.out"), "r", encoding="utf-8") as f:
        bios = [line.strip().split() for line in f if line.strip()]
    with open(os.path.join(base, "label"), "r", encoding="utf-8") as f:
        intents_lc = [line.strip() for line in f if line.strip()]
    assert len(utts) == len(bios) == len(intents_lc), \
        f"length mismatch: {len(utts)} {len(bios)} {len(intents_lc)}"
    # Align lengths defensively
    for i, (u, s) in enumerate(zip(utts, bios)):
        if len(u) != len(s):
            if len(s) < len(u):
                bios[i] = s + ["O"] * (len(u) - len(s))
            else:
                bios[i] = s[:len(u)]
    return utts, bios, intents_lc


def bio_to_chunks(tokens, tags):
    """Group BIO tagged tokens into Snips NLU 'data' chunks.

    Snips NLU concatenates chunk['text'] in order to reconstruct the utterance,
    so chunks must include the spaces between tokens. Convention (matching the
    official SNIPS dataset.json files): slot chunks contain only the slot value
    (no leading/trailing space); text chunks own ALL the whitespace around them
    (both leading after a slot, and trailing before the next chunk).
    """
    chunks = []
    cur_text = []  # tokens currently being collected
    cur_slot = None
    pending_leading_space = False  # next text chunk needs a leading space (came after a slot)

    def flush_text():
        nonlocal cur_text, pending_leading_space
        if not cur_text:
            return
        prefix = " " if pending_leading_space else ""
        # Trailing space so the next chunk (slot or text) sits cleanly after.
        chunks.append({"text": prefix + " ".join(cur_text) + " "})
        cur_text = []
        pending_leading_space = False

    def flush_slot():
        nonlocal cur_text, cur_slot, pending_leading_space
        if cur_slot is None:
            return
        chunks.append({
            "entity": SLOT_TO_ENTITY[cur_slot],
            "slot_name": cur_slot,
            "text": " ".join(cur_text),
        })
        cur_text = []
        cur_slot = None
        pending_leading_space = True

    for tok, tag in zip(tokens, tags):
        if tag == "O":
            if cur_slot is not None:
                flush_slot()
            cur_text.append(tok)
        elif tag.startswith("B-"):
            if cur_slot is not None:
                flush_slot()
            else:
                flush_text()
            cur_slot = tag[2:]
            cur_text.append(tok)
        elif tag.startswith("I-"):
            if cur_slot is None:
                cur_slot = tag[2:]
            cur_text.append(tok)

    # Final flush
    if cur_slot is not None:
        flush_slot()
    else:
        # Trailing text — emit without forced trailing space.
        if cur_text:
            prefix = " " if pending_leading_space else ""
            chunks.append({"text": prefix + " ".join(cur_text)})

    return chunks


def stratified_downsample(intents_lc, n_target, seed):
    """Return a sorted list of indices, stratified by intent label."""
    by_intent = defaultdict(list)
    for i, lab in enumerate(intents_lc):
        by_intent[lab].append(i)
    rng = random.Random(seed)
    n_total = len(intents_lc)
    sampled = []
    for lab, idxs in by_intent.items():
        n_lab = round(len(idxs) / n_total * n_target)
        rng.shuffle(idxs)
        sampled.extend(sorted(idxs[:n_lab]))
    return sorted(sampled)


def collect_entity_values(utts, bios):
    """Collect the unique slot fillers for each custom entity from training data."""
    values = defaultdict(set)
    for toks, tags in zip(utts, bios):
        cur, cur_slot = [], None
        for t, g in zip(toks, tags):
            if g.startswith("B-"):
                if cur_slot and cur:
                    if cur_slot != "brightness":
                        values[cur_slot].add(" ".join(cur))
                cur, cur_slot = [t], g[2:]
            elif g.startswith("I-") and cur_slot:
                cur.append(t)
            else:
                if cur_slot and cur:
                    if cur_slot != "brightness":
                        values[cur_slot].add(" ".join(cur))
                cur, cur_slot = [], None
        if cur_slot and cur:
            if cur_slot != "brightness":
                values[cur_slot].add(" ".join(cur))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir",
                    default="data/snips_multilingual_pipeline/processed_data/en/all")
    ap.add_argument("--out-dir", default="data/snips_synth_for_snipsnlu")
    ap.add_argument("--downsample", type=int, default=0,
                    help="If >0, stratified-sample to this many utterances.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    utts, bios, intents_lc = load_bio(args.in_dir)
    print(f"loaded {len(utts)} utterances")

    if args.downsample and args.downsample < len(utts):
        keep = stratified_downsample(intents_lc, args.downsample, args.seed)
        utts = [utts[i] for i in keep]
        bios = [bios[i] for i in keep]
        intents_lc = [intents_lc[i] for i in keep]
        kept_idx = keep
        print(f"downsampled to {len(utts)} (stratified, seed={args.seed})")
    else:
        kept_idx = list(range(len(utts)))

    # Group utterances by intent
    by_intent = defaultdict(list)
    for u, b, lab in zip(utts, bios, intents_lc):
        by_intent[lab].append((u, b))

    # Build dataset.json
    dataset = {"language": "en", "entities": {}, "intents": {}}
    # Custom entities + builtin
    values = collect_entity_values(utts, bios)
    for slot, ent in SLOT_TO_ENTITY.items():
        if ent.startswith("snips/"):
            dataset["entities"][ent] = {"entity_type": "builtin"}
        else:
            vals = sorted(values.get(slot, []))
            dataset["entities"][ent] = {
                "data": [{"value": v, "synonyms": []} for v in vals],
                "use_synonyms": True,
                "automatically_extensible": False,
                "matching_strictness": 1.0,
            }
    # Intents
    for lab_lc in sorted(by_intent):
        camel = INTENT_LC2CAMEL[lab_lc]
        dataset["intents"][camel] = {"utterances": []}
        for toks, tags in by_intent[lab_lc]:
            chunks = bio_to_chunks(toks, tags)
            dataset["intents"][camel]["utterances"].append({"data": chunks})

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "dataset.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    # Write a parallel BIO so the BIO 5-fold script can use the SAME rows
    bio_dir = os.path.join(args.out_dir, "bio")
    os.makedirs(bio_dir, exist_ok=True)
    with open(os.path.join(bio_dir, "seq.in"), "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join(u) for u in utts) + "\n")
    with open(os.path.join(bio_dir, "seq.out"), "w", encoding="utf-8") as f:
        f.write("\n".join(" ".join(b) for b in bios) + "\n")
    with open(os.path.join(bio_dir, "label"), "w", encoding="utf-8") as f:
        f.write("\n".join(intents_lc) + "\n")

    with open(os.path.join(args.out_dir, "index_map.json"), "w", encoding="utf-8") as f:
        json.dump({"kept_idx_from_original": kept_idx}, f, indent=2)

    summary = {
        "n_utterances": len(utts),
        "n_per_intent": {INTENT_LC2CAMEL[k]: len(v) for k, v in by_intent.items()},
        "downsampled_from": len(load_bio(args.in_dir)[0]) if args.downsample else None,
        "seed": args.seed,
        "dataset_json": out_path,
    }
    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
