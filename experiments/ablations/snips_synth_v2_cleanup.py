#!/usr/bin/env python3
"""
Produce a cleaned + expanded synth SNIPS dataset for Snips NLU based on the
diagnostic findings from snips_synthonly_error_analysis.py:

1) FIX: Remove the 48 utterances where room slot value is literally "room"
   — these are mislabels ("in the room" tagged as room=room). Convert them
   to plain text chunks so the utterance stays as an intent example but
   without a bogus room slot.

2) EXPAND: For each room value appearing in real train utterances but never
   in synth utterances (21 values, e.g. "bed room", "kids room", "lounge",
   "toilet", "entire house"), substitute it into K copies of existing synth
   utterances. K is proportional to the room's real-frequency, so
   commonly-said real rooms get more synth coverage.

Inputs (read-only):
   data/snips_synth_for_snipsnlu/dataset.json
   data/snips_real_close/dataset.json

Output (created fresh):
   data/snips_synth_for_snipsnlu_v2/dataset.json
   data/snips_synth_for_snipsnlu_v2/CLEANUP_NOTES.md
"""
import copy
import json
import os
import random
from collections import Counter, defaultdict

REAL = "data/snips_real_close/dataset.json"
SYNTH = "data/snips_synth_for_snipsnlu/dataset.json"
OUT_DIR = "data/snips_synth_for_snipsnlu_v2"
SEED = 42
UTTS_PER_ROOM_MAX = 20  # cap synth expansion per rare room
UTTS_PER_ROOM_MIN = 3


def utt_room_slot(u):
    for c in u.get("data", []):
        if c.get("slot_name") == "room":
            return c.get("text", "").strip().lower()
    return None


def utt_room_index(u):
    """Return index of the room slot chunk in u['data'], or None."""
    for i, c in enumerate(u.get("data", [])):
        if c.get("slot_name") == "room":
            return i
    return None


def unmark_room_slot(u):
    """Merge the room-slot chunk back into surrounding plain text so the
    utterance text is unchanged but no room slot is emitted."""
    new_data = []
    for c in u.get("data", []):
        if c.get("slot_name") == "room":
            # convert to plain text (keep the text)
            new_data.append({"text": c.get("text", "")})
        else:
            new_data.append(c)
    # Merge adjacent plain-text chunks
    merged = []
    for c in new_data:
        if merged and "slot_name" not in c and "slot_name" not in merged[-1]:
            merged[-1] = {"text": merged[-1]["text"] + c["text"]}
        else:
            merged.append(c)
    return {"data": merged}


def substitute_room(u, new_room):
    """Return a copy of u with the room slot chunk's text replaced by new_room."""
    u2 = copy.deepcopy(u)
    for c in u2["data"]:
        if c.get("slot_name") == "room":
            c["text"] = new_room
    return u2


def main():
    rng = random.Random(SEED)
    synth = json.load(open(SYNTH, encoding="utf-8"))
    real = json.load(open(REAL, encoding="utf-8"))

    intents = list(synth["intents"].keys())
    notes = []
    notes.append("# SNIPS synth-v2 cleanup notes\n")
    notes.append("Source: `data/snips_synth_for_snipsnlu/dataset.json`\n")

    # Count original per-intent
    orig_counts = {i: len(synth["intents"][i]["utterances"]) for i in intents}

    # === Step 1: fix room="room" mislabels ===
    fixed = 0
    for intent in intents:
        for u in synth["intents"][intent]["utterances"]:
            if utt_room_slot(u) == "room":
                new_u = unmark_room_slot(u)
                u["data"] = new_u["data"]
                fixed += 1
    notes.append(f"\n## Step 1: unmark bogus room='room' slots\n")
    notes.append(f"Fixed {fixed} synth utterances (room slot demoted to plain text).\n")

    # === Step 2: expand rare-room coverage ===
    # Real room values used in real utterances (all)
    real_rooms_used = Counter()
    for intent in real["intents"]:
        for u in real["intents"][intent]["utterances"]:
            for c in u["data"]:
                if c.get("slot_name") == "room":
                    v = c.get("text", "").strip().lower()
                    if v:
                        real_rooms_used[v] += 1

    synth_rooms_used = Counter()
    for intent in intents:
        for u in synth["intents"][intent]["utterances"]:
            for c in u["data"]:
                if c.get("slot_name") == "room":
                    v = c.get("text", "").strip().lower()
                    if v:
                        synth_rooms_used[v] += 1

    missing_rooms = {r: n for r, n in real_rooms_used.items()
                     if r not in synth_rooms_used}

    # Filter out obvious real-side artifacts we don't want to teach:
    # - room ending with a period (e.g. "room.") — noise in real annotation
    # - the value "room" — same "in the room" issue
    exclude = {"room", "room.", "flat room", "house room", "cella room"}
    missing_rooms = {r: n for r, n in missing_rooms.items() if r not in exclude}

    notes.append(f"\n## Step 2: expand for missing real-room vocabulary\n")
    notes.append(f"Real rooms not seen in synth (after excluding noise): {len(missing_rooms)}\n")
    for r, n in sorted(missing_rooms.items(), key=lambda x: -x[1]):
        notes.append(f"- `{r}`: real n={n}\n")

    # For substitution we need synth utts with a room slot present.
    # Group them by intent so we can pick roughly balanced substitutions.
    synth_room_utts_by_intent = defaultdict(list)
    for intent in intents:
        for i, u in enumerate(synth["intents"][intent]["utterances"]):
            if utt_room_index(u) is not None:
                synth_room_utts_by_intent[intent].append(i)

    # Compute per-room synth-add counts (K)
    max_real = max(missing_rooms.values()) if missing_rooms else 1
    added_per_intent = defaultdict(int)
    added_utts = []
    for room, real_n in sorted(missing_rooms.items(), key=lambda x: -x[1]):
        # Scale K proportionally, clamp to [MIN, MAX]
        k = max(UTTS_PER_ROOM_MIN,
                min(UTTS_PER_ROOM_MAX, int(round(real_n / max_real * UTTS_PER_ROOM_MAX))))
        # Distribute across intents in proportion to synth intent size
        for _ in range(k):
            intent = rng.choice(list(synth_room_utts_by_intent.keys()))
            src_idx = rng.choice(synth_room_utts_by_intent[intent])
            src_u = synth["intents"][intent]["utterances"][src_idx]
            new_u = substitute_room(src_u, room)
            synth["intents"][intent]["utterances"].append(new_u)
            added_utts.append((intent, room))
            added_per_intent[intent] += 1

    notes.append(f"\n## Utterances added (by intent):\n")
    for intent in intents:
        notes.append(f"- {intent}: +{added_per_intent[intent]}\n")

    # === Save ===
    os.makedirs(OUT_DIR, exist_ok=True)
    new_counts = {i: len(synth["intents"][i]["utterances"]) for i in intents}
    total_orig = sum(orig_counts.values())
    total_new = sum(new_counts.values())
    notes.append(f"\n## Final counts\n")
    notes.append(f"| Intent | before | after |\n|---|---:|---:|\n")
    for i in intents:
        notes.append(f"| {i} | {orig_counts[i]} | {new_counts[i]} |\n")
    notes.append(f"| **TOTAL** | **{total_orig}** | **{total_new}** |\n")

    out_json = os.path.join(OUT_DIR, "dataset.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(synth, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT_DIR, "CLEANUP_NOTES.md"), "w", encoding="utf-8") as f:
        f.write("".join(notes))
    print(f"Wrote: {out_json}")
    print(f"Added utts total: {len(added_utts)}")
    print(f"Before: {total_orig}  After: {total_new}")


if __name__ == "__main__":
    main()
