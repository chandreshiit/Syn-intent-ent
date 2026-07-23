"""Thorough analysis of synthetic MultiATIS++ data vs original paper statistics."""
import json
from collections import Counter, defaultdict

INPUT = r"data\multiatis_multilingual_pipeline\multiatis_bio_all_languages.json"
PROCESSED = r"data\multiatis_multilingual_pipeline\processed_data"

# Original MultiATIS++ Table 1 from the paper
ORIGINAL = {
    "en": {"train": 4488, "dev": 490, "test": 893, "train_tok": 50755, "dev_tok": 5445, "test_tok": 9164, "intents": 18, "slots": 84},
    "es": {"train": 4488, "dev": 490, "test": 893, "train_tok": 55197, "dev_tok": 5927, "test_tok": 10338, "intents": 18, "slots": 84},
    "pt": {"train": 4488, "dev": 490, "test": 893, "train_tok": 55052, "dev_tok": 5909, "test_tok": 10228, "intents": 18, "slots": 84},
    "de": {"train": 4488, "dev": 490, "test": 893, "train_tok": 51111, "dev_tok": 5517, "test_tok": 9383, "intents": 18, "slots": 84},
    "fr": {"train": 4488, "dev": 490, "test": 893, "train_tok": 55909, "dev_tok": 5769, "test_tok": 10511, "intents": 18, "slots": 84},
    "zh": {"train": 4488, "dev": 490, "test": 893, "train_tok": 88194, "dev_tok": 9652, "test_tok": 16710, "intents": 18, "slots": 84},
    "ja": {"train": 4488, "dev": 490, "test": 893, "train_tok": 133890, "dev_tok": 14416, "test_tok": 25939, "intents": 18, "slots": 84},
    "hi": {"train": 1440, "dev": 160, "test": 893, "train_tok": 16422, "dev_tok": 1753, "test_tok": 9755, "intents": 17, "slots": 75},
    "tr": {"train": 578,  "dev": 60,  "test": 715, "train_tok": 6132,  "dev_tok": 686,  "test_tok": 7683, "intents": 17, "slots": 71},
}

# Known ATIS slot types (84 in original)
ORIGINAL_SLOT_TYPES = [
    "aircraft_code", "airline_code", "airline_name", "airport_code", "airport_name",
    "arrive_date.date_relative", "arrive_date.day_name", "arrive_date.day_number",
    "arrive_date.month_name", "arrive_date.today_relative", "arrive_date.year",
    "arrive_time.end_time", "arrive_time.period_mod", "arrive_time.period_of_day",
    "arrive_time.start_time", "arrive_time.time", "arrive_time.time_relative",
    "city_name", "class_type", "compartment", "connect", "cost_relative",
    "cuisine", "day_name", "day_number", "days_code",
    "depart_date.date_relative", "depart_date.day_name", "depart_date.day_number",
    "depart_date.month_name", "depart_date.today_relative", "depart_date.year",
    "depart_time.end_time", "depart_time.period_mod", "depart_time.period_of_day",
    "depart_time.start_time", "depart_time.time", "depart_time.time_relative",
    "economy", "fare_amount", "fare_basis_code", "flight_days", "flight_mod",
    "flight_number", "flight_stop", "flight_time", "fromloc.airport_code",
    "fromloc.airport_name", "fromloc.city_name", "fromloc.state_code",
    "fromloc.state_name", "meal", "meal_code", "meal_description", "mod",
    "month_name", "or", "period_of_day", "restriction_code", "return_date.date_relative",
    "return_date.day_name", "return_date.day_number", "return_date.month_name",
    "return_date.today_relative", "round_trip", "state_code", "state_name",
    "stoploc.airport_code", "stoploc.airport_name", "stoploc.city_name",
    "stoploc.state_code", "time", "time_relative", "today_relative",
    "toloc.airport_code", "toloc.airport_name", "toloc.city_name",
    "toloc.country_name", "toloc.state_code", "toloc.state_name",
    "transport_type"
]

ORIGINAL_INTENT_TYPES = [
    "abbreviation", "aircraft", "airfare", "airline", "airport",
    "capacity", "cheapest", "city", "day_name", "distance",
    "flight", "flight_no", "flight_time", "ground_fare",
    "ground_service", "meal", "quantity", "restriction"
]

print("=" * 80)
print("MULTIATIS++ SYNTHETIC DATA ANALYSIS vs ORIGINAL PAPER")
print("=" * 80)

# Load data
with open(INPUT, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"\nTotal entries: {len(data)}")

# ─── 1. LANGUAGE COUNTS ──────────────────────────────────────────
print("\n" + "=" * 80)
print("1. TOTAL ENTRIES PER LANGUAGE")
print("=" * 80)

lang_counts = Counter(d.get('language_code', '') for d in data)
print(f"{'Lang':<6} {'Ours':<8} {'Original Total':<16} {'Diff':<8}")
print("-" * 40)
for lc in sorted(ORIGINAL.keys()):
    ours = lang_counts.get(lc, 0)
    orig = sum(ORIGINAL[lc][s] for s in ['train', 'dev', 'test'])
    diff = ours - orig
    flag = " !!!" if abs(diff) > 10 else ""
    print(f"{lc:<6} {ours:<8} {orig:<16} {diff:+d}{flag}")

# ─── 2. SPLIT SIZES ──────────────────────────────────────────────
print("\n" + "=" * 80)
print("2. TRAIN/DEV/TEST SPLIT COMPARISON")
print("=" * 80)

# Load our splits
import os
our_splits = {}
for lc in sorted(ORIGINAL.keys()):
    our_splits[lc] = {}
    for split in ['train', 'dev', 'test']:
        seq_in = os.path.join(PROCESSED, lc, split, 'seq.in')
        if os.path.exists(seq_in):
            with open(seq_in, 'r', encoding='utf-8') as f:
                our_splits[lc][split] = sum(1 for _ in f)
        else:
            our_splits[lc][split] = 0

print(f"{'Lang':<6} {'Split':<7} {'Ours':<8} {'Original':<10} {'Diff':<8} {'Status'}")
print("-" * 55)
for lc in sorted(ORIGINAL.keys()):
    for split in ['train', 'dev', 'test']:
        ours = our_splits[lc].get(split, 0)
        orig = ORIGINAL[lc][split]
        diff = ours - orig
        status = "OK" if abs(diff) <= 5 else "MISMATCH"
        print(f"{lc:<6} {split:<7} {ours:<8} {orig:<10} {diff:+d}{'':>4} {status}")

# ─── 3. SLOT ANALYSIS ────────────────────────────────────────────
print("\n" + "=" * 80)
print("3. SLOT TYPE ANALYSIS (CRITICAL)")
print("=" * 80)

all_slot_types = set()
per_lang_slots = defaultdict(set)
per_lang_bio_tags = defaultdict(Counter)

for d in data:
    lc = d.get('language_code', '')
    tags = d.get('bio_tags', 'O').split()
    for t in tags:
        per_lang_bio_tags[lc][t] += 1
        if t.startswith('B-') or t.startswith('I-'):
            slot = t[2:]
            all_slot_types.add(slot)
            per_lang_slots[lc].add(slot)

print(f"\nOur slot types ({len(all_slot_types)} total):")
for s in sorted(all_slot_types):
    print(f"  {s}")

print(f"\nOriginal slot types: {len(ORIGINAL_SLOT_TYPES)}")
print(f"\nMISSING slot types ({len(ORIGINAL_SLOT_TYPES) - len(all_slot_types)} missing from original 84):")
missing_slots = set(ORIGINAL_SLOT_TYPES) - all_slot_types
for s in sorted(missing_slots):
    print(f"  - {s}")

extra_slots = all_slot_types - set(ORIGINAL_SLOT_TYPES)
if extra_slots:
    print(f"\nEXTRA slot types not in original ({len(extra_slots)}):")
    for s in sorted(extra_slots):
        print(f"  + {s}")

print(f"\nPer-language slot counts:")
print(f"{'Lang':<6} {'Ours':<8} {'Original':<10} {'Diff'}")
print("-" * 35)
for lc in sorted(ORIGINAL.keys()):
    ours = len(per_lang_slots.get(lc, set()))
    orig = ORIGINAL[lc]['slots']
    print(f"{lc:<6} {ours:<8} {orig:<10} {ours - orig:+d}")

# ─── 4. INTENT ANALYSIS ──────────────────────────────────────────
print("\n" + "=" * 80)
print("4. INTENT ANALYSIS")
print("=" * 80)

per_lang_intents = defaultdict(set)
for d in data:
    lc = d.get('language_code', '')
    per_lang_intents[lc].add(d.get('intent', '').lower())

our_intents = set()
for s in per_lang_intents.values():
    our_intents.update(s)

print(f"Our intents ({len(our_intents)}): {sorted(our_intents)}")
print(f"Original intents ({len(ORIGINAL_INTENT_TYPES)}): {sorted(ORIGINAL_INTENT_TYPES)}")

missing_intents = set(ORIGINAL_INTENT_TYPES) - our_intents
extra_intents = our_intents - set(ORIGINAL_INTENT_TYPES)
if missing_intents:
    print(f"\nMissing intents: {sorted(missing_intents)}")
if extra_intents:
    print(f"Extra intents: {sorted(extra_intents)}")

print(f"\nPer-language intent counts:")
print(f"{'Lang':<6} {'Ours':<8} {'Original':<10}")
print("-" * 25)
for lc in sorted(ORIGINAL.keys()):
    ours = len(per_lang_intents.get(lc, set()))
    orig = ORIGINAL[lc]['intents']
    flag = " !!!" if ours != orig else ""
    print(f"{lc:<6} {ours:<8} {orig:<10}{flag}")

# ─── 5. TOKEN COUNT ANALYSIS ─────────────────────────────────────
print("\n" + "=" * 80)
print("5. TOKEN COUNT ANALYSIS")
print("=" * 80)

per_lang_tokens = defaultdict(list)
for d in data:
    lc = d.get('language_code', '')
    tc = d.get('token_count', 0)
    per_lang_tokens[lc].append(tc)

print(f"{'Lang':<6} {'Avg Tokens':<12} {'Min':<6} {'Max':<6} {'Total':<10}")
print("-" * 45)
for lc in sorted(ORIGINAL.keys()):
    toks = per_lang_tokens.get(lc, [0])
    avg = sum(toks) / len(toks) if toks else 0
    total = sum(toks)
    orig_total = sum(ORIGINAL[lc][f'{s}_tok'] for s in ['train', 'dev', 'test'])
    print(f"{lc:<6} {avg:<12.1f} {min(toks):<6} {max(toks):<6} {total:<10} (orig: {orig_total})")

# ─── 6. CROSS-LINGUAL ALIGNMENT ──────────────────────────────────
print("\n" + "=" * 80)
print("6. CROSS-LINGUAL ALIGNMENT CHECK")
print("=" * 80)

en_commands = set(d['english_command'] for d in data if d.get('language_code') == 'en')
print(f"Unique English commands: {len(en_commands)}")

for lc in sorted(ORIGINAL.keys()):
    if lc == 'en':
        continue
    lang_cmds = set(d['english_command'] for d in data if d.get('language_code') == lc)
    missing = en_commands - lang_cmds
    extra = lang_cmds - en_commands
    coverage = len(lang_cmds & en_commands) / len(en_commands) * 100 if en_commands else 0
    print(f"  {lc}: {len(lang_cmds)} commands, {coverage:.1f}% EN coverage, {len(missing)} missing, {len(extra)} extra")

# ─── 7. EMPTY TRANSLATIONS ───────────────────────────────────────
print("\n" + "=" * 80)
print("7. EMPTY/MISSING TRANSLATIONS")
print("=" * 80)

for lc in sorted(ORIGINAL.keys()):
    lang_data = [d for d in data if d.get('language_code') == lc]
    empty = sum(1 for d in lang_data if not d.get('translated_command', '').strip())
    if empty > 0:
        print(f"  {lc}: {empty}/{len(lang_data)} entries have empty translated_command ({empty/len(lang_data)*100:.1f}%)")

# ─── 8. BIO TAG QUALITY ──────────────────────────────────────────
print("\n" + "=" * 80)
print("8. BIO TAG QUALITY CHECK")
print("=" * 80)

# Check entries where bio_tags = all O (no slots tagged)
for lc in sorted(ORIGINAL.keys()):
    lang_data = [d for d in data if d.get('language_code') == lc]
    all_o = sum(1 for d in lang_data if 'B-' not in d.get('bio_tags', ''))
    has_entity = sum(1 for d in lang_data if d.get('entity', '').strip())
    has_entity_but_all_o = sum(1 for d in lang_data 
                                if d.get('entity', '').strip() and 'B-' not in d.get('bio_tags', ''))
    print(f"  {lc}: {all_o}/{len(lang_data)} all-O ({all_o/len(lang_data)*100:.1f}%), "
          f"{has_entity_but_all_o} have entity but no B- tag")

# ─── 9. SAMPLE BIO COMPARISON ────────────────────────────────────
print("\n" + "=" * 80)
print("9. SAMPLE CROSS-LINGUAL BIO ALIGNMENT (like paper Figure)")
print("=" * 80)

# Find an entry like "show departures from atlanta for american"
target = None
for d in data:
    if d.get('language_code') == 'en' and 'atlanta' in d.get('english_command', '').lower() and 'american' in d.get('english_command', '').lower():
        target = d['english_command']
        break

if target:
    print(f"English command: '{target}'")
    for lc in sorted(ORIGINAL.keys()):
        matches = [d for d in data if d.get('language_code') == lc and d.get('english_command') == target]
        if matches:
            m = matches[0]
            print(f"\n  {lc}: {m.get('translated_command', 'N/A')}")
            print(f"      tokens: {m.get('tokens', [])}")
            print(f"      bio:    {m.get('bio_tags', 'N/A')}")
else:
    print("Could not find a suitable example command.")

# ─── 10. DOWNSAMPLING CHECK ──────────────────────────────────────
print("\n" + "=" * 80)
print("10. HINDI/TURKISH DOWNSAMPLING STATUS")
print("=" * 80)

for lc in ['hi', 'tr']:
    lang_data = [d for d in data if d.get('language_code') == lc]
    orig_total = sum(ORIGINAL[lc][s] for s in ['train', 'dev', 'test'])
    print(f"  {lc}: We have {len(lang_data)} entries, original has {orig_total}")
    
    intents = Counter(d.get('intent', '').lower() for d in lang_data)
    orig_intents = ORIGINAL[lc]['intents']
    print(f"      Our intents: {len(intents)}, Original: {orig_intents}")
    if lc == 'hi':
        print(f"      NOTE: Hindi in original excludes 1 intent (has 17, not 18)")
        print(f"      Hindi test set in original is 893 (same as full-resource languages)")
    elif lc == 'tr':
        print(f"      NOTE: Turkish in original excludes 1 intent (has 17, not 18)")
        print(f"      Turkish has reduced train/dev/test: 578/60/715")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
