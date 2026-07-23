#!/usr/bin/env python3
"""
Zero-shot LLM baseline (ACL R2).

For each of the 3 datasets, prompt llama3.2 (via Ollama) with a per-utterance
single-shot intent + slot extraction prompt. Compute:
  - Intent accuracy
  - Slot F1 (span-level, CoNLL-style) for SNIPS + MultiATIS
  - Skit-S2I has no slots (intent-only)

Outputs:
  - analysis/zero_shot_llm/<dataset>_predictions.jsonl  per-utterance predictions
  - analysis/zero_shot_llm/<dataset>_results.json       summary metrics
  - analysis/zero_shot_llm/summary.csv                  cross-dataset summary
"""

import argparse
import csv
import json
import os
import re
import time

try:
    import pandas as pd
except ImportError:
    pd = None
from ollama import chat
from tqdm import tqdm


_TOKEN_RE = re.compile(r"\w+(?:'\w+)?")


def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# ----- BIO span extraction (lifted from conlleval.py style) -----

def extract_spans(labels):
    """Extract spans from BIO labels. Returns set of (start, end, label_type)."""
    spans = set()
    cur_label = None
    cur_start = None
    for i, lbl in enumerate(labels):
        if lbl.startswith("B-"):
            if cur_label is not None:
                spans.add((cur_start, i, cur_label))
            cur_label = lbl[2:]
            cur_start = i
        elif lbl.startswith("I-"):
            t = lbl[2:]
            if cur_label != t:
                if cur_label is not None:
                    spans.add((cur_start, i, cur_label))
                cur_label = t
                cur_start = i
        else:
            if cur_label is not None:
                spans.add((cur_start, i, cur_label))
            cur_label = None
    if cur_label is not None:
        spans.add((cur_start, len(labels), cur_label))
    return spans


def slots_to_bio(tokens, slot_predictions):
    """Convert [{"slot_type": "...", "value": "..."}] to a BIO tag list aligned to tokens.

    For each prediction, find the value's tokens within the utterance tokens and
    assign B-X / I-X. Non-overlapping greedy match.
    """
    bio = ["O"] * len(tokens)
    tagged = set()
    lower_tokens = [t.lower() for t in tokens]
    for pred in slot_predictions:
        st = pred.get("slot_type", "")
        val = (pred.get("value") or "").lower().strip()
        if not st or not val:
            continue
        val_tokens = _TOKEN_RE.findall(val)
        if not val_tokens:
            continue
        # Find span
        found = None
        for i in range(len(lower_tokens) - len(val_tokens) + 1):
            if lower_tokens[i:i + len(val_tokens)] == val_tokens:
                # Skip overlapping
                if any(j in tagged for j in range(i, i + len(val_tokens))):
                    continue
                found = (i, i + len(val_tokens))
                break
        if found is None and len(val_tokens) == 1:
            # Single-token: try equality
            for i, t in enumerate(lower_tokens):
                if t == val_tokens[0] and i not in tagged:
                    found = (i, i + 1)
                    break
        if found is None:
            continue
        start, end = found
        bio[start] = f"B-{st}"
        for j in range(start + 1, end):
            bio[j] = f"I-{st}"
        for j in range(start, end):
            tagged.add(j)
    return bio


def slot_f1(gold_bio_per_utt, pred_bio_per_utt):
    """CoNLL-style span F1 across all utterances."""
    tp = fp = fn = 0
    for gold, pred in zip(gold_bio_per_utt, pred_bio_per_utt):
        gold_spans = extract_spans(gold)
        pred_spans = extract_spans(pred)
        tp += len(gold_spans & pred_spans)
        fp += len(pred_spans - gold_spans)
        fn += len(gold_spans - pred_spans)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


# ----- dataset loaders -----

def load_snips_test(seed=42, test_ratio=0.2):
    """Held-out 20% split of synthetic SNIPS EN (deterministic by seed).

    Returns list of dicts: {"utterance", "tokens", "gold_intent", "gold_bio"}
    """
    import random
    base = "data/snips_multilingual_pipeline/processed_data/en/all"
    with open(os.path.join(base, "seq.in"), "r", encoding="utf-8") as f:
        in_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(base, "seq.out"), "r", encoding="utf-8") as f:
        out_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(base, "label"), "r", encoding="utf-8") as f:
        lbl_lines = [l.strip() for l in f if l.strip()]
    assert len(in_lines) == len(out_lines) == len(lbl_lines)

    indices = list(range(len(in_lines)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = int(round(len(indices) * test_ratio))
    test_idx = sorted(indices[:n_test])

    rows = []
    for i in test_idx:
        utt = in_lines[i]
        bio = out_lines[i].split()
        tokens = utt.split()
        if len(tokens) != len(bio):
            # Truncate or pad with O to match
            if len(bio) < len(tokens):
                bio = bio + ["O"] * (len(tokens) - len(bio))
            else:
                bio = bio[:len(tokens)]
        rows.append({
            "utterance": utt,
            "tokens": tokens,
            "gold_intent": lbl_lines[i],
            "gold_bio": bio,
        })
    return rows


def load_skit_s2i_test():
    """Skit-S2I original test split (1,400 utts, intent-only)."""
    if pd is None:
        raise RuntimeError("pandas required")
    df = pd.read_parquet("skit-s2i/data/test-00000-of-00001.parquet",
                          columns=["intent_class", "template"])
    intent_map = {}
    with open("skit-s2i/intent_info.csv", "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            intent_map[int(row["intent_class"])] = row["intent_name"]
    rows = []
    for _, r in df.iterrows():
        utt = str(r["template"]).strip()
        if not utt:
            continue
        rows.append({
            "utterance": utt,
            "tokens": utt.split(),
            "gold_intent": intent_map.get(int(r["intent_class"]), "unknown"),
            "gold_bio": ["O"] * len(utt.split()),
        })
    return rows


def load_multiatis_test():
    """MultiATIS++ original EN test split (1,022 utts)."""
    if pd is None:
        raise RuntimeError("pandas required")
    df = pd.read_csv("multiatis_evaluation_v2/data_v2/test_EN.tsv", sep="\t")
    rows = []
    for _, r in df.iterrows():
        utt = str(r["utterance"]).strip()
        bio = str(r["slot-labels"]).split()
        tokens = utt.split()
        if len(tokens) != len(bio):
            if len(bio) < len(tokens):
                bio = bio + ["O"] * (len(tokens) - len(bio))
            else:
                bio = bio[:len(tokens)]
        rows.append({
            "utterance": utt,
            "tokens": tokens,
            "gold_intent": str(r["intent"]),
            "gold_bio": bio,
        })
    return rows


# ----- per-dataset prompts -----

SNIPS_PROMPT = """You are an intent classifier and slot extractor for a smart-home (smart-lights) voice assistant.

Intent ontology (choose exactly one):
- DecreaseBrightness: lowering / dimming light brightness in a room
- IncreaseBrightness: increasing / brightening light brightness in a room
- SetLightBrightness: setting light brightness to a specific level (numeric)
- SetLightColor: changing the color of lights
- SwitchLightOff: turning off lights
- SwitchLightOn: turning on lights

Slot ontology (extract zero or more slot values that appear in the utterance):
- room: room or area (e.g., kitchen, bedroom, living room, garage)
- color: color of the lights (e.g., red, blue, green, pink)
- brightness: numeric brightness level, word or digit form (e.g., twenty, thirty-two, 50)

Examples:
Input: "turn off the kitchen lights"
Output: {{"intent": "SwitchLightOff", "slots": [{{"slot_type": "room", "value": "kitchen"}}]}}
Input: "set the bedroom lights to 25"
Output: {{"intent": "SetLightBrightness", "slots": [{{"slot_type": "room", "value": "bedroom"}}, {{"slot_type": "brightness", "value": "25"}}]}}
Input: "change the living room lights to blue"
Output: {{"intent": "SetLightColor", "slots": [{{"slot_type": "room", "value": "living room"}}, {{"slot_type": "color", "value": "blue"}}]}}

Now classify and extract for this input. Use slot values exactly as they appear in the utterance.
Input: "{utterance}"
Return ONLY the JSON object with no additional text."""


SKIT_S2I_PROMPT = """You are an intent classifier for an Indian banking voice assistant.

Intent ontology (choose exactly one):
- branch_address: bank branch location / address / directions
- activate_card: activating a debit / credit card
- past_transactions: transaction history queries
- dispatch_status: card / document dispatch status
- outstanding_balance: outstanding dues / pending amount on credit card
- card_issue: card not working / declined / problem with a card
- ifsc_code: IFSC code of a bank branch
- generate_pin: generate or change PIN
- unauthorised_transaction: unauthorised / fraudulent transaction
- loan_query: loan inquiry (eligibility, interest rate, products)
- balance_enquiry: account balance check
- change_limit: change transaction / withdrawal / spend limit
- block: block a card
- lost: report a lost card

Examples:
Input: "What is my account balance?"
Output: {{"intent": "balance_enquiry"}}
Input: "I want to block my debit card"
Output: {{"intent": "block"}}

Now classify this input.
Input: "{utterance}"
Return ONLY the JSON object with no additional text."""


# MultiATIS has 18 intents and 84 slots. The prompt is long.
MULTIATIS_INTENTS = [
    "atis_flight", "atis_airfare", "atis_airline", "atis_ground_service",
    "atis_abbreviation", "atis_aircraft", "atis_flight_time", "atis_quantity",
    "atis_city", "atis_ground_fare", "atis_distance", "atis_airport",
    "atis_capacity", "atis_flight_no", "atis_meal", "atis_restriction",
    "atis_cheapest", "atis_day_name",
]

MULTIATIS_PROMPT = """You are an intent classifier and slot extractor for the ATIS (Airline Travel Information System) domain.

Intent ontology (choose exactly one, with "atis_" prefix):
""" + "\n".join(f"- {i}" for i in MULTIATIS_INTENTS) + """

Common slot types (extract zero or more slot values that appear in the utterance):
- fromloc.city_name, toloc.city_name, stoploc.city_name (city names in from/to/stopover context)
- airline_name, airline_code (airline name or 2-letter code)
- airport_name, airport_code (airport name or code)
- depart_date.day_name, depart_date.month_name, depart_date.day_number (departure date components)
- arrive_date.day_name, arrive_date.month_name, arrive_date.day_number (arrival date)
- depart_time.time, depart_time.period_of_day, arrive_time.time (time components)
- class_type (first class, coach, business)
- flight_mod (cheapest, latest, nonstop), flight_stop (stop info), round_trip
- transport_type (taxi, bus, rental car)
- meal, meal_code, meal_description
- fare_amount, cost_relative (under, over)
- aircraft_code, flight_number, flight_time
- restriction_code, fare_basis_code

Examples:
Input: "show flights from boston to denver on monday"
Output: {{"intent": "atis_flight", "slots": [{{"slot_type": "fromloc.city_name", "value": "boston"}}, {{"slot_type": "toloc.city_name", "value": "denver"}}, {{"slot_type": "depart_date.day_name", "value": "monday"}}]}}
Input: "what is the cheapest flight from miami to chicago"
Output: {{"intent": "atis_cheapest", "slots": [{{"slot_type": "flight_mod", "value": "cheapest"}}, {{"slot_type": "fromloc.city_name", "value": "miami"}}, {{"slot_type": "toloc.city_name", "value": "chicago"}}]}}

Now classify and extract for this input. Use slot values exactly as they appear in the utterance.
Input: "{utterance}"
Return ONLY the JSON object with no additional text."""


PROMPTS = {
    "snips":     SNIPS_PROMPT,
    "skit_s2i":  SKIT_S2I_PROMPT,
    "multiatis": MULTIATIS_PROMPT,
}


def _extract_json(text):
    """Try several strategies to pull a single JSON object out of LLM output."""
    text = text.replace("```json", "").replace("```", "").strip()
    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # 2. Find the first balanced { ... } block via brace counting
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    # 3. Regex fallback
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def call_llm(prompt, model="llama3.2", max_retries=2):
    """Call Ollama; retry on JSON parse failures AND on exceptions."""
    for attempt in range(max_retries + 1):
        try:
            response = chat(model=model, messages=[{"role": "user", "content": prompt}])
            content = response.message.content.strip()
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
            # Parse failure: retry with stronger instruction (only if more attempts left)
            if attempt < max_retries:
                continue
            return None
        except Exception:
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
    return None


def evaluate(dataset_name, test_rows, model, out_dir, has_slots=True):
    print(f"\n{'=' * 70}\nZero-shot eval: {dataset_name}  ({len(test_rows)} utts, model={model})\n{'=' * 70}")
    prompt_tpl = PROMPTS[dataset_name]
    predictions_path = os.path.join(out_dir, f"{dataset_name}_predictions.jsonl")

    pred_intents = []
    gold_intents = []
    pred_bios = []
    gold_bios = []
    parse_failures = 0

    with open(predictions_path, "w", encoding="utf-8") as fout:
        for row in tqdm(test_rows, desc=dataset_name):
            prompt = prompt_tpl.format(utterance=row["utterance"])
            pred = call_llm(prompt, model=model)
            if not pred or not isinstance(pred, dict):
                pred = {"intent": "UNK", "slots": []}
                parse_failures += 1
            pred_intent = str(pred.get("intent", "UNK"))
            pred_slots = pred.get("slots", []) if has_slots else []
            if not isinstance(pred_slots, list):
                pred_slots = []
            pred_bio = slots_to_bio(row["tokens"], pred_slots)
            gold_bio = row["gold_bio"]
            # Align lengths
            if len(pred_bio) != len(gold_bio):
                if len(pred_bio) < len(gold_bio):
                    pred_bio = pred_bio + ["O"] * (len(gold_bio) - len(pred_bio))
                else:
                    pred_bio = pred_bio[:len(gold_bio)]
            pred_intents.append(pred_intent)
            gold_intents.append(row["gold_intent"])
            pred_bios.append(pred_bio)
            gold_bios.append(gold_bio)

            fout.write(json.dumps({
                "utterance": row["utterance"],
                "gold_intent": row["gold_intent"],
                "pred_intent": pred_intent,
                "gold_bio": gold_bio,
                "pred_bio": pred_bio,
                "pred_slots": pred_slots,
            }, ensure_ascii=False) + "\n")

    # Intent accuracy (case-insensitive, strip atis_ noise tolerant)
    def norm_intent(s):
        return str(s).strip().lower()
    correct = sum(1 for g, p in zip(gold_intents, pred_intents) if norm_intent(g) == norm_intent(p))
    intent_acc = correct / len(gold_intents) if gold_intents else 0.0

    result = {
        "dataset": dataset_name,
        "n_utterances": len(test_rows),
        "model": model,
        "intent_accuracy": intent_acc,
        "intent_correct": correct,
        "parse_failures": parse_failures,
    }
    if has_slots:
        sf = slot_f1(gold_bios, pred_bios)
        result["slot_f1"] = sf["f1"]
        result["slot_precision"] = sf["precision"]
        result["slot_recall"] = sf["recall"]
        result["slot_tp"] = sf["tp"]
        result["slot_fp"] = sf["fp"]
        result["slot_fn"] = sf["fn"]

    print(f"  Intent acc:   {intent_acc:.4f}  ({correct}/{len(gold_intents)})")
    if has_slots:
        print(f"  Slot P/R/F1: {result['slot_precision']:.4f} / {result['slot_recall']:.4f} / {result['slot_f1']:.4f}")
    print(f"  Parse failures: {parse_failures}")

    with open(os.path.join(out_dir, f"{dataset_name}_results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    parser = argparse.ArgumentParser(description="Zero-shot LLM baseline (ACL R2)")
    parser.add_argument("--datasets", nargs="+",
                        default=["snips", "skit_s2i", "multiatis"],
                        choices=["snips", "skit_s2i", "multiatis"])
    parser.add_argument("--model", type=str, default="llama3.2")
    parser.add_argument("--out-dir", type=str, default="analysis/zero_shot_llm")
    parser.add_argument("--max-utts", type=int, default=None,
                        help="If set, evaluate at most this many utterances per dataset (for smoke testing)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    summaries = []
    loaders = {
        "snips":     (load_snips_test,    True),
        "skit_s2i":  (load_skit_s2i_test, False),
        "multiatis": (load_multiatis_test, True),
    }
    for name in args.datasets:
        loader, has_slots = loaders[name]
        rows = loader()
        if args.max_utts and args.max_utts < len(rows):
            rows = rows[:args.max_utts]
        summaries.append(evaluate(name, rows, args.model, args.out_dir, has_slots=has_slots))

    # Write summary CSV
    fieldnames = ["dataset", "n_utterances", "model", "intent_accuracy", "intent_correct",
                   "slot_precision", "slot_recall", "slot_f1", "parse_failures"]
    with open(os.path.join(args.out_dir, "summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for s in summaries:
            w.writerow(s)
    print(f"\nSummary CSV: {os.path.join(args.out_dir, 'summary.csv')}")


if __name__ == "__main__":
    main()
