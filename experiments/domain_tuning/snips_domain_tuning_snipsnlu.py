#!/usr/bin/env python3
"""
R3 SNIPS domain-tuning sweep — mirror of phase3/snips_domain_tuning.py but
using the original Snips NLU library (CRF intent + slot tagger from the SNIPS
paper) instead of JointBERT. Apples-to-apples comparison: same 80/20 real
split, same seed=42, same ratios.

Conditions evaluated:
  RATIOS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00] of real_train added to full synth
  + real_only_baseline (only real_train, no synth)

Test = the 20% held-out real test split (deterministic via sklearn
train_test_split with random_state=42, matching snips_domain_tuning.py exactly).

Outputs:
  phase3/results/snips_domain_tuning_snipsnlu.json
  phase3/results/snips_domain_tuning_snipsnlu.log

Runs in WSL .venv_wsl (Python 3.8 + snips-nlu 0.20.2).
"""
import argparse
import copy
import json
import os
import random
import time
from collections import Counter, defaultdict


RATIOS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]
REAL_DATASET = "data/snips_real_close/dataset.json"
SYNTH_DATASET = "data/snips_synth_for_snipsnlu/dataset.json"


def flatten_dataset(ds):
    """Return list of (intent_name, utterance_dict) tuples."""
    out = []
    for intent, intent_data in ds["intents"].items():
        for utt in intent_data["utterances"]:
            out.append((intent, utt))
    return out


def build_dataset(language, intents_list, entities_template, utts_by_intent):
    """Build a fresh dataset.json from per-intent utterance lists.

    `entities_template` is the merged entity block (we keep union from real+synth).
    """
    out = {
        "language": language,
        "intents": {i: {"utterances": utts_by_intent.get(i, [])} for i in intents_list},
        "entities": entities_template,
    }
    return out


def merge_entities(real_ents, synth_ents):
    """Union of entity values from both datasets. Keeps same intents-of-entity types."""
    merged = {}
    keys = set(real_ents.keys()) | set(synth_ents.keys())
    for k in keys:
        r = real_ents.get(k, {})
        s = synth_ents.get(k, {})
        if not r:
            merged[k] = copy.deepcopy(s)
            continue
        if not s:
            merged[k] = copy.deepcopy(r)
            continue
        out = copy.deepcopy(r)
        # If both define `data` (custom entity), union by `value` field
        if "data" in r and "data" in s:
            seen = {item["value"].lower() for item in r["data"]}
            union = list(r["data"])
            for item in s["data"]:
                v = item["value"].lower()
                if v not in seen:
                    union.append(item)
                    seen.add(v)
            out["data"] = union
        merged[k] = out
    return merged


def stratified_subsample(items, fraction, seed):
    """items: list of (intent, utt) tuples. Returns stratified subset."""
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0:
        return []
    by_intent = defaultdict(list)
    for intent, utt in items:
        by_intent[intent].append((intent, utt))
    rng = random.Random(seed)
    out = []
    for intent, group in by_intent.items():
        n = max(1, int(round(len(group) * fraction)))
        out.extend(rng.sample(group, min(n, len(group))))
    rng.shuffle(out)
    return out


def utt_to_text(u):
    return "".join(c.get("text", "") for c in u.get("data", [])).strip()


def gold_slots(u):
    out = []
    for c in u.get("data", []):
        if c.get("slot_name"):
            out.append({
                "slot_name": c["slot_name"],
                "entity": c.get("entity"),
                "raw_value": c.get("text", "").strip(),
            })
    return out


def predicted_slots(pred):
    out = []
    for s in pred.get("slots", []):
        out.append({"slot_name": s.get("slotName"),
                    "raw_value": s.get("rawValue", "").strip()})
    return out


def evaluate(engine, test_items):
    """Return dict of intent_accuracy, slot_micro_f1, plus tp/fp/fn counters."""
    intent_tp = defaultdict(int); intent_fp = defaultdict(int); intent_fn = defaultdict(int)
    slot_tp = defaultdict(int); slot_fp = defaultdict(int); slot_fn = defaultdict(int)
    n_eval = 0
    for gold_intent, u in test_items:
        text = utt_to_text(u)
        if not text:
            continue
        pred = engine.parse(text)
        pred_intent = pred.get("intent", {}).get("intentName")
        if pred_intent == gold_intent:
            intent_tp[gold_intent] += 1
        else:
            intent_fn[gold_intent] += 1
            if pred_intent is not None:
                intent_fp[pred_intent] += 1
        # Slot eval (Snips convention: only count slots where intent matched)
        gold_s = gold_slots(u)
        pred_s = predicted_slots(pred) if pred_intent == gold_intent else []
        gold_pairs = [(s["slot_name"], s["raw_value"].lower()) for s in gold_s]
        pred_pairs = [(s["slot_name"], s["raw_value"].lower()) for s in pred_s]
        gold_used = [False] * len(gold_pairs)
        for ps in pred_pairs:
            matched = False
            for j, gs in enumerate(gold_pairs):
                if not gold_used[j] and ps == gs:
                    slot_tp[(gold_intent, ps[0])] += 1
                    gold_used[j] = True
                    matched = True
                    break
            if not matched:
                slot_fp[(gold_intent, ps[0])] += 1
        for j, gs in enumerate(gold_pairs):
            if not gold_used[j]:
                slot_fn[(gold_intent, gs[0])] += 1
        n_eval += 1

    all_tp = sum(intent_tp.values())
    intent_acc = all_tp / n_eval if n_eval else 0.0
    s_tp = sum(slot_tp.values()); s_fp = sum(slot_fp.values()); s_fn = sum(slot_fn.values())
    s_p = s_tp / (s_tp + s_fp) if (s_tp + s_fp) > 0 else 0.0
    s_r = s_tp / (s_tp + s_fn) if (s_tp + s_fn) > 0 else 0.0
    s_f = 2 * s_p * s_r / (s_p + s_r) if (s_p + s_r) > 0 else 0.0
    return {
        "n_eval": n_eval, "intent_accuracy": intent_acc,
        "slot_micro_p": s_p, "slot_micro_r": s_r, "slot_micro_f1": s_f,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", default=REAL_DATASET)
    ap.add_argument("--synth", default=SYNTH_DATASET)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="phase3/results")
    ap.add_argument("--out-name", default="snips_domain_tuning_snipsnlu.json")
    ap.add_argument("--resume", action="store_true",
                    help="Skip ratios already present in the output JSON.")
    ap.add_argument("--ratios", type=float, nargs="+", default=None,
                    help="Override the ratio list (e.g. --ratios 0.0 for synth-only).")
    ap.add_argument("--skip-real-only", action="store_true",
                    help="Skip the real_only baseline run.")
    args = ap.parse_args()

    global RATIOS
    if args.ratios is not None:
        RATIOS = args.ratios

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, args.out_name)

    # Lazy import so any failure surfaces in the log
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN
    from sklearn.model_selection import train_test_split

    real_ds = json.load(open(args.real, encoding="utf-8"))
    synth_ds = json.load(open(args.synth, encoding="utf-8"))

    # Both datasets share the same 6 intents (verified manually).
    intents = list(real_ds["intents"].keys())
    print(f"intents: {intents}")
    print(f"real per intent: {[(i, len(real_ds['intents'][i]['utterances'])) for i in intents]}")
    print(f"synth per intent: {[(i, len(synth_ds['intents'][i]['utterances'])) for i in intents]}")

    real_flat = flatten_dataset(real_ds)
    synth_flat = flatten_dataset(synth_ds)
    real_intents_only = [it for (it, _) in real_flat]
    print(f"real total: {len(real_flat)}  synth total: {len(synth_flat)}")

    # 80/20 stratified split of REAL (deterministic; matches snips_domain_tuning.py)
    real_train, real_test, _, _ = train_test_split(
        real_flat, real_intents_only,
        test_size=0.20, stratify=real_intents_only,
        random_state=args.seed,
    )
    print(f"real_train: {len(real_train)}  real_test: {len(real_test)}")
    print(f"real_train per intent: {Counter(it for (it, _) in real_train)}")
    print(f"real_test per intent: {Counter(it for (it, _) in real_test)}")

    merged_entities = merge_entities(real_ds.get("entities", {}),
                                      synth_ds.get("entities", {}))

    # Resume handling
    results = []
    real_only = None
    done_ratios = set()
    if args.resume and os.path.exists(summary_path):
        prior = json.load(open(summary_path, encoding="utf-8"))
        results = prior.get("per_ratio", [])
        done_ratios = {round(float(r["ratio"]), 4) for r in results}
        real_only = prior.get("real_only_baseline")
        print(f"Resume: loaded {len(results)} ratios {sorted(done_ratios)}; real_only={'present' if real_only else 'missing'}")

    def save():
        payload = {
            "per_ratio": results,
            "config": {"seed": args.seed, "ratios": RATIOS,
                       "real_dataset": args.real, "synth_dataset": args.synth,
                       "n_real_train": len(real_train), "n_real_test": len(real_test),
                       "n_synth": len(synth_flat)},
        }
        if real_only is not None:
            payload["real_only_baseline"] = real_only
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    for ratio in RATIOS:
        rkey = round(float(ratio), 4)
        if args.resume and rkey in done_ratios:
            print(f"\n=== ratio={ratio:.2f} SKIPPED (already in prior JSON) ===", flush=True)
            continue
        real_subset = stratified_subsample(real_train, ratio,
                                            args.seed + int(ratio * 1000))
        train_items = list(synth_flat) + real_subset
        print(f"\n=== ratio={ratio:.2f} (real_added={len(real_subset)}, synth={len(synth_flat)}, total_train={len(train_items)}) ===", flush=True)

        # Rebuild a dataset.json from the merged training items
        by_intent = defaultdict(list)
        for intent, utt in train_items:
            by_intent[intent].append(utt)
        train_ds = build_dataset(real_ds.get("language", "en"), intents,
                                  merged_entities, by_intent)

        t0 = time.time()
        engine = SnipsNLUEngine(config=CONFIG_EN)
        engine.fit(train_ds)
        train_secs = time.time() - t0

        metrics = evaluate(engine, real_test)
        secs = time.time() - t0
        results.append({
            "ratio": ratio, "real_added": len(real_subset), "synth_used": len(synth_flat),
            "n_train": len(train_items), "n_test": len(real_test),
            "intent_accuracy": metrics["intent_accuracy"],
            "slot_micro_f1": metrics["slot_micro_f1"],
            "slot_micro_p": metrics["slot_micro_p"],
            "slot_micro_r": metrics["slot_micro_r"],
            "train_secs": train_secs, "eval_secs": secs - train_secs,
        })
        print(f"  intent_acc={metrics['intent_accuracy']:.4f}  slot_f1={metrics['slot_micro_f1']:.4f}  secs={secs:.1f}", flush=True)
        save()

    # real_only baseline
    if real_only is None and not args.skip_real_only:
        print(f"\n=== real_only_baseline (real_train={len(real_train)}, synth=0) ===", flush=True)
        by_intent = defaultdict(list)
        for intent, utt in real_train:
            by_intent[intent].append(utt)
        train_ds = build_dataset(real_ds.get("language", "en"), intents,
                                  real_ds.get("entities", {}), by_intent)
        t0 = time.time()
        engine = SnipsNLUEngine(config=CONFIG_EN)
        engine.fit(train_ds)
        train_secs = time.time() - t0
        metrics = evaluate(engine, real_test)
        secs = time.time() - t0
        real_only = {
            "ratio": "real_only", "real_added": len(real_train), "synth_used": 0,
            "n_train": len(real_train), "n_test": len(real_test),
            "intent_accuracy": metrics["intent_accuracy"],
            "slot_micro_f1": metrics["slot_micro_f1"],
            "slot_micro_p": metrics["slot_micro_p"],
            "slot_micro_r": metrics["slot_micro_r"],
            "train_secs": train_secs, "eval_secs": secs - train_secs,
        }
        print(f"  intent_acc={real_only['intent_accuracy']:.4f}  slot_f1={real_only['slot_micro_f1']:.4f}  secs={secs:.1f}", flush=True)
        save()

    print(f"\nResults: {summary_path}")
    print("\n=== Summary ===")
    print(f"{'ratio':>10}  {'real_added':>10}  {'intent_acc':>10}  {'slot_f1':>10}")
    for r in results:
        print(f"{r['ratio']:>10.2f}  {r['real_added']:>10}  {r['intent_accuracy']:>10.4f}  {r['slot_micro_f1']:>10.4f}")
    if real_only:
        print(f"{'real_only':>10}  {real_only['real_added']:>10}  {real_only['intent_accuracy']:>10.4f}  {real_only['slot_micro_f1']:>10.4f}")


if __name__ == "__main__":
    main()
