#!/usr/bin/env python3
"""
5-fold stratified CV of the original Snips NLU engine (Python library) on
a Snips-NLU schema dataset.json. Matches the protocol of the SNIPS paper's
Tables 13 and 14 (Language Model Generalization Error, 5-fold CV).

Two modes:
  --dataset path/to/dataset.json     (real or synthetic Snips-NLU dataset)
  --tag    label used in output JSON

For each fold:
  * Train SnipsNLUEngine on 4/5 utterances (per-intent stratified split)
  * Predict on the held-out 1/5 utterances using their gold text
  * Compute per-intent precision/recall/F1 for intent classification
  * Compute per-slot precision/recall/F1 for slot filling (exact span match)

Outputs:
  phase3/results/snipsnlu_<tag>.json
  phase3/results/snipsnlu_<tag>.log (stdout tee)

This script runs in WSL with the .venv_wsl venv (Python 3.8 + snips-nlu 0.20.2).
"""

import argparse
import copy
import json
import os
import random
import statistics
import time
from collections import defaultdict


def split_dataset_by_index(ds_full, indices_per_intent):
    """Build a new dataset.json that contains only the given utterance indices
    per intent. Other top-level keys (entities, language) preserved as-is."""
    new = copy.deepcopy(ds_full)
    new["intents"] = {}
    for intent, idxs in indices_per_intent.items():
        utts = ds_full["intents"][intent]["utterances"]
        new["intents"][intent] = {"utterances": [utts[i] for i in idxs]}
    return new


def utt_to_text(u):
    return "".join(c.get("text", "") for c in u.get("data", [])).strip()


def gold_slots(u):
    """Return list of {entity, slot_name, raw_value} for an utterance."""
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
    """Extract list of {slot_name, raw_value} from a Snips NLU parse() output."""
    out = []
    for s in pred.get("slots", []):
        out.append({
            "slot_name": s.get("slotName"),
            "raw_value": s.get("rawValue", "").strip(),
        })
    return out


def stratified_kfold(by_intent, n_folds, seed):
    """For each intent, split its utterance indices into n_folds shuffled chunks.

    Returns: list of n_folds lists; each fold[k] = {intent: test_indices_for_intent}
    """
    rng = random.Random(seed)
    folds_per_intent = {}
    for intent, idxs in by_intent.items():
        shuffled = idxs[:]
        rng.shuffle(shuffled)
        chunks = [[] for _ in range(n_folds)]
        for i, v in enumerate(shuffled):
            chunks[i % n_folds].append(v)
        folds_per_intent[intent] = chunks
    fold_test = []
    for k in range(n_folds):
        fold_test.append({intent: folds_per_intent[intent][k] for intent in by_intent})
    return fold_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help="Path to Snips-NLU schema dataset.json")
    ap.add_argument("--tag", required=True,
                    help="Label for this run (e.g. real_close, synth_downsampled)")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="phase3/results")
    args = ap.parse_args()

    # snips-nlu imports inside main so any errors land in the log
    from snips_nlu import SnipsNLUEngine
    from snips_nlu.default_configs import CONFIG_EN

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, f"snipsnlu_{args.tag}.log")
    summary_path = os.path.join(args.out_dir, f"snipsnlu_{args.tag}.json")

    with open(args.dataset, "r", encoding="utf-8") as f:
        ds_full = json.load(f)

    intents = list(ds_full["intents"].keys())
    by_intent = {intent: list(range(len(ds_full["intents"][intent]["utterances"])))
                 for intent in intents}
    n_total = sum(len(v) for v in by_intent.values())
    print(f"loaded {args.dataset}  n_intents={len(intents)}  n_utts={n_total}")
    print(f"per intent: " + ", ".join(f"{i}={len(by_intent[i])}" for i in intents))

    fold_test_indices = stratified_kfold(by_intent, args.n_folds, args.seed)

    per_fold = []
    for k in range(args.n_folds):
        print(f"\n=== Fold {k+1}/{args.n_folds} ===", flush=True)
        test_idx = fold_test_indices[k]
        # train = everything not in test_idx per intent
        train_idx = {}
        for intent, all_idx in by_intent.items():
            test_set = set(test_idx[intent])
            train_idx[intent] = [i for i in all_idx if i not in test_set]

        train_ds = split_dataset_by_index(ds_full, train_idx)
        n_train = sum(len(v["utterances"]) for v in train_ds["intents"].values())
        n_test = sum(len(v) for v in test_idx.values())
        print(f"  train: {n_train}  test: {n_test}")

        # Train the Snips NLU engine
        t0 = time.time()
        engine = SnipsNLUEngine(config=CONFIG_EN)
        engine.fit(train_ds)
        train_secs = time.time() - t0
        print(f"  train_secs: {train_secs:.1f}")

        # Eval
        # Per-intent counters for intent: tp / fp / fn relative to gold intent.
        intent_tp = defaultdict(int)
        intent_fp = defaultdict(int)
        intent_fn = defaultdict(int)
        # Per-slot counters: keyed by (intent, slot_name)
        slot_tp = defaultdict(int)
        slot_fp = defaultdict(int)
        slot_fn = defaultdict(int)
        n_eval = 0

        for gold_intent, idxs in test_idx.items():
            for ui in idxs:
                u = ds_full["intents"][gold_intent]["utterances"][ui]
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
                # Slot evaluation done only on utterances where gold intent
                # matches predicted intent (Snips eval convention).
                gold_s = gold_slots(u)
                pred_s = predicted_slots(pred) if pred_intent == gold_intent else []
                # Build (slot_name, normalized_value) sets
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

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return p, r, f

        # Per-intent metrics
        intent_metrics = {}
        for intent in intents:
            p, r, f = prf(intent_tp[intent], intent_fp[intent], intent_fn[intent])
            intent_metrics[intent] = {
                "precision": p, "recall": r, "f1": f,
                "tp": intent_tp[intent], "fp": intent_fp[intent], "fn": intent_fn[intent],
            }
        # Micro-average intent
        all_tp = sum(intent_tp.values())
        all_fp = sum(intent_fp.values())
        all_fn = sum(intent_fn.values())
        micro_p, micro_r, micro_f = prf(all_tp, all_fp, all_fn)
        # Overall intent accuracy
        intent_acc = all_tp / n_eval if n_eval > 0 else 0.0

        # Per-slot metrics
        slot_metrics = {}
        for key in set(list(slot_tp.keys()) + list(slot_fp.keys()) + list(slot_fn.keys())):
            intent_name, slot_name = key
            p, r, f = prf(slot_tp[key], slot_fp[key], slot_fn[key])
            slot_metrics[f"{intent_name}::{slot_name}"] = {
                "precision": p, "recall": r, "f1": f,
                "tp": slot_tp[key], "fp": slot_fp[key], "fn": slot_fn[key],
            }
        # Micro slot
        s_tp = sum(slot_tp.values())
        s_fp = sum(slot_fp.values())
        s_fn = sum(slot_fn.values())
        s_p, s_r, s_f = prf(s_tp, s_fp, s_fn)

        fold_result = {
            "fold": k,
            "n_train": n_train,
            "n_test": n_test,
            "n_eval": n_eval,
            "train_secs": train_secs,
            "intent_accuracy": intent_acc,
            "intent_micro_p": micro_p,
            "intent_micro_r": micro_r,
            "intent_micro_f1": micro_f,
            "intent_per_intent": intent_metrics,
            "slot_micro_p": s_p,
            "slot_micro_r": s_r,
            "slot_micro_f1": s_f,
            "slot_per_intent": slot_metrics,
        }
        per_fold.append(fold_result)
        print(f"  intent_acc={intent_acc:.4f}  intent_micro_f1={micro_f:.4f}  slot_micro_f1={s_f:.4f}", flush=True)
        # Save snapshot
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_fold": per_fold, "tag": args.tag}, f, indent=2)

    # Aggregate
    def aggregate(field):
        vals = [r[field] for r in per_fold]
        return {
            "mean": statistics.mean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
        }

    summary = {
        "tag": args.tag,
        "dataset": args.dataset,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "per_fold": per_fold,
        "summary": {
            "intent_accuracy": aggregate("intent_accuracy"),
            "intent_micro_p": aggregate("intent_micro_p"),
            "intent_micro_r": aggregate("intent_micro_r"),
            "intent_micro_f1": aggregate("intent_micro_f1"),
            "slot_micro_p": aggregate("slot_micro_p"),
            "slot_micro_r": aggregate("slot_micro_r"),
            "slot_micro_f1": aggregate("slot_micro_f1"),
        },
    }
    # Mean per-intent table
    intent_means = {}
    for intent in intents:
        for metric in ("precision", "recall", "f1"):
            vals = [pf["intent_per_intent"][intent][metric] for pf in per_fold]
            intent_means.setdefault(intent, {})[metric] = {
                "mean": statistics.mean(vals),
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
    summary["intent_per_intent_mean"] = intent_means

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary across folds ===")
    print(f"intent_accuracy = {summary['summary']['intent_accuracy']['mean']:.4f} +/- {summary['summary']['intent_accuracy']['std']:.4f}")
    print(f"intent_micro_f1 = {summary['summary']['intent_micro_f1']['mean']:.4f} +/- {summary['summary']['intent_micro_f1']['std']:.4f}")
    print(f"slot_micro_f1   = {summary['summary']['slot_micro_f1']['mean']:.4f} +/- {summary['summary']['slot_micro_f1']['std']:.4f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
