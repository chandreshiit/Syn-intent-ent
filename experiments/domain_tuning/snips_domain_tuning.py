#!/usr/bin/env python3
"""
Domain-tuning sweep on SNIPS smart-lights (JointBERT arm).

Train JointBERT on (synthetic 100% + real x%) and evaluate on a real held-out
test set, for x in {0%, 5%, 10%, 25%, 50%, 100%}. Produces the annotation-
efficiency curve showing how much real data is needed to close the gap between
purely synthetic training and a real-only baseline.

Real data source:      $SLU_GAP_DATA/snips_real_close/dataset.json
Synthetic data source: $SLU_GAP_DATA/snips_multilingual_pipeline/processed_data/en/all/

The x=100% point is accompanied by a real_only baseline trained without any
synthetic data, which anchors the top of the curve.

The paper reports the Snips NLU (CRF) version of this sweep, since that is the
toolchain the original SNIPS benchmark used; see snips_domain_tuning_snipsnlu.py.
This JointBERT variant is the pretrained-encoder comparison.

Usage:
    python experiments/domain_tuning/snips_domain_tuning.py [--epochs 20] [--seed 42]
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import time
from collections import Counter

import torch
from sklearn.model_selection import train_test_split

from slu_gap.models import bert_alone


# Mapping from original SNIPS dataset.json intent names (CamelCase) to the
# lowercase intent labels our synthetic pipeline emits.
INTENT_NAME_MAP = {
    "DecreaseBrightness":  "decreasebrightness",
    "IncreaseBrightness":  "increasebrightness",
    "SetLightBrightness":  "setlightbrightness",
    "SetLightColor":       "setlightcolor",
    "SwitchLightOff":      "switchlightoff",
    "SwitchLightOn":       "switchlighton",
}


# ----- token / BIO helpers -----

import re
_TOKEN_RE = re.compile(r"\w+(?:'\w+)?")

def tokenize(text):
    return _TOKEN_RE.findall(text.lower())


# ----- real SNIPS loader -----

def load_real_snips(dataset_json="SNIPS/smart-lights-en-close-field/dataset.json"):
    """Parse SNIPS dataset.json into (utterance_tokens, bio_tags, intent_name) tuples.

    Each `utterance.data` is a list of segments; segments without `slot_name`
    contribute O tokens, segments with `slot_name` contribute B-/I- tagged tokens.
    """
    with open(dataset_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for intent_camel, intent_data in d.get("intents", {}).items():
        intent = INTENT_NAME_MAP.get(intent_camel, intent_camel.lower())
        for utt in intent_data.get("utterances", []):
            tokens, bio = [], []
            for seg in utt.get("data", []):
                text = seg.get("text", "")
                slot_name = seg.get("slot_name")
                seg_tokens = tokenize(text)
                if not seg_tokens:
                    continue
                if slot_name:
                    tokens.extend(seg_tokens)
                    bio.append(f"B-{slot_name}")
                    bio.extend([f"I-{slot_name}"] * (len(seg_tokens) - 1))
                else:
                    tokens.extend(seg_tokens)
                    bio.extend(["O"] * len(seg_tokens))
            if tokens:
                rows.append({"tokens": tokens, "bio": bio, "intent": intent})
    return rows


# ----- synthetic SNIPS loader -----

def load_synth_snips(base="data/snips_multilingual_pipeline/processed_data/en/all"):
    with open(os.path.join(base, "seq.in"), "r", encoding="utf-8") as f:
        in_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(base, "seq.out"), "r", encoding="utf-8") as f:
        out_lines = [l.strip() for l in f if l.strip()]
    with open(os.path.join(base, "label"), "r", encoding="utf-8") as f:
        lbl_lines = [l.strip() for l in f if l.strip()]
    assert len(in_lines) == len(out_lines) == len(lbl_lines)
    rows = []
    for u, b, l in zip(in_lines, out_lines, lbl_lines):
        tokens = u.split()
        bio = b.split()
        if len(tokens) != len(bio):
            if len(bio) < len(tokens):
                bio = bio + ["O"] * (len(tokens) - len(bio))
            else:
                bio = bio[:len(tokens)]
        rows.append({"tokens": tokens, "bio": bio, "intent": l})
    return rows


# ----- TSV writer -----

def write_tsv(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("u_id\tutterance\tslot-labels\tintent\n")
        for i, r in enumerate(rows):
            f.write(f"{i}\t{' '.join(r['tokens'])}\t{' '.join(r['bio'])}\t{r['intent']}\n")


# ----- experiment -----

RATIOS = [0.0, 0.05, 0.10, 0.25, 0.50, 1.00]


def stratified_subsample(rows, fraction, seed):
    """Return a stratified-by-intent subset that's `fraction` of `rows`."""
    if fraction >= 1.0:
        return list(rows)
    if fraction <= 0.0:
        return []
    by_intent = {}
    for r in rows:
        by_intent.setdefault(r["intent"], []).append(r)
    rng = random.Random(seed)
    out = []
    for intent, items in by_intent.items():
        n = max(1, int(round(len(items) * fraction))) if fraction > 0 else 0
        picked = rng.sample(items, min(n, len(items)))
        out.extend(picked)
    rng.shuffle(out)
    return out


def run_experiment(args):
    real_all = load_real_snips()
    synth_all = load_synth_snips()
    print(f"Real:      {len(real_all)} utts, intents={Counter(r['intent'] for r in real_all)}")
    print(f"Synthetic: {len(synth_all)} utts, intents={Counter(r['intent'] for r in synth_all)}")

    # Stratified 80/20 split on REAL data: train / test
    real_intents = [r["intent"] for r in real_all]
    real_train, real_test, _, _ = train_test_split(
        real_all, real_intents,
        test_size=0.20, stratify=real_intents,
        random_state=args.seed,
    )
    print(f"Real train: {len(real_train)}  Real test: {len(real_test)}")

    tmp_root = tempfile.mkdtemp(prefix="snips_dt_")
    bert_alone.model_dir = os.path.join(tmp_root, "ckpt")
    os.makedirs(bert_alone.model_dir, exist_ok=True)

    test_tsv = os.path.join(tmp_root, "real_test.tsv")
    write_tsv(test_tsv, real_test)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "snips_domain_tuning.json")

    results = []
    for ratio in RATIOS:
        real_subset = stratified_subsample(real_train, ratio, args.seed + int(ratio * 1000))
        train_rows = list(synth_all) + real_subset
        random.Random(args.seed).shuffle(train_rows)

        # Dev: 10% of train (random)
        rng = random.Random(args.seed)
        n_dev = max(1, int(round(0.1 * len(train_rows))))
        rng.shuffle(train_rows)
        dev_rows = train_rows[:n_dev]
        train_only = train_rows[n_dev:]

        ratio_dir = os.path.join(tmp_root, f"ratio_{int(ratio * 1000):04d}")
        os.makedirs(ratio_dir, exist_ok=True)
        train_tsv = os.path.join(ratio_dir, "train.tsv")
        dev_tsv = os.path.join(ratio_dir, "dev.tsv")
        write_tsv(train_tsv, train_only)
        write_tsv(dev_tsv, dev_rows)

        print(f"\n=== ratio={ratio:.2f} (real_added={len(real_subset)}, synth={len(synth_all)}, total_train={len(train_only)}) ===", flush=True)

        # Build label vocabs from train data so dev/test labels remain consistent
        intent2idx, label2idx = bert_alone.get_label_indices(train_tsv)
        print(f"  intents={len(intent2idx)} slot_labels={len(label2idx)}", flush=True)

        t0 = time.time()
        model_name = f"snips_dt_ratio_{int(ratio * 1000):04d}_seed_{args.seed}"
        try:
            model = bert_alone.train(model_name, train_tsv, dev_tsv, intent2idx, label2idx, epochs=args.epochs)
        except Exception as e:
            print(f"  TRAINING FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            raise
        train_secs = time.time() - t0

        tokenizer = bert_alone.BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        intent_acc, slot_f1 = bert_alone.evaluate(
            model, test_tsv, tokenizer, intent2idx, label2idx,
            model_path=os.path.join(bert_alone.model_dir, f"{model_name}.pt"),
        )

        results.append({
            "ratio": ratio,
            "real_added": len(real_subset),
            "synth_used": len(synth_all),
            "n_train": len(train_only),
            "n_dev": len(dev_rows),
            "n_test": len(real_test),
            "intent_acc": float(intent_acc),
            "slot_f1": float(slot_f1),
            "train_secs": train_secs,
        })
        print(f"  intent_acc={intent_acc:.4f}  slot_f1={slot_f1:.4f}  train_secs={train_secs:.1f}", flush=True)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_ratio": results, "config": {"seed": args.seed, "epochs": args.epochs, "test_n": len(real_test)}}, f, indent=2)

    # Also: real-only baseline (no synthetic)
    print(f"\n=== real-only baseline (no synthetic) ===", flush=True)
    real_only_rows = list(real_train)
    random.Random(args.seed).shuffle(real_only_rows)
    n_dev = max(1, int(round(0.1 * len(real_only_rows))))
    dev_rows = real_only_rows[:n_dev]
    train_only = real_only_rows[n_dev:]

    ratio_dir = os.path.join(tmp_root, "real_only")
    os.makedirs(ratio_dir, exist_ok=True)
    train_tsv = os.path.join(ratio_dir, "train.tsv")
    dev_tsv = os.path.join(ratio_dir, "dev.tsv")
    write_tsv(train_tsv, train_only)
    write_tsv(dev_tsv, dev_rows)

    intent2idx, label2idx = bert_alone.get_label_indices(train_tsv)
    t0 = time.time()
    model_name = f"snips_dt_real_only_seed_{args.seed}"
    try:
        model = bert_alone.train(model_name, train_tsv, dev_tsv, intent2idx, label2idx, epochs=args.epochs)
    except Exception as e:
        print(f"  TRAINING FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        raise
    train_secs = time.time() - t0
    tokenizer = bert_alone.BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    intent_acc, slot_f1 = bert_alone.evaluate(
        model, test_tsv, tokenizer, intent2idx, label2idx,
        model_path=os.path.join(bert_alone.model_dir, f"{model_name}.pt"),
    )
    real_only = {
        "ratio": "real_only",
        "real_added": len(real_train),
        "synth_used": 0,
        "n_train": len(train_only),
        "n_dev": len(dev_rows),
        "n_test": len(real_test),
        "intent_acc": float(intent_acc),
        "slot_f1": float(slot_f1),
        "train_secs": train_secs,
    }
    print(f"  intent_acc={intent_acc:.4f}  slot_f1={slot_f1:.4f}  train_secs={train_secs:.1f}", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_ratio": results,
            "real_only_baseline": real_only,
            "config": {"seed": args.seed, "epochs": args.epochs, "test_n": len(real_test)},
        }, f, indent=2)

    print(f"\nSaved: {summary_path}")
    print("\nSummary table:")
    print(f"  {'ratio':>10}  {'real_added':>10}  {'intent_acc':>10}  {'slot_f1':>10}")
    for r in results:
        print(f"  {r['ratio']:>10.2f}  {r['real_added']:>10}  {r['intent_acc']:>10.4f}  {r['slot_f1']:>10.4f}")
    print(f"  {'real_only':>10}  {real_only['real_added']:>10}  {real_only['intent_acc']:>10.4f}  {real_only['slot_f1']:>10.4f}")

    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="phase3/results")
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    bert_alone.set_seed(args.seed)
    run_experiment(args)


if __name__ == "__main__":
    main()
