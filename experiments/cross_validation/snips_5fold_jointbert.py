#!/usr/bin/env python3
"""
5-fold stratified cross-validation on synthetic SNIPS smart-lights (EN) with
JointBERT. Reports the standard deviations quoted in the paper.

Reuses BERTForICSL + ICSLLoss + train/evaluate from `slu_gap.models.bert_alone`
(a joint intent + slot model, despite the file name inherited from the
MultiATIS++ repository). This script adds a thin orchestration layer that:

  1. Loads SNIPS BIO data from
     $SLU_GAP_DATA/snips_multilingual_pipeline/processed_data/en/all/
     ({seq.in, seq.out, label})
  2. Builds 5 stratified-by-intent folds
  3. For each fold k=0..4: writes train/eval TSVs in MultiATIS shape, trains a
     fresh BERTForICSL, evaluates on the held-out fold, records intent accuracy
     and slot F1
  4. Prints and writes mean +/- std across folds

Note that this is a WITHIN-distribution measurement: train and test come from
the same synthetic source. It is not a synthetic-to-real transfer test; see
experiments/domain_tuning/snips_audio_transfer.py for that.

Usage:
    python experiments/cross_validation/snips_5fold_jointbert.py [--epochs 30] [--seed 42]

Outputs:
    results/cross_validation/snips_5fold_jointbert.json
"""

import argparse
import json
import os
import random
import shutil
import statistics
import tempfile
import time

import torch
from sklearn.model_selection import StratifiedKFold

from slu_gap import paths
from slu_gap.models import bert_alone


# ----- data helpers -----

def load_snips_bio(seq_in_path, seq_out_path, label_path):
    """Load SNIPS BIO data into parallel lists (utterances, slot_labels, intents)."""
    with open(seq_in_path, "r", encoding="utf-8") as f:
        utterances = [line.strip().split() for line in f if line.strip()]
    with open(seq_out_path, "r", encoding="utf-8") as f:
        slot_labels = [line.strip().split() for line in f if line.strip()]
    with open(label_path, "r", encoding="utf-8") as f:
        intents = [line.strip() for line in f if line.strip()]
    assert len(utterances) == len(slot_labels) == len(intents), (
        f"mismatched lengths: {len(utterances)}, {len(slot_labels)}, {len(intents)}"
    )
    # Align token-tag lengths
    for i, (u, s) in enumerate(zip(utterances, slot_labels)):
        if len(u) != len(s):
            if len(s) < len(u):
                slot_labels[i] = s + ["O"] * (len(u) - len(s))
            else:
                slot_labels[i] = s[:len(u)]
    return utterances, slot_labels, intents


def write_tsv(path, indices, utterances, slot_labels, intents):
    """Write a subset to MultiATIS-shaped TSV."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("u_id\tutterance\tslot-labels\tintent\n")
        for i in indices:
            f.write(f"{i}\t{' '.join(utterances[i])}\t{' '.join(slot_labels[i])}\t{intents[i]}\n")


# ----- 5-fold CV -----

def run_kfold(args):
    base = args.bio_dir
    utterances, slot_labels, intents = load_snips_bio(
        os.path.join(base, "seq.in"),
        os.path.join(base, "seq.out"),
        os.path.join(base, "label"),
    )
    print(f"Loaded {len(utterances)} utterances; intents = {sorted(set(intents))}")

    # Stratified 5-fold by intent
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(utterances, intents))
    print(f"Generated {args.n_folds} stratified folds")

    # Temp dir for per-fold TSVs and per-fold model checkpoints; deleted at end.
    fold_results = []
    if args.tmp_root:
        os.makedirs(args.tmp_root, exist_ok=True)
        tmp_root = tempfile.mkdtemp(prefix="snips_kfold_", dir=args.tmp_root)
    else:
        tmp_root = tempfile.mkdtemp(prefix="snips_kfold_")
    print(f"Temp dir: {tmp_root}")

    # Patch bert_alone module-level config to point at fold data dir + ckpt dir.
    bert_alone.model_dir = os.path.join(tmp_root, "ckpt")
    os.makedirs(bert_alone.model_dir, exist_ok=True)

    out_name = f"snips_5fold_jointbert_{args.tag}.json" if args.tag else "snips_5fold_jointbert.json"
    summary_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        fold_dir = os.path.join(tmp_root, f"fold_{fold_idx}")
        os.makedirs(fold_dir, exist_ok=True)
        train_tsv = os.path.join(fold_dir, "train_EN.tsv")
        dev_tsv = os.path.join(fold_dir, "dev_EN.tsv")
        test_tsv = os.path.join(fold_dir, "test_EN.tsv")

        # Hold out 10% of train as dev (for early stopping in inner training loop)
        rng = random.Random(args.seed + fold_idx)
        train_idx_list = list(train_idx)
        rng.shuffle(train_idx_list)
        n_dev = max(1, int(round(0.1 * len(train_idx_list))))
        dev_indices = train_idx_list[:n_dev]
        train_only = train_idx_list[n_dev:]

        write_tsv(train_tsv, train_only, utterances, slot_labels, intents)
        write_tsv(dev_tsv, dev_indices, utterances, slot_labels, intents)
        write_tsv(test_tsv, list(test_idx), utterances, slot_labels, intents)

        print(f"\n=== Fold {fold_idx + 1}/{args.n_folds} ===", flush=True)
        print(f"  train: {len(train_only)}  dev: {len(dev_indices)}  test: {len(test_idx)}", flush=True)

        # Build label vocab from THIS fold's training data
        intent2idx, label2idx = bert_alone.get_label_indices(train_tsv)
        print(f"  intents={len(intent2idx)}  slot_labels={len(label2idx)}")

        # Train
        t0 = time.time()
        model_name = f"snips_fold_{fold_idx}_seed_{args.seed}"
        try:
            model = bert_alone.train(model_name, train_tsv, dev_tsv, intent2idx, label2idx, epochs=args.epochs)
        except Exception as e:
            print(f"  FOLD {fold_idx} TRAINING FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            raise
        train_secs = time.time() - t0

        # Test on held-out fold
        tokenizer = bert_alone.BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
        intent_acc, slot_f1 = bert_alone.evaluate(
            model, test_tsv, tokenizer, intent2idx, label2idx,
            model_path=os.path.join(bert_alone.model_dir, f"{model_name}.pt"),
        )
        fold_results.append({
            "fold": fold_idx,
            "n_train": len(train_only),
            "n_dev": len(dev_indices),
            "n_test": len(test_idx),
            "intent_acc": float(intent_acc),
            "slot_f1": float(slot_f1),
            "train_secs": train_secs,
        })
        print(f"  intent_acc={intent_acc:.4f}  slot_f1={slot_f1:.4f}  train_secs={train_secs:.1f}", flush=True)

        # Save running snapshot
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_fold": fold_results}, f, indent=2)

        # Explicitly free GPU memory between folds
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Summary stats
    intent_accs = [r["intent_acc"] for r in fold_results]
    slot_f1s = [r["slot_f1"] for r in fold_results]
    summary = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "per_fold": fold_results,
        "intent_acc_mean": statistics.mean(intent_accs),
        "intent_acc_std":  statistics.pstdev(intent_accs) if len(intent_accs) > 1 else 0.0,
        "slot_f1_mean":    statistics.mean(slot_f1s),
        "slot_f1_std":     statistics.pstdev(slot_f1s) if len(slot_f1s) > 1 else 0.0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("Summary across folds:")
    print(f"  intent_acc: {summary['intent_acc_mean']:.4f} +/- {summary['intent_acc_std']:.4f}")
    print(f"  slot_f1:    {summary['slot_f1_mean']:.4f} +/- {summary['slot_f1_std']:.4f}")
    print(f"Saved: {summary_path}")

    # Clean up temp dir (preserves only the summary JSON in args.out_dir)
    if not args.keep_tmp:
        shutil.rmtree(tmp_root, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per fold (default 20; bert_alone default is 50)")
    parser.add_argument("--out-dir", type=str, default="phase3/results")
    parser.add_argument("--keep-tmp", action="store_true",
                        help="Keep the temp fold dirs + per-fold checkpoints for debugging")
    parser.add_argument("--bio-dir", type=str,
                        default="data/snips_multilingual_pipeline/processed_data/en/all",
                        help="Directory with seq.in/seq.out/label files")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag appended to the output JSON name (e.g. real_close).")
    parser.add_argument("--tmp-root", type=str, default=str(paths.scratch("kfold")),
                        help="Where to put per-fold checkpoints. Defaults to "
                             "$SLU_GAP_SCRATCH/kfold; point it at a roomy volume, "
                             "since each fold writes a full BERT checkpoint.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    bert_alone.set_seed(args.seed)

    run_kfold(args)


if __name__ == "__main__":
    main()
