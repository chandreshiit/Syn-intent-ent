#!/usr/bin/env python3
"""
MultiATIS++ synthetic-only transfer, per language.

For each of the 9 languages in
$SLU_GAP_DATA/multiatis_multilingual_pipeline/processed_data/, trains:
  1. Joint intent + slot BERT (bert-base-multilingual-uncased), slu_gap.models.bert_alone
  2. Joint intent + slot Bi-LSTM (randomly initialised), slu_gap.models.lstm_alone

Reports per-language intent accuracy and slot F1 on the per-language test split.

Languages are trained in separate subprocesses (see multiatis_bert_per_lang.py)
because training nine models in one process fragments CUDA memory badly enough
to OOM on the later languages.

Usage:
    python experiments/multiatis/multiatis_multilingual_baselines.py [--epochs 20] [--models bert lstm]

Outputs:
    results/multiatis/multiatis_multilingual_baselines.json
"""

import argparse
import json
import os
import shutil
import tempfile
import time

import torch

from slu_gap.models import bert_alone, lstm_alone
from slu_gap.models.utils import build_vocab


LANGUAGES = ["en", "es", "pt", "de", "fr", "zh", "ja", "hi", "tr"]
PROCESSED_DATA = "data/multiatis_multilingual_pipeline/processed_data"


def load_bio_split(lang_code, split):
    base = os.path.join(PROCESSED_DATA, lang_code, split)
    # Read without filtering empty lines so positional alignment is preserved.
    with open(os.path.join(base, "seq.in"), "r", encoding="utf-8") as f:
        utts_raw = [line.rstrip("\n") for line in f]
    with open(os.path.join(base, "seq.out"), "r", encoding="utf-8") as f:
        bio_raw = [line.rstrip("\n") for line in f]
    with open(os.path.join(base, "label"), "r", encoding="utf-8") as f:
        intents_raw = [line.rstrip("\n") for line in f]
    # Strip any trailing empty lines past the shortest file; the SNIPS/MultiATIS
    # writer occasionally leaves a stray empty seq.in line that doesn't appear
    # in seq.out / label.
    n = min(len(utts_raw), len(bio_raw), len(intents_raw))
    utts = [u.split() for u in utts_raw[:n]]
    bio = [b.split() for b in bio_raw[:n]]
    intents = [s.strip() for s in intents_raw[:n]]
    # Drop any row where ALL three are empty (truly blank rows at the end).
    keep = [(u, b, lbl) for u, b, lbl in zip(utts, bio, intents) if u or b or lbl.strip()]
    utts = [t[0] for t in keep]
    bio = [t[1] for t in keep]
    intents = [t[2] for t in keep]
    assert len(utts) == len(bio) == len(intents), \
        f"after cleanup, lengths still differ: {len(utts)} {len(bio)} {len(intents)}"
    # Align token-tag lengths
    for i, (u, s) in enumerate(zip(utts, bio)):
        if len(u) != len(s):
            if len(s) < len(u):
                bio[i] = s + ["O"] * (len(u) - len(s))
            else:
                bio[i] = s[:len(u)]
    return utts, bio, intents


def write_tsv(path, utts, bio, intents, start_uid=0):
    with open(path, "w", encoding="utf-8") as f:
        f.write("u_id\tutterance\tslot-labels\tintent\n")
        for i, (u, b, lbl) in enumerate(zip(utts, bio, intents)):
            f.write(f"{start_uid + i}\t{' '.join(u)}\t{' '.join(b)}\t{lbl}\n")


def run_lang(lang_code, args, tmp_root):
    print(f"\n{'=' * 60}\nLanguage: {lang_code}\n{'=' * 60}", flush=True)
    lang_dir = os.path.join(tmp_root, lang_code)
    os.makedirs(lang_dir, exist_ok=True)

    # Load + write TSVs
    train_tsv = os.path.join(lang_dir, "train.tsv")
    dev_tsv = os.path.join(lang_dir, "dev.tsv")
    test_tsv = os.path.join(lang_dir, "test.tsv")

    for split, path in (("train", train_tsv), ("dev", dev_tsv), ("test", test_tsv)):
        utts, bio, intents = load_bio_split(lang_code, split)
        write_tsv(path, utts, bio, intents)
    n_train = sum(1 for _ in open(train_tsv, encoding="utf-8")) - 1
    n_dev = sum(1 for _ in open(dev_tsv, encoding="utf-8")) - 1
    n_test = sum(1 for _ in open(test_tsv, encoding="utf-8")) - 1
    print(f"  train: {n_train}  dev: {n_dev}  test: {n_test}", flush=True)

    intent2idx, label2idx = bert_alone.get_label_indices(train_tsv)
    print(f"  intents={len(intent2idx)}  slot_labels={len(label2idx)}", flush=True)

    result = {"lang": lang_code, "n_train": n_train, "n_dev": n_dev, "n_test": n_test,
              "intents": len(intent2idx), "slot_labels": len(label2idx)}

    # --- BERT ---
    if "bert" in args.models:
        bert_alone.model_dir = os.path.join(lang_dir, "bert_ckpt")
        os.makedirs(bert_alone.model_dir, exist_ok=True)
        t0 = time.time()
        try:
            model = bert_alone.train(
                f"bert_{lang_code}", train_tsv, dev_tsv,
                intent2idx, label2idx, epochs=args.epochs,
            )
            train_secs = time.time() - t0
            tokenizer = bert_alone.BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
            intent_acc, slot_f1 = bert_alone.evaluate(
                model, test_tsv, tokenizer, intent2idx, label2idx,
                model_path=os.path.join(bert_alone.model_dir, f"bert_{lang_code}.pt"),
            )
            result["bert"] = {
                "intent_acc": float(intent_acc),
                "slot_f1": float(slot_f1),
                "train_secs": train_secs,
            }
            print(f"  BERT: intent_acc={intent_acc:.4f}  slot_f1={slot_f1:.4f}  secs={train_secs:.1f}", flush=True)
            del model
        except Exception as e:
            print(f"  BERT FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            result["bert"] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # --- BiLSTM ---
    if "lstm" in args.models:
        lstm_alone.model_dir = os.path.join(lang_dir, "lstm_ckpt")
        os.makedirs(lstm_alone.model_dir, exist_ok=True)
        # Build vocab from THIS language's train file
        vocab = build_vocab([train_tsv], min_freq=1)
        t0 = time.time()
        try:
            model = lstm_alone.train(
                f"lstm_{lang_code}", train_tsv, dev_tsv,
                vocab, intent2idx, label2idx, epochs=args.epochs,
            )
            train_secs = time.time() - t0
            intent_acc, slot_f1 = lstm_alone.evaluate(
                model, test_tsv, vocab, intent2idx, label2idx,
                model_path=os.path.join(lstm_alone.model_dir, f"lstm_{lang_code}.pt"),
            )
            result["lstm"] = {
                "intent_acc": float(intent_acc),
                "slot_f1": float(slot_f1),
                "train_secs": train_secs,
            }
            print(f"  LSTM: intent_acc={intent_acc:.4f}  slot_f1={slot_f1:.4f}  secs={train_secs:.1f}", flush=True)
            del model
        except Exception as e:
            print(f"  LSTM FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            result["lstm"] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Clean per-language tmp to free C: disk during sweep
    shutil.rmtree(lang_dir, ignore_errors=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=LANGUAGES,
                        choices=LANGUAGES)
    parser.add_argument("--models", nargs="+", default=["bert", "lstm"],
                        choices=["bert", "lstm"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="phase3/results")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    bert_alone.set_seed(args.seed)
    lstm_alone.set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "multiatis_multilingual_baselines.json")

    tmp_root = tempfile.mkdtemp(prefix="multiatis_ml_")
    print(f"Tmp dir: {tmp_root}")

    all_results = []
    for lang in args.languages:
        r = run_lang(lang, args, tmp_root)
        all_results.append(r)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_language": all_results, "config": vars(args)}, f, indent=2)

    # Print Table 1-style summary
    print("\n" + "=" * 70)
    print("MultiATIS++ multilingual baselines (synthetic data)")
    print("=" * 70)
    header = f"  {'Lang':<6} {'#train':>7} {'#test':>6} {'BERT_intent':>12} {'BERT_slotF1':>12} {'LSTM_intent':>12} {'LSTM_slotF1':>12}"
    print(header)
    for r in all_results:
        be = r.get("bert", {})
        ls = r.get("lstm", {})
        be_int = f"{be.get('intent_acc', 0):.4f}" if "intent_acc" in be else "n/a"
        be_slf = f"{be.get('slot_f1', 0):.4f}" if "slot_f1" in be else "n/a"
        ls_int = f"{ls.get('intent_acc', 0):.4f}" if "intent_acc" in ls else "n/a"
        ls_slf = f"{ls.get('slot_f1', 0):.4f}" if "slot_f1" in ls else "n/a"
        print(f"  {r['lang']:<6} {r['n_train']:>7} {r['n_test']:>6} {be_int:>12} {be_slf:>12} {ls_int:>12} {ls_slf:>12}")
    print(f"\nSaved: {summary_path}")

    shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
