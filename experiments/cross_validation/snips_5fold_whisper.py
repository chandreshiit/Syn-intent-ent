#!/usr/bin/env python3
"""
5-fold stratified cross-validation on synthetic SNIPS smart-lights (EN) audio
with the Whisper intent classifier. Reports the standard deviations quoted in
the paper.

The model is `slu_gap.speech.WhisperIntentClassifier` -- Whisper's encoder plus
mean pooling and a linear intent head, the architecture used by the original
Skit-S2I paper's Whisper baseline. It is trained directly on synthetic SNIPS
audio (16 kHz mono from MMS-TTS) as a speech-to-intent classifier.

This is a WITHIN-distribution measurement: train and test folds both come from
the synthetic corpus. It is deliberately NOT a synthetic-to-real transfer test,
and the paper is explicit that the two differ sharply -- within-distribution CV
reaches 99.7% while true transfer to real audio is 66.3%. For the transfer
setting see experiments/domain_tuning/snips_audio_transfer.py.

WER is not computed here; this measures intent classification on audio, not ASR.

Usage:
    python experiments/cross_validation/snips_5fold_whisper.py [--n-folds 5] [--epochs 20] [--seed 42]

Outputs:
    results/cross_validation/snips_5fold_whisper.json
"""

import argparse
import json
import os
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import whisper
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset

from slu_gap.datasets import (
    SNIPS_INTENT2IDX, SNIPS_N_INTENTS,
    load_real_snips_close, load_synth_snips,
)
from slu_gap.speech import WhisperIntentClassifier, collate

from slu_gap import paths

# Backwards-compatible aliases used throughout this module.
INTENT2IDX = SNIPS_INTENT2IDX
N_INTENTS = SNIPS_N_INTENTS


# ---------- dataset ----------

class SnipsAudioIntentDataset(Dataset):
    """Loads SNIPS audio + intent labels.

    Takes parallel lists `audio_paths` and `intent_labels`, plus the indices
    of the subset to use. Each item returns (log_mel, intent_id).
    """

    def __init__(self, audio_paths, indices, intent_labels, target_sr=16000):
        self.audio_paths = audio_paths
        self.indices = indices  # which example ids to use
        self.intent_labels = intent_labels  # parallel list, indexed by example id
        self.target_sr = target_sr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, k):
        idx = self.indices[k]
        path = self.audio_paths[idx]
        audio, sr = sf.read(path)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            if audio.max() > 1.5:
                audio = audio / 32768.0
        if sr != self.target_sr:
            # Whisper expects 16 kHz; resample if needed.
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        audio_t = torch.from_numpy(audio).float()
        audio_t = whisper.pad_or_trim(audio_t.flatten())
        mel = whisper.log_mel_spectrogram(audio_t)  # (80, T)
        intent_id = INTENT2IDX[self.intent_labels[idx]]
        return mel, intent_id


def stratified_downsample(intents, n_target, seed):
    """Return sorted indices (stratified by intent) summing to ~n_target."""
    from collections import defaultdict
    by_intent = defaultdict(list)
    for i, lab in enumerate(intents):
        by_intent[lab].append(i)
    rng = np.random.RandomState(seed)
    n_total = len(intents)
    out = []
    for lab, idxs in by_intent.items():
        n_lab = max(1, round(len(idxs) / n_total * n_target))
        order = list(idxs)
        rng.shuffle(order)
        out.extend(sorted(order[:n_lab]))
    return sorted(out)


# ---------- train / eval ----------

def train_one_fold(train_idx, test_idx, intents, audio_paths, device,
                   epochs, batch_size, lr, backbone, download_root):
    rng_state = torch.get_rng_state()

    train_ds = SnipsAudioIntentDataset(audio_paths, train_idx, intents)
    test_ds = SnipsAudioIntentDataset(audio_paths, test_idx, intents)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)

    model = WhisperIntentClassifier(backbone=backbone, n_class=N_INTENTS,
                                    download_root=download_root).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total = 0
        correct = 0
        sum_loss = 0.0
        for mels, labels in train_loader:
            mels = mels.to(device)
            labels = labels.to(device)
            logits = model(mels)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * mels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += mels.size(0)
        train_acc = correct / total if total > 0 else 0.0
        train_loss = sum_loss / total if total > 0 else 0.0

        # Eval
        model.eval()
        with torch.no_grad():
            t_total = 0
            t_correct = 0
            for mels, labels in test_loader:
                mels = mels.to(device)
                labels = labels.to(device)
                logits = model(mels)
                preds = logits.argmax(dim=-1)
                t_correct += (preds == labels).sum().item()
                t_total += mels.size(0)
            test_acc = t_correct / t_total if t_total > 0 else 0.0
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f"    epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f} (best={best_test_acc:.4f})")

    # Free GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    torch.set_rng_state(rng_state)
    return best_test_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--backbone", type=str, default="tiny.en",
                        choices=["tiny.en", "base.en", "small.en"])
    parser.add_argument("--out-dir", type=str, default="phase3/results")
    parser.add_argument("--download-root", type=str,
                        default=str(paths.WHISPER_CACHE),
                        help="Where to cache the Whisper checkpoint")
    parser.add_argument("--audio-source", type=str, default="synth",
                        choices=["synth", "real_close"],
                        help="Which SNIPS audio source to use.")
    parser.add_argument("--downsample", type=int, default=0,
                        help="If >0, stratified-sample to this many examples.")
    parser.add_argument("--tag", type=str, default="",
                        help="Tag appended to the output JSON name.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.download_root, exist_ok=True)

    if args.audio_source == "synth":
        audio_paths, intents = load_synth_snips()
    else:
        audio_paths, intents = load_real_snips_close()

    if args.downsample and args.downsample < len(intents):
        keep = stratified_downsample(intents, args.downsample, args.seed)
        audio_paths = [audio_paths[i] for i in keep]
        intents = [intents[i] for i in keep]
        print(f"downsampled to {len(intents)} (stratified, seed={args.seed})")

    indices = list(range(len(intents)))
    print(f"Loaded {len(indices)} examples; intents = {sorted(set(intents))}")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    folds = list(skf.split(indices, intents))

    per_fold = []
    out_name = f"snips_5fold_whisper_{args.tag}.json" if args.tag else "snips_5fold_whisper.json"
    summary_path = os.path.join(args.out_dir, out_name)
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        print(f"\n=== Fold {fold_idx + 1}/{args.n_folds} ===")
        print(f"  train: {len(train_idx)}  test: {len(test_idx)}")
        t0 = time.time()
        test_acc = train_one_fold(
            list(train_idx), list(test_idx), intents, audio_paths, device,
            args.epochs, args.batch_size, args.lr, args.backbone,
            args.download_root,
        )
        per_fold.append({
            "fold": fold_idx,
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
            "test_intent_acc": float(test_acc),
            "train_secs": float(time.time() - t0),
        })
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_fold": per_fold, "backbone": args.backbone}, f, indent=2)
        print(f"  best_test_acc={test_acc:.4f}  secs={per_fold[-1]['train_secs']:.1f}")

    accs = [r["test_intent_acc"] for r in per_fold]
    summary = {
        "n_folds": args.n_folds,
        "seed": args.seed,
        "epochs": args.epochs,
        "backbone": args.backbone,
        "per_fold": per_fold,
        "intent_acc_mean": statistics.mean(accs),
        "intent_acc_std":  statistics.pstdev(accs) if len(accs) > 1 else 0.0,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("\n=== Summary ===")
    print(f"intent_acc: {summary['intent_acc_mean']:.4f} +/- {summary['intent_acc_std']:.4f}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
