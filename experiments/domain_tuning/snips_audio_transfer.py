#!/usr/bin/env python3
"""
SNIPS synth-audio -> real-audio TRANSFER (the missing cross-distribution
experiment, analogous to the Skit-S2I domain-tuning sweep).

Unlike snips_5fold_whisper.py (which does WITHIN-distribution 5-fold CV:
synth->synth 99.7%, real->real 88.1%), this trains on synthetic SNIPS audio
and evaluates on REAL SNIPS close-field audio — the true transfer setting where
train/test speaker mismatch (single MMS-TTS voice vs many real speakers) bites,
and where voice cloning would be the relevant intervention.

Design mirrors phase3/skit_s2i_domain_tuning.py exactly:
  * Stratified 80/20 split of REAL close-field (seed 42) -> real_train / real_test
  * Ratios [0.0, 0.25, 1.0] of real_train mixed into full synthetic audio
  * real_only baseline (real_train only, no synth)
  * Whisper-tiny.en encoder + linear head, lr 1e-4, batch 16, 10 epochs
  * Test always = the held-out real_test (same across all conditions)

Output: phase3/results/snips_audio_transfer.json / .log
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
import whisper
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from slu_gap.datasets import (
    SNIPS_INTENT2IDX as INTENT2IDX,
    SNIPS_N_INTENTS as N_INTENTS,
    load_real_snips_close,
    load_synth_snips,
)
from slu_gap.speech import WhisperIntentClassifier, collate

_HERE = os.path.dirname(os.path.abspath(__file__))

REPO = os.path.dirname(_HERE)
DL_ROOT = os.path.join(REPO, ".hf_cache/whisper")
RATIOS = [0.0, 0.25, 1.0]


class MixedAudioDS(Dataset):
    """items: list of (audio_path, intent_label_str)."""
    def __init__(self, items, target_sr=16000):
        self.items = items
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, lab = self.items[k]
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if audio.max() > 1.5:
            audio = audio / 32768.0
        if sr != self.target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        at = whisper.pad_or_trim(torch.from_numpy(audio).float().flatten())
        return whisper.log_mel_spectrogram(at), INTENT2IDX[lab]


def stratified_subsample(items, fraction, seed):
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0:
        return []
    by = defaultdict(list)
    for it in items:
        by[it[1]].append(it)
    rng = random.Random(seed)
    out = []
    for lab, g in by.items():
        n = max(1, int(round(len(g) * fraction)))
        out.extend(rng.sample(g, min(n, len(g))))
    rng.shuffle(out)
    return out


def train_eval(train_items, test_items, device, epochs, bs, lr, backbone):
    tr = DataLoader(MixedAudioDS(train_items), batch_size=bs, shuffle=True,
                    num_workers=0, collate_fn=collate)
    te = DataLoader(MixedAudioDS(test_items), batch_size=bs, shuffle=False,
                    num_workers=0, collate_fn=collate)
    model = WhisperIntentClassifier(backbone=backbone, n_class=N_INTENTS,
                                    download_root=DL_ROOT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best = 0.0
    for ep in range(epochs):
        model.train()
        for mels, labels in tr:
            mels, labels = mels.to(device), labels.to(device)
            loss = loss_fn(model(mels), labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            c = t = 0
            for mels, labels in te:
                mels, labels = mels.to(device), labels.to(device)
                c += (model(mels).argmax(-1) == labels).sum().item()
                t += labels.size(0)
        acc = c / t if t else 0.0
        best = max(best, acc)
        print(f"    epoch {ep+1}/{epochs}: test_acc={acc:.4f} (best={best:.4f})", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--backbone", type=str, default="tiny.en")
    ap.add_argument("--out-name", type=str, default="snips_audio_transfer.json")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--synth-audio-dir", type=str, default=None,
                    help="Override synth audio dir (e.g. data/snips_f5_cloned_audio). "
                         "Expects cmd_XXXX_en.wav aligned with processed_data/en/all/label.")
    ap.add_argument("--skip-real-only", action="store_true",
                    help="Skip the real_only cell (reuse the one from a prior run).")
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.synth_audio_dir:
        # Same label file / index convention as load_synth_snips(), different audio dir.
        lab_path = "data/snips_multilingual_pipeline/processed_data/en/all/label"
        synth_intents = [l.strip() for l in open(lab_path, encoding="utf-8") if l.strip()]
        synth_paths = [os.path.join(args.synth_audio_dir, f"cmd_{i:04d}_en.wav")
                       for i in range(len(synth_intents))]
        missing = [p for p in synth_paths if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} synth wavs missing, e.g. {missing[0]}")
        print(f"synth audio override: {args.synth_audio_dir}")
    else:
        synth_paths, synth_intents = load_synth_snips()
    real_paths, real_intents = load_real_snips_close()
    synth_items = list(zip(synth_paths, synth_intents))
    real_items = list(zip(real_paths, real_intents))
    print(f"synth: {len(synth_items)}  real: {len(real_items)}")

    real_train, real_test = train_test_split(
        real_items, test_size=0.20, stratify=[x[1] for x in real_items],
        random_state=args.seed)
    print(f"real_train: {len(real_train)}  real_test: {len(real_test)}")

    out_path = os.path.join(REPO, "phase3/results", args.out_name)
    results = []
    real_only = None
    done = set()
    if args.resume and os.path.exists(out_path):
        prior = json.load(open(out_path))
        results = prior.get("per_ratio", [])
        done = {round(r["ratio"], 4) for r in results}
        real_only = prior.get("real_only_baseline")

    def save():
        payload = {"per_ratio": results,
                   "config": {"epochs": args.epochs, "lr": args.lr, "seed": args.seed,
                              "backbone": args.backbone, "n_synth": len(synth_items),
                              "n_real_train": len(real_train), "n_real_test": len(real_test)}}
        if real_only is not None:
            payload["real_only_baseline"] = real_only
        json.dump(payload, open(out_path, "w"), indent=2)

    for ratio in RATIOS:
        if round(ratio, 4) in done:
            print(f"=== ratio {ratio} SKIPPED ==="); continue
        sub = stratified_subsample(real_train, ratio, args.seed + int(ratio * 1000))
        train_items = list(synth_items) + sub
        random.Random(args.seed).shuffle(train_items)
        print(f"\n=== ratio={ratio:.2f} (real_added={len(sub)}, synth={len(synth_items)}, total={len(train_items)}) ===", flush=True)
        t0 = time.time()
        best = train_eval(train_items, real_test, device, args.epochs, args.batch_size, args.lr, args.backbone)
        results.append({"ratio": ratio, "real_added": len(sub), "synth_used": len(synth_items),
                        "n_train": len(train_items), "n_test": len(real_test),
                        "test_intent_acc": best, "train_secs": time.time() - t0})
        print(f"  best_test_acc={best:.4f}", flush=True)
        save()

    if real_only is None and not args.skip_real_only:
        print(f"\n=== real_only (n={len(real_train)}) ===", flush=True)
        t0 = time.time()
        best = train_eval(real_train, real_test, device, args.epochs, args.batch_size, args.lr, args.backbone)
        real_only = {"ratio": "real_only", "real_added": len(real_train), "synth_used": 0,
                     "n_train": len(real_train), "n_test": len(real_test),
                     "test_intent_acc": best, "train_secs": time.time() - t0}
        print(f"  best_test_acc={best:.4f}", flush=True)
        save()

    print("\n=== Summary (SNIPS synth-audio -> real-audio transfer) ===")
    for r in results:
        print(f"  ratio={r['ratio']:.2f}  real_added={r['real_added']:>4}  acc={r['test_intent_acc']:.4f}")
    if real_only:
        print(f"  real_only         real_added={real_only['real_added']:>4}  acc={real_only['test_intent_acc']:.4f}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
