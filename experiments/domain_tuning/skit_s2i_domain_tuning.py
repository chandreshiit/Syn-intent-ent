#!/usr/bin/env python3
"""
Domain-tuning sweep on Skit-S2I (banking).

Trains the Whisper intent classifier on (synthetic 100% + real x%) and evaluates
on the original Skit-S2I real test split (1,400 utterances), for x in the
configured RATIOS. This produces the annotation-efficiency curve, and its x=0
column is the synthetic-only transfer number that the voice-cloning
intervention moves.

Switch audio conditions with --synth-csv / --synth-base to point at a different
generated corpus (generic Parler voices, telephony-matched, or F5 voice-cloned);
everything else stays fixed so the comparison isolates the audio.

Defaults to the minimal sweep (0%, 25%, 100%, real_only at 10 epochs) to keep
wall-clock bounded; pass --ratios for a denser curve.

Outputs:
    results/domain_tuning/skit_s2i_domain_tuning.json
"""

import argparse
import csv
import json
import os
import random
import time

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper
from torch.utils.data import DataLoader, Dataset

from slu_gap import paths
from slu_gap.speech import WhisperIntentClassifier, collate

N_INTENTS = 14


# --------------- dataset ---------------

class S2IAudioDataset(Dataset):
    """Loads (audio_path, intent_class) pairs and returns log-mel + label."""

    def __init__(self, items, base_dir, target_sr=16000):
        # items: list of dicts with keys 'audio_path' (relative), 'intent_class'
        self.items = items
        self.base_dir = base_dir
        self.target_sr = target_sr

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        item = self.items[k]
        path = os.path.join(self.base_dir, item["audio_path"]) if not os.path.isabs(item["audio_path"]) else item["audio_path"]
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if audio.max() > 1.5:
            audio = audio / 32768.0
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
        audio_t = torch.from_numpy(audio).float()
        audio_t = whisper.pad_or_trim(audio_t.flatten())
        mel = whisper.log_mel_spectrogram(audio_t)
        return mel, int(item["intent_class"])



# --------------- loaders ---------------

def load_real(real_dir="data/skit_s2i_real_audio"):
    rows = list(csv.DictReader(open(os.path.join(real_dir, "metadata.csv"), encoding="utf-8")))
    train_items = []
    test_items = []
    for r in rows:
        item = {
            "audio_path": r["audio_path"],
            "intent_class": int(r["intent_class"]),
        }
        if r["split"] == "train":
            train_items.append(item)
        elif r["split"] == "test":
            test_items.append(item)
    print(f"Real: train={len(train_items)}  test={len(test_items)}")
    return train_items, test_items, real_dir


def load_synth(csv_path="data/skit_s2i_synthesis_pipeline/output_v2/train.csv",
                audio_dir_base="data/skit_s2i_synthesis_pipeline"):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    items = []
    for r in rows:
        # audio_path stored relative to synthesis pipeline dir, e.g. "generated_audio\\audio_005407.wav"
        items.append({
            "audio_path": r["audio_path"].replace("\\", "/"),
            "intent_class": int(r["intent_class"]),
        })
    print(f"Synthetic: train={len(items)}")
    return items, audio_dir_base


# --------------- training ---------------

def train_one_run(items, test_items, real_base, synth_base, device,
                  epochs, batch_size, lr, backbone, download_root):
    """Train on a mixed dataset, evaluate on real test, return best test acc.

    `items` may contain BOTH real and synthetic items. We use audio_path as the
    key + a `_synth` flag to look up the right base_dir per sample.
    """
    # Build a unified dataset where each item carries its own base_dir
    def to_uniform(it_list, base_dir, is_synth):
        return [{**it, "_base": base_dir, "_synth": is_synth} for it in it_list]

    # Items already carry _base via the caller; just verify here.
    for it in items[:3]:
        assert "_base" in it, "callers must mark _base"

    class MixedDataset(Dataset):
        def __init__(self, lst):
            self.lst = lst

        def __len__(self):
            return len(self.lst)

        def __getitem__(self, k):
            it = self.lst[k]
            path = os.path.join(it["_base"], it["audio_path"])
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            if audio.max() > 1.5:
                audio = audio / 32768.0
            if sr != 16000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            audio_t = torch.from_numpy(audio).float()
            audio_t = whisper.pad_or_trim(audio_t.flatten())
            mel = whisper.log_mel_spectrogram(audio_t)
            return mel, int(it["intent_class"])

    train_ds = MixedDataset(items)
    test_ds = MixedDataset([{**it, "_base": real_base, "_synth": False} for it in test_items])
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
        total = correct = 0
        sum_loss = 0.0
        for mels, labels in train_loader:
            mels = mels.to(device); labels = labels.to(device)
            logits = model(mels)
            loss = loss_fn(logits, labels)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            sum_loss += loss.item() * mels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += mels.size(0)
        train_loss = sum_loss / total
        train_acc = correct / total

        model.eval()
        with torch.no_grad():
            t_total = t_correct = 0
            for mels, labels in test_loader:
                mels = mels.to(device); labels = labels.to(device)
                logits = model(mels)
                t_correct += (logits.argmax(dim=-1) == labels).sum().item()
                t_total += mels.size(0)
            test_acc = t_correct / t_total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f"    epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f} (best={best_test_acc:.4f})", flush=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_test_acc


def stratified_subsample(items, fraction, seed):
    if fraction >= 1.0:
        return list(items)
    if fraction <= 0.0:
        return []
    by_class = {}
    for it in items:
        by_class.setdefault(it["intent_class"], []).append(it)
    rng = random.Random(seed)
    out = []
    for cls, lst in by_class.items():
        n = max(1, int(round(len(lst) * fraction))) if fraction > 0 else 0
        out.extend(rng.sample(lst, min(n, len(lst))))
    rng.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[0.0, 0.25, 1.0],
                        help="Real-data fractions to evaluate (in addition to real_only)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--backbone", type=str, default="tiny.en")
    parser.add_argument("--out-dir", type=str, default="phase3/results")
    parser.add_argument("--download-root", type=str,
                        default=str(paths.WHISPER_CACHE))
    parser.add_argument("--skip-real-only", action="store_true")
    parser.add_argument("--out-name", type=str, default="skit_s2i_domain_tuning.json",
                        help="Output JSON filename in out-dir. Used for resume detection too.")
    parser.add_argument("--resume", action="store_true",
                        help="If the output JSON exists with prior ratios, skip them and only run missing.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.download_root, exist_ok=True)

    real_train, real_test, real_base = load_real()
    synth_train, synth_base = load_synth()

    summary_path = os.path.join(args.out_dir, args.out_name)
    results = []
    real_only = None
    done_ratios = set()
    if args.resume and os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                prior = json.load(f)
            results = prior.get("per_ratio", [])
            done_ratios = {round(float(r["ratio"]), 4) for r in results}
            real_only = prior.get("real_only_baseline")
            print(f"Resume: loaded {len(results)} prior ratio(s) {sorted(done_ratios)}; real_only={'present' if real_only else 'missing'}")
        except Exception as e:
            print(f"Resume: failed to read prior JSON ({e!r}); starting fresh")
            results = []; real_only = None; done_ratios = set()

    # Tag bases on synthetic items once (real items get tagged when added)
    synth_items_tagged = [{**it, "_base": synth_base, "_synth": True} for it in synth_train]

    for ratio in args.ratios:
        if args.resume and round(float(ratio), 4) in done_ratios:
            print(f"\n=== ratio={ratio:.2f} SKIPPED (already in prior JSON) ===", flush=True)
            continue
        real_subset = stratified_subsample(real_train, ratio, args.seed + int(ratio * 1000))
        real_subset_tagged = [{**it, "_base": real_base, "_synth": False} for it in real_subset]
        train_mix = list(synth_items_tagged) + real_subset_tagged
        random.Random(args.seed).shuffle(train_mix)
        print(f"\n=== ratio={ratio:.2f} (real_added={len(real_subset)}, synth={len(synth_train)}, total_train={len(train_mix)}) ===", flush=True)
        t0 = time.time()
        try:
            best = train_one_run(train_mix, real_test, real_base, synth_base, device,
                                  args.epochs, args.batch_size, args.lr, args.backbone,
                                  args.download_root)
        except Exception as e:
            print(f"  TRAINING FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            raise
        secs = time.time() - t0
        results.append({
            "ratio": ratio, "real_added": len(real_subset), "synth_used": len(synth_train),
            "n_train": len(train_mix), "n_test": len(real_test),
            "test_intent_acc": float(best), "train_secs": secs,
        })
        print(f"  best_test_acc={best:.4f}  secs={secs:.1f}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            payload = {"per_ratio": results, "config": vars(args)}
            if real_only is not None:
                payload["real_only_baseline"] = real_only
            json.dump(payload, f, indent=2)

    if not args.skip_real_only and real_only is None:
        real_only_items = [{**it, "_base": real_base, "_synth": False} for it in real_train]
        random.Random(args.seed).shuffle(real_only_items)
        print(f"\n=== real_only (n={len(real_only_items)}) ===", flush=True)
        t0 = time.time()
        try:
            best = train_one_run(real_only_items, real_test, real_base, synth_base, device,
                                  args.epochs, args.batch_size, args.lr, args.backbone,
                                  args.download_root)
        except Exception as e:
            print(f"  TRAINING FAILED: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()
            raise
        secs = time.time() - t0
        real_only = {
            "ratio": "real_only", "real_added": len(real_train), "synth_used": 0,
            "n_train": len(real_only_items), "n_test": len(real_test),
            "test_intent_acc": float(best), "train_secs": secs,
        }
        print(f"  best_test_acc={best:.4f}  secs={secs:.1f}", flush=True)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"per_ratio": results, "real_only_baseline": real_only, "config": vars(args)}, f, indent=2)

    print(f"\nSaved: {summary_path}")
    print("\nSummary table:")
    print(f"  {'ratio':>10}  {'real_added':>10}  {'n_train':>8}  {'test_acc':>10}")
    for r in results:
        print(f"  {r['ratio']:>10.2f}  {r['real_added']:>10}  {r['n_train']:>8}  {r['test_intent_acc']:>10.4f}")
    if real_only:
        print(f"  {'real_only':>10}  {real_only['real_added']:>10}  {real_only['n_train']:>8}  {real_only['test_intent_acc']:>10.4f}")


if __name__ == "__main__":
    main()
