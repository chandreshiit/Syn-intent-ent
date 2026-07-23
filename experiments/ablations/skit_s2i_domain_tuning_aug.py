#!/usr/bin/env python3
"""
R3 Whisper-tiny.en domain-tuning sweep WITH training-side audio augmentation.

Fork of skit_s2i_domain_tuning.py. The model, optimizer, dataset loading, ratio
sweep, and per_ratio/real_only_baseline structure are identical to the original
so results are directly comparable to phase3/results/skit_s2i_domain_tuning.json.

Augmentation (applied to TRAIN samples only, NEVER to test):
  * Speed perturbation on the waveform: p=0.5 keep, p=0.25 -> 0.9x, p=0.25 -> 1.1x
    (librosa.effects.time_stretch)
  * SpecAugment on the log-mel:
      - 2 time masks, each up to 20 frames wide
      - 2 frequency masks, each up to 15 bins wide

Outputs:
  phase3/results/skit_s2i_domain_tuning_aug.json
  phase3/results/skit_s2i_domain_tuning_aug.log

CLI is identical to the original script plus --no-aug (debug: disable aug entirely
to verify the fork matches baseline numbers).
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
from tqdm import tqdm  # noqa: F401

from slu_gap import paths

N_INTENTS = 14

# --------------- augmentation knobs (CLI-overridable) ---------------
SPEED_FACTORS = [1.0, 1.0, 0.9, 1.1]  # default: 50% keep, 25%+25% perturb
TIME_MASK_NUM = 2
TIME_MASK_MAX = 20
FREQ_MASK_NUM = 2
FREQ_MASK_MAX = 15


def apply_speed_perturbation(audio: np.ndarray, rng: random.Random) -> np.ndarray:
    factor = rng.choice(SPEED_FACTORS)
    if factor == 1.0:
        return audio
    # librosa.effects.time_stretch with rate>1.0 -> faster (shorter); rate<1.0 -> slower (longer)
    # We want speed factor: 0.9 (slower) and 1.1 (faster). librosa rate matches speed.
    try:
        return librosa.effects.time_stretch(y=audio, rate=factor)
    except TypeError:
        # Older librosa positional API
        return librosa.effects.time_stretch(audio, factor)


def apply_specaugment(mel: torch.Tensor, rng: random.Random) -> torch.Tensor:
    """In-place style SpecAugment on a (n_mels, n_frames) tensor."""
    n_mels, n_frames = mel.shape
    mel = mel.clone()
    # Time masks
    for _ in range(TIME_MASK_NUM):
        t = rng.randint(0, TIME_MASK_MAX)
        if t == 0 or t >= n_frames:
            continue
        t0 = rng.randint(0, n_frames - t)
        mel[:, t0:t0 + t] = 0.0
    # Frequency masks
    for _ in range(FREQ_MASK_NUM):
        f = rng.randint(0, FREQ_MASK_MAX)
        if f == 0 or f >= n_mels:
            continue
        f0 = rng.randint(0, n_mels - f)
        mel[f0:f0 + f, :] = 0.0
    return mel


# --------------- model ---------------

class WhisperIntentClassifier(nn.Module):
    FEATURE_DIMS = {
        "tiny.en": 384, "tiny": 384,
        "base.en": 512, "base": 512,
        "small.en": 768, "small": 768,
    }

    def __init__(self, backbone="tiny.en", n_class=N_INTENTS, download_root=None):
        super().__init__()
        self.encoder = whisper.load_model(backbone, download_root=download_root).encoder
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.classifier = nn.Linear(self.FEATURE_DIMS[backbone], n_class)

    def forward(self, mel):
        z = self.encoder(mel).mean(dim=1)
        return self.classifier(z)


# --------------- dataset ---------------

class MixedDataset(Dataset):
    """`items` already tagged with `_base`. `train_mode=True` enables aug."""

    def __init__(self, items, train_mode=False, aug_enabled=True, seed_offset=0):
        self.items = items
        self.train_mode = train_mode
        self.aug_enabled = aug_enabled
        self.seed_offset = seed_offset

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        # Per-item rng so the same epoch sample is deterministic per-process.
        # Use a different seed each epoch via Python's worker_init or accept
        # that DataLoader shuffles; for simplicity we seed off (idx + os.getpid).
        rng = random.Random(k * 7919 + self.seed_offset + os.getpid())
        path = os.path.join(it["_base"], it["audio_path"])
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if audio.max() > 1.5:
            audio = audio / 32768.0
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        if self.train_mode and self.aug_enabled:
            audio = apply_speed_perturbation(audio, rng)

        audio_t = torch.from_numpy(audio).float()
        audio_t = whisper.pad_or_trim(audio_t.flatten())
        mel = whisper.log_mel_spectrogram(audio_t)

        if self.train_mode and self.aug_enabled:
            mel = apply_specaugment(mel, rng)

        return mel, int(it["intent_class"])


def collate(batch):
    mels = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return mels, labels


# --------------- loaders ---------------

def load_real(real_dir="data/skit_s2i_real_audio"):
    rows = list(csv.DictReader(open(os.path.join(real_dir, "metadata.csv"), encoding="utf-8")))
    train, test = [], []
    for r in rows:
        it = {"audio_path": r["audio_path"], "intent_class": int(r["intent_class"])}
        (train if r["split"] == "train" else test).append(it)
    print(f"Real: train={len(train)}  test={len(test)}")
    return train, test, real_dir


def load_synth(csv_path="data/skit_s2i_synthesis_pipeline/output_v2/train.csv",
               audio_dir_base="data/skit_s2i_synthesis_pipeline"):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    items = [{"audio_path": r["audio_path"].replace("\\", "/"),
              "intent_class": int(r["intent_class"])} for r in rows]
    print(f"Synthetic: train={len(items)}")
    return items, audio_dir_base


# --------------- training ---------------

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
        n = max(1, int(round(len(lst) * fraction)))
        out.extend(rng.sample(lst, min(n, len(lst))))
    rng.shuffle(out)
    return out


def train_one_run(items, test_items, real_base, device, epochs, batch_size, lr,
                  backbone, download_root, aug_enabled):
    train_ds = MixedDataset(items, train_mode=True, aug_enabled=aug_enabled)
    test_ds = MixedDataset([{**it, "_base": real_base, "_synth": False} for it in test_items],
                            train_mode=False, aug_enabled=False)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=0, collate_fn=collate)
    model = WhisperIntentClassifier(backbone=backbone, n_class=N_INTENTS,
                                    download_root=download_root).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    for ep in range(epochs):
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
        if test_acc > best:
            best = test_acc
        print(f"    epoch {ep+1}/{epochs}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} test_acc={test_acc:.4f} (best={best:.4f})", flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best


def main():
    global SPEED_FACTORS, TIME_MASK_NUM, TIME_MASK_MAX, FREQ_MASK_NUM, FREQ_MASK_MAX
    p = argparse.ArgumentParser()
    p.add_argument("--ratios", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default="tiny.en")
    p.add_argument("--out-dir", type=str, default="phase3/results")
    p.add_argument("--download-root", type=str,
                   default=str(paths.WHISPER_CACHE))
    p.add_argument("--skip-real-only", action="store_true")
    p.add_argument("--no-aug", action="store_true",
                   help="Debug: disable augmentation entirely (should match baseline numbers)")
    p.add_argument("--out-suffix", type=str, default="_aug",
                   help="Filename suffix for outputs (default: _aug)")
    p.add_argument("--no-speed-perturb", action="store_true",
                   help="Disable speed perturbation; use SpecAugment only.")
    p.add_argument("--time-mask-num", type=int, default=TIME_MASK_NUM)
    p.add_argument("--time-mask-max", type=int, default=TIME_MASK_MAX)
    p.add_argument("--freq-mask-num", type=int, default=FREQ_MASK_NUM)
    p.add_argument("--freq-mask-max", type=int, default=FREQ_MASK_MAX)
    args = p.parse_args()

    # Apply CLI overrides to the module-level aug knobs
    if args.no_speed_perturb:
        SPEED_FACTORS = [1.0]
    TIME_MASK_NUM = args.time_mask_num
    TIME_MASK_MAX = args.time_mask_max
    FREQ_MASK_NUM = args.freq_mask_num
    FREQ_MASK_MAX = args.freq_mask_max

    torch.manual_seed(args.seed); random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.download_root, exist_ok=True)
    aug_enabled = not args.no_aug

    real_train, real_test, real_base = load_real()
    synth_train, synth_base = load_synth()

    summary_path = os.path.join(args.out_dir, f"skit_s2i_domain_tuning{args.out_suffix}.json")
    results = []
    synth_tagged = [{**it, "_base": synth_base, "_synth": True} for it in synth_train]

    for ratio in args.ratios:
        real_subset = stratified_subsample(real_train, ratio, args.seed + int(ratio * 1000))
        real_tagged = [{**it, "_base": real_base, "_synth": False} for it in real_subset]
        train_mix = list(synth_tagged) + real_tagged
        random.Random(args.seed).shuffle(train_mix)
        print(f"\n=== ratio={ratio:.2f} (real={len(real_subset)}, synth={len(synth_train)}, total={len(train_mix)}, aug={aug_enabled}) ===", flush=True)
        t0 = time.time()
        best = train_one_run(train_mix, real_test, real_base, device,
                             args.epochs, args.batch_size, args.lr, args.backbone,
                             args.download_root, aug_enabled)
        secs = time.time() - t0
        results.append({"ratio": ratio, "real_added": len(real_subset), "synth_used": len(synth_train),
                        "n_train": len(train_mix), "n_test": len(real_test),
                        "test_intent_acc": float(best), "train_secs": secs,
                        "aug_enabled": aug_enabled})
        print(f"  best_test_acc={best:.4f}  secs={secs:.1f}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_ratio": results, "config": vars(args), "aug_enabled": aug_enabled,
                       "aug_params": {"speed_factors": SPEED_FACTORS,
                                       "time_mask_num": TIME_MASK_NUM, "time_mask_max": TIME_MASK_MAX,
                                       "freq_mask_num": FREQ_MASK_NUM, "freq_mask_max": FREQ_MASK_MAX}},
                      f, indent=2)

    real_only = None
    if not args.skip_real_only:
        items = [{**it, "_base": real_base, "_synth": False} for it in real_train]
        print(f"\n=== real_only_baseline (real={len(items)}, aug={aug_enabled}) ===", flush=True)
        t0 = time.time()
        best = train_one_run(items, real_test, real_base, device,
                             args.epochs, args.batch_size, args.lr, args.backbone,
                             args.download_root, aug_enabled)
        secs = time.time() - t0
        real_only = {"ratio": "real_only", "real_added": len(real_train), "synth_used": 0,
                     "n_train": len(real_train), "n_test": len(real_test),
                     "test_intent_acc": float(best), "train_secs": secs,
                     "aug_enabled": aug_enabled}
        print(f"  best_test_acc={best:.4f}  secs={secs:.1f}", flush=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({"per_ratio": results, "real_only_baseline": real_only,
                       "config": vars(args), "aug_enabled": aug_enabled,
                       "aug_params": {"speed_factors": SPEED_FACTORS,
                                       "time_mask_num": TIME_MASK_NUM, "time_mask_max": TIME_MASK_MAX,
                                       "freq_mask_num": FREQ_MASK_NUM, "freq_mask_max": FREQ_MASK_MAX}},
                      f, indent=2)
    print(f"\nResults saved to {summary_path}")


if __name__ == "__main__":
    main()
