#!/usr/bin/env python3
"""
Skit-S2I identity-vs-channel diagnostic.

Trains Whisper-tiny.en on synth-only (current generated_audio = F5-v1), predicts
on the real Skit-S2I test split, and asks: do the errors cluster by SPEAKER
IDENTITY or by CHANNEL QUALITY (SNR / noise floor)?

For each real test utterance we record:
  gold_intent, pred_intent, correct, speaker_id, snr_db, noise_floor_rms, duration

Then we report:
  (A) error rate per speaker (11 speakers) + spread
  (B) error rate per SNR quartile + spread
  (C) which axis explains more variance (a simple eta^2-style comparison)

If per-speaker error rate varies widely while per-SNR-bucket error rate is flat,
that supports "identity dominates channel" for synthetic-speech SLU.

Output:
  phase3/results/skit_s2i_identity_diagnostic.json  (per-utt + summary)
  phase3/results/skit_s2i_identity_diagnostic.log
"""
import csv
import json
import os
import statistics
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import whisper
from torch.utils.data import DataLoader, Dataset

from slu_gap.speech import WhisperIntentClassifier, collate

_HERE = os.path.dirname(os.path.abspath(__file__))

REPO = os.path.dirname(_HERE)
N_INTENTS = 14
REAL_DIR = os.path.join(REPO, "data/skit_s2i_real_audio")
SYNTH_CSV = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline/output_v2/train.csv")
SYNTH_BASE = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline")
DL_ROOT = os.path.join(REPO, ".hf_cache/whisper")


def load_real_with_speaker(split="test"):
    rows = list(csv.DictReader(open(os.path.join(REAL_DIR, "metadata.csv"), encoding="utf-8")))
    out = []
    for r in rows:
        if r["split"] != split:
            continue
        out.append({
            "audio_path": r["audio_path"].replace("\\", "/"),
            "intent_class": int(r["intent_class"]),
            "speaker_id": int(r["speaker_id"]),
        })
    return out


def load_synth():
    rows = list(csv.DictReader(open(SYNTH_CSV, encoding="utf-8")))
    return [{"audio_path": r["audio_path"].replace("\\", "/"),
             "intent_class": int(r["intent_class"])} for r in rows]


class AudioDS(Dataset):
    def __init__(self, items, base):
        self.items, self.base = items, base

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        path = os.path.join(self.base, it["audio_path"])
        audio, sr = sf.read(path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if audio.max() > 1.5:
            audio = audio / 32768.0
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        at = whisper.pad_or_trim(torch.from_numpy(audio).float().flatten())
        return whisper.log_mel_spectrogram(at), int(it["intent_class"])


def acoustic_features(path):
    """SNR (dB), noise floor RMS, duration for one wav (8 kHz native)."""
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n = len(audio)
    dur = n / sr if sr else 0.0
    win = int(0.1 * sr)
    if win > 0 and n >= win * 2:
        nw = n // win
        wr = np.array([np.sqrt(np.mean(audio[i*win:(i+1)*win]**2) + 1e-12) for i in range(nw)])
        noise = float(np.percentile(wr, 10))
        speech = float(np.percentile(wr, 90))
    else:
        rms = float(np.sqrt(np.mean(audio**2) + 1e-12))
        noise = speech = rms
    snr = 20 * np.log10((speech + 1e-9) / (noise + 1e-9))
    return {"snr_db": float(snr), "noise_floor_rms": noise, "duration": dur}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-source", choices=["synth", "real"], default="synth",
                    help="synth = train on synthetic (identity diagnostic); "
                         "real = train on real_train (intrinsic-difficulty control).")
    ap.add_argument("--out-suffix", default="",
                    help="Suffix for output filename (e.g. _realonly_control).")
    args = ap.parse_args()

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    real_test = load_real_with_speaker("test")
    if args.train_source == "synth":
        train_items, train_base = load_synth(), SYNTH_BASE
        print(f"train=synth: {len(train_items)}  real_test: {len(real_test)}")
    else:
        train_items, train_base = load_real_with_speaker("train"), REAL_DIR
        print(f"train=real_train: {len(train_items)}  real_test: {len(real_test)}")

    train_loader = DataLoader(AudioDS(train_items, train_base), batch_size=16, shuffle=True,
                              num_workers=0, collate_fn=collate)
    # Test loader must preserve order so we can align preds with metadata.
    test_ds = AudioDS(real_test, REAL_DIR)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=0, collate_fn=collate)

    model = WhisperIntentClassifier(backbone="tiny.en", n_class=N_INTENTS,
                                    download_root=DL_ROOT).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training {args.train_source}-only (tiny.en, 10 epochs) ...", flush=True)
    for ep in range(10):
        model.train()
        for mels, labels in train_loader:
            mels, labels = mels.to(device), labels.to(device)
            logits = model(mels)
            loss = loss_fn(logits, labels)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f"  epoch {ep+1}/10 done ({time.time()-t_start:.0f}s)", flush=True)

    # Predict on real test (ordered)
    model.eval()
    preds = []
    with torch.no_grad():
        for mels, labels in test_loader:
            mels = mels.to(device)
            logits = model(mels)
            preds.extend(logits.argmax(dim=-1).cpu().tolist())
    assert len(preds) == len(real_test)

    # Attach acoustic features + build per-utt records
    per_utt = []
    for it, pred in zip(real_test, preds):
        feat = acoustic_features(os.path.join(REAL_DIR, it["audio_path"]))
        per_utt.append({
            "speaker_id": it["speaker_id"],
            "gold_intent": it["intent_class"],
            "pred_intent": pred,
            "correct": pred == it["intent_class"],
            **feat,
        })

    overall_acc = sum(r["correct"] for r in per_utt) / len(per_utt)
    print(f"\noverall synth-only intent acc on real test: {overall_acc:.4f}")

    # (A) per-speaker error rate
    by_spk = defaultdict(list)
    for r in per_utt:
        by_spk[r["speaker_id"]].append(r["correct"])
    spk_err = {s: 1 - (sum(v) / len(v)) for s, v in by_spk.items()}
    spk_err_vals = list(spk_err.values())

    # (B) per-SNR-quartile error rate
    snrs = sorted(r["snr_db"] for r in per_utt)
    q = [np.percentile(snrs, p) for p in (25, 50, 75)]
    def snr_bucket(x):
        if x <= q[0]: return "Q1_low_snr"
        if x <= q[1]: return "Q2"
        if x <= q[2]: return "Q3"
        return "Q4_high_snr"
    by_snr = defaultdict(list)
    for r in per_utt:
        by_snr[snr_bucket(r["snr_db"])].append(r["correct"])
    snr_err = {b: 1 - (sum(v) / len(v)) for b, v in by_snr.items()}
    snr_err_vals = list(snr_err.values())

    # (C) variance explained: eta^2 = SS_between / SS_total for each grouping
    def eta_sq(groups):
        all_vals = [x for g in groups.values() for x in g]
        grand = statistics.mean(all_vals)
        ss_tot = sum((x - grand) ** 2 for x in all_vals)
        ss_bet = sum(len(g) * (statistics.mean(g) - grand) ** 2 for g in groups.values() if g)
        return ss_bet / ss_tot if ss_tot > 0 else 0.0
    eta_speaker = eta_sq(by_spk)
    eta_snr = eta_sq(by_snr)

    # (D) WITHIN-speaker SNR effect (confound-controlled): for each speaker,
    # split its utts at its own median SNR, compare error in low vs high half.
    by_spk_recs = defaultdict(list)
    for r in per_utt:
        by_spk_recs[r["speaker_id"]].append(r)
    lo_errs, hi_errs = [], []
    for s, g in by_spk_recs.items():
        if len(g) < 8:
            continue
        med = statistics.median(x["snr_db"] for x in g)
        lo = [x for x in g if x["snr_db"] <= med]
        hi = [x for x in g if x["snr_db"] > med]
        if lo and hi:
            lo_errs.append(1 - sum(x["correct"] for x in lo) / len(lo))
            hi_errs.append(1 - sum(x["correct"] for x in hi) / len(hi))
    within_snr_effect_pp = (statistics.mean(lo_errs) - statistics.mean(hi_errs)) * 100 if lo_errs else 0.0

    summary = {
        "overall_intent_acc": overall_acc,
        "n_test": len(per_utt),
        "per_speaker_error_rate": {str(k): v for k, v in sorted(spk_err.items())},
        "speaker_error_spread": {
            "min": min(spk_err_vals), "max": max(spk_err_vals),
            "range": max(spk_err_vals) - min(spk_err_vals),
            "std": statistics.pstdev(spk_err_vals),
        },
        "per_snr_quartile_error_rate": snr_err,
        "snr_quartile_boundaries_db": {"q25": q[0], "q50": q[1], "q75": q[2]},
        "snr_error_spread": {
            "min": min(snr_err_vals), "max": max(snr_err_vals),
            "range": max(snr_err_vals) - min(snr_err_vals),
            "std": statistics.pstdev(snr_err_vals),
        },
        "eta_squared_speaker": eta_speaker,
        "eta_squared_snr": eta_snr,
        "train_source": args.train_source,
        "within_speaker_snr_effect_pp": within_snr_effect_pp,
        "within_speaker_low_snr_err": statistics.mean(lo_errs) if lo_errs else None,
        "within_speaker_high_snr_err": statistics.mean(hi_errs) if hi_errs else None,
    }
    out = os.path.join(REPO, f"phase3/results/skit_s2i_identity_diagnostic{args.out_suffix}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_utt": per_utt}, f, indent=2)

    # Print
    print("\n=== (A) Error rate per speaker ===")
    for s in sorted(spk_err):
        n = len(by_spk[s])
        print(f"  speaker {s:>2}: err={spk_err[s]*100:5.1f}%  (n={n})")
    print(f"  spread: {min(spk_err_vals)*100:.1f}% .. {max(spk_err_vals)*100:.1f}%  "
          f"range={summary['speaker_error_spread']['range']*100:.1f}pp  std={summary['speaker_error_spread']['std']*100:.1f}pp")

    print("\n=== (B) Error rate per SNR quartile ===")
    for b in ["Q1_low_snr", "Q2", "Q3", "Q4_high_snr"]:
        if b in snr_err:
            print(f"  {b:>12}: err={snr_err[b]*100:5.1f}%  (n={len(by_snr[b])})")
    print(f"  spread: range={summary['snr_error_spread']['range']*100:.1f}pp  std={summary['snr_error_spread']['std']*100:.1f}pp")

    print("\n=== (C) Variance explained (eta^2) ===")
    print(f"  by SPEAKER identity: {eta_speaker:.4f}")
    print(f"  by SNR channel     : {eta_snr:.4f}")
    print(f"  ratio speaker/snr  : {eta_speaker/eta_snr:.1f}x" if eta_snr > 0 else "  (snr eta ~0)")

    print("\n=== (D) WITHIN-speaker SNR effect (confound-controlled) ===")
    if lo_errs:
        print(f"  low-SNR half error:  {statistics.mean(lo_errs)*100:.1f}%")
        print(f"  high-SNR half error: {statistics.mean(hi_errs)*100:.1f}%")
    print(f"  within-speaker SNR effect: {within_snr_effect_pp:+.1f}pp  (train_source={args.train_source})")
    print(f"\nSaved: {out}  ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
