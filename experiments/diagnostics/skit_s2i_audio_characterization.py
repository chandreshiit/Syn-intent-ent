#!/usr/bin/env python3
"""
Characterize real Skit-S2I audio vs current synthetic Skit-S2I audio, so we
can identify what changes the synthesis pipeline needs to match real
telephony characteristics.

Samples N audios from each side (real, synth), measures:
  - sample rate, channels, bit depth
  - duration distribution
  - RMS amplitude (overall loudness)
  - noise floor (RMS of quietest 100ms window)
  - SNR estimate (loudness above noise floor)
  - spectral centroid (rough brightness)
  - band-energy ratio (low <300 Hz, mid 300-3400 Hz, high >3400 Hz)
  - dynamic range

Outputs:
  analysis/skit_s2i_audio_characterization.json
  analysis/skit_s2i_audio_characterization.md
"""
import argparse
import json
import os
import random
import statistics

import numpy as np
import soundfile as sf


def measure(wav_path):
    """Return dict of acoustic measurements for one wav."""
    try:
        audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    except Exception as e:
        return {"error": str(e), "path": wav_path}

    info = sf.info(wav_path)
    n_channels = info.channels
    bit_depth = info.subtype

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    n_samples = len(audio)
    duration = n_samples / sr if sr > 0 else 0.0

    if n_samples == 0:
        return {"path": wav_path, "sr": sr, "channels": n_channels, "duration": 0.0, "empty": True}

    # Overall RMS (use abs)
    rms = float(np.sqrt(np.mean(audio ** 2) + 1e-12))

    # Noise floor: minimum RMS over 100ms windows
    win = int(0.1 * sr)
    if win > 0 and n_samples >= win * 2:
        n_win = n_samples // win
        win_rms = np.array([
            np.sqrt(np.mean(audio[i * win:(i + 1) * win] ** 2) + 1e-12)
            for i in range(n_win)
        ])
        noise_floor = float(np.percentile(win_rms, 10))
        speech_rms = float(np.percentile(win_rms, 90))
    else:
        noise_floor = rms
        speech_rms = rms

    snr_db = 20 * np.log10((speech_rms + 1e-9) / (noise_floor + 1e-9))

    # Spectrum via FFT (windowed)
    if n_samples >= 1024:
        fft_size = min(8192, 2 ** int(np.log2(n_samples)))
        windowed = audio[:fft_size] * np.hanning(fft_size)
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)
        if spec.sum() > 0:
            centroid = float((freqs * spec).sum() / spec.sum())
        else:
            centroid = 0.0
        # Band energies
        low_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs <= 3400)
        high_mask = freqs > 3400
        total_e = (spec ** 2).sum() + 1e-12
        e_low = float(((spec ** 2)[low_mask].sum() / total_e))
        e_mid = float(((spec ** 2)[mid_mask].sum() / total_e))
        e_high = float(((spec ** 2)[high_mask].sum() / total_e))
    else:
        centroid = 0.0
        e_low = e_mid = e_high = 0.0

    # Dynamic range = peak/RMS in dB (crest factor)
    peak = float(np.abs(audio).max())
    crest_db = 20 * np.log10((peak + 1e-9) / (rms + 1e-9))

    return {
        "path": wav_path,
        "sr": int(sr),
        "channels": n_channels,
        "bit_depth": bit_depth,
        "duration": duration,
        "n_samples": n_samples,
        "rms": rms,
        "noise_floor_rms": noise_floor,
        "speech_rms": speech_rms,
        "snr_db": float(snr_db),
        "spectral_centroid_hz": centroid,
        "band_energy_lt_300hz": e_low,
        "band_energy_300_3400hz": e_mid,
        "band_energy_gt_3400hz": e_high,
        "crest_db": float(crest_db),
        "peak": peak,
    }


def aggregate(measurements):
    """Compute median, p10, p90 for each numeric metric."""
    keys = ["sr", "duration", "rms", "noise_floor_rms", "speech_rms", "snr_db",
            "spectral_centroid_hz", "band_energy_lt_300hz", "band_energy_300_3400hz",
            "band_energy_gt_3400hz", "crest_db", "peak"]
    out = {}
    for k in keys:
        vals = [m[k] for m in measurements if k in m and isinstance(m[k], (int, float))]
        if vals:
            out[k] = {
                "median": statistics.median(vals),
                "p10": np.percentile(vals, 10),
                "p90": np.percentile(vals, 90),
                "n": len(vals),
            }
    # Sample rate counts (categorical)
    srs = [m["sr"] for m in measurements if "sr" in m]
    out["sr_distribution"] = dict(
        (str(s), srs.count(s)) for s in sorted(set(srs))
    )
    # Channels
    chs = [m.get("channels", 1) for m in measurements]
    out["channels_distribution"] = dict(
        (str(c), chs.count(c)) for c in sorted(set(chs))
    )
    # Bit depths
    bds = [m.get("bit_depth", "?") for m in measurements]
    out["bit_depth_distribution"] = dict(
        (str(b), bds.count(b)) for b in sorted(set(bds))
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-dir", default="data/skit_s2i_real_audio/audio")
    ap.add_argument("--synth-dir", default="data/skit_s2i_synthesis_pipeline/generated_audio")
    ap.add_argument("--n-sample", type=int, default=100,
                    help="How many wavs to sample from each side.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/skit_s2i_audio_characterization.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    real_files = [os.path.join(args.real_dir, f) for f in os.listdir(args.real_dir) if f.endswith(".wav")]
    synth_files = [os.path.join(args.synth_dir, f) for f in os.listdir(args.synth_dir) if f.endswith(".wav")]
    print(f"Real: {len(real_files)} wavs total")
    print(f"Synth: {len(synth_files)} wavs total")

    rng.shuffle(real_files)
    rng.shuffle(synth_files)
    real_sample = real_files[:args.n_sample]
    synth_sample = synth_files[:args.n_sample]

    print(f"\nMeasuring {len(real_sample)} real + {len(synth_sample)} synth...")
    real_meas = [measure(p) for p in real_sample]
    synth_meas = [measure(p) for p in synth_sample]

    real_agg = aggregate([m for m in real_meas if "error" not in m])
    synth_agg = aggregate([m for m in synth_meas if "error" not in m])

    summary = {
        "n_sampled_real": len(real_sample),
        "n_sampled_synth": len(synth_sample),
        "real": real_agg,
        "synth": synth_agg,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {args.out}")

    # Print comparison table
    print("\n=== Comparison (median values) ===")
    print(f"{'Metric':<35} {'Real':>14} {'Synth':>14} {'Δ':>10}")
    print("-" * 80)
    for k in ["sr", "duration", "rms", "noise_floor_rms", "speech_rms", "snr_db",
              "spectral_centroid_hz", "band_energy_lt_300hz",
              "band_energy_300_3400hz", "band_energy_gt_3400hz",
              "crest_db", "peak"]:
        if k in real_agg and k in synth_agg:
            r = real_agg[k]["median"]
            s = synth_agg[k]["median"]
            try:
                diff = f"{s-r:+.4f}"
            except TypeError:
                diff = "?"
            print(f"{k:<35} {r:>14.4f} {s:>14.4f} {diff:>10}")
    print(f"\nReal SR distribution:  {real_agg.get('sr_distribution')}")
    print(f"Synth SR distribution: {synth_agg.get('sr_distribution')}")
    print(f"Real bit depth: {real_agg.get('bit_depth_distribution')}")
    print(f"Synth bit depth: {synth_agg.get('bit_depth_distribution')}")
    print(f"Real channels: {real_agg.get('channels_distribution')}")
    print(f"Synth channels: {synth_agg.get('channels_distribution')}")


if __name__ == "__main__":
    main()
