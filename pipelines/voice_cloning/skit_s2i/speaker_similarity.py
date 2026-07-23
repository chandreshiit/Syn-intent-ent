#!/usr/bin/env python3
"""
Speaker-embedding similarity: how close is a synthesized clip's voice to the
real speaker's voice?

For each (speaker, command) in the F5 smoke set:
  1) compute embedding of the F5 telephony output
  2) compute embedding of the parler-v3 telephony output for the same
     speaker (closest matching command from parler manifest, or any clip
     from that speaker if no command match)
  3) compute a centroid embedding from N=10 random real clips of that speaker
  4) report cosine(F5, real_centroid) and cosine(parler, real_centroid)

Uses Resemblyzer (lightweight VoiceEncoder, ~30 MB). Falls back to
SpeechBrain ECAPA-TDNN if Resemblyzer is unavailable.

Outputs:
  phase3/f5tts_smoke/speaker_similarity.json
  printed table to stdout
"""
import csv
import json
import os
import random
from collections import defaultdict

import numpy as np

from slu_gap import paths

REPO = str(paths.REPO_ROOT)
SMOKE_MANIFEST = os.path.join(REPO, "phase3/f5tts_smoke/manifest.json")
PARLER_META = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline/generated_audio/audio_metadata.json")
PARLER_DIR = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline/generated_audio")
REAL_META = os.path.join(REPO, "data/skit_s2i_real_audio/metadata.csv")
REAL_BASE = os.path.join(REPO, "data/skit_s2i_real_audio")
OUT_JSON = os.path.join(REPO, "phase3/f5tts_smoke/speaker_similarity.json")

N_REAL_PER_SPEAKER = 10


def load_encoder():
    """Try Resemblyzer first; fall back to SpeechBrain."""
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
        enc = VoiceEncoder()
        print("Using Resemblyzer VoiceEncoder")
        def embed(path):
            wav = preprocess_wav(path)
            return enc.embed_utterance(wav)
        return embed, "resemblyzer"
    except Exception as e:
        print(f"Resemblyzer unavailable ({e!r}); trying SpeechBrain ECAPA-TDNN")
        from speechbrain.inference.speaker import EncoderClassifier
        clf = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=os.path.join(REPO, ".torch_cache/spkrec-ecapa-voxceleb"),
        )
        import torchaudio
        def embed(path):
            wav, sr = torchaudio.load(path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
            emb = clf.encode_batch(wav).squeeze().detach().cpu().numpy()
            return emb
        return embed, "speechbrain-ecapa"


def cos(a, b):
    a = np.asarray(a).reshape(-1); b = np.asarray(b).reshape(-1)
    na = np.linalg.norm(a) + 1e-9; nb = np.linalg.norm(b) + 1e-9
    return float(np.dot(a, b) / (na * nb))


def main():
    with open(SMOKE_MANIFEST, "r", encoding="utf-8") as f:
        smoke = json.load(f)
    with open(PARLER_META, "r", encoding="utf-8") as f:
        parler = json.load(f)
    parler_by_spk = defaultdict(list)
    for p in parler:
        parler_by_spk[p["speaker_id"]].append(p)
    real_by_spk = defaultdict(list)
    with open(REAL_META, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            real_by_spk[int(r["speaker_id"])].append(
                os.path.join(REAL_BASE, r["audio_path"].replace("\\", "/")))

    embed, backend = load_encoder()

    rng = random.Random(0)
    # Build per-speaker real centroid + sanity check (real-vs-real within-speaker similarity)
    spk_centroid = {}
    spk_self_cos = {}
    for spk_id, paths in real_by_spk.items():
        sample = rng.sample(paths, min(N_REAL_PER_SPEAKER, len(paths)))
        embs = [embed(p) for p in sample]
        spk_centroid[spk_id] = np.mean(embs, axis=0)
        # self-cos: cos(each clip, centroid_of_other_clips) — measures within-speaker consistency
        sub = []
        for i, e in enumerate(embs):
            other = np.mean([embs[j] for j in range(len(embs)) if j != i], axis=0)
            sub.append(cos(e, other))
        spk_self_cos[spk_id] = float(np.mean(sub))
        print(f"  spk {spk_id:>2}: real-centroid from {len(embs)} clips, within-speaker mean cos = {spk_self_cos[spk_id]:.3f}")

    # Also build a "different speaker" baseline: cos(spk A centroid, spk B centroid) averaged across pairs
    keys = sorted(spk_centroid.keys())
    cross_cos = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            cross_cos.append(cos(spk_centroid[keys[i]], spk_centroid[keys[j]]))
    cross_mean = float(np.mean(cross_cos))
    print(f"\n  Real-vs-other-speaker mean cos = {cross_mean:.3f} (lower = better separation)")
    print(f"  Real within-speaker mean cos = {np.mean(list(spk_self_cos.values())):.3f} (higher = tighter cluster)\n")

    # Now compare F5 + parler against each speaker's real centroid
    rows = []
    for s in smoke:
        spk = s["speaker_id"]
        f5_path = os.path.join(REPO, s["tele_wav"])
        f5_emb = embed(f5_path)
        f5_cos = cos(f5_emb, spk_centroid[spk])

        # Pick a parler clip from the same speaker (random for now — text content
        # only weakly conditions speaker identity in parler since voices are
        # text-prompted, not cloned)
        parler_clips = parler_by_spk[spk]
        if parler_clips:
            p = rng.choice(parler_clips)
            p_path = os.path.join(PARLER_DIR, p["file"])
            p_emb = embed(p_path)
            p_cos = cos(p_emb, spk_centroid[spk])
        else:
            p_path, p_cos = None, None

        rows.append({
            "speaker_id": spk,
            "cmd_idx": s["cmd_idx"],
            "f5_cos_to_real": f5_cos,
            "parler_cos_to_real": p_cos,
            "parler_clip": os.path.basename(p_path) if p_path else None,
            "within_speaker_baseline": spk_self_cos[spk],
            "delta_f5_vs_parler": (f5_cos - p_cos) if p_cos is not None else None,
        })

    # Summary
    f5_mean = float(np.mean([r["f5_cos_to_real"] for r in rows]))
    p_mean = float(np.mean([r["parler_cos_to_real"] for r in rows if r["parler_cos_to_real"] is not None]))
    self_mean = float(np.mean(list(spk_self_cos.values())))

    print(f"{'spk':>3} {'cmd':>3} {'F5-vs-real':>10} {'parler-vs-real':>14} {'within-spk':>11} {'Δ(F5-parler)':>13}")
    print("-" * 65)
    for r in rows:
        print(f"{r['speaker_id']:>3} {r['cmd_idx']:>3} {r['f5_cos_to_real']:>10.3f} {r['parler_cos_to_real']:>14.3f} {r['within_speaker_baseline']:>11.3f} {r['delta_f5_vs_parler']:>13.3f}")
    print("-" * 65)
    print(f"{'mean':>3} {'':>3} {f5_mean:>10.3f} {p_mean:>14.3f} {self_mean:>11.3f} {f5_mean - p_mean:>13.3f}")
    print(f"\nReal-vs-other-speaker baseline: {cross_mean:.3f} (anything close to this = no speaker match)")

    summary = {
        "backend": backend,
        "n_real_per_speaker": N_REAL_PER_SPEAKER,
        "real_within_speaker_mean_cos": self_mean,
        "real_cross_speaker_mean_cos": cross_mean,
        "f5_mean_cos_to_real": f5_mean,
        "parler_mean_cos_to_real": p_mean,
        "f5_minus_parler": f5_mean - p_mean,
        "per_clip": rows,
        "per_speaker_within_cos": spk_self_cos,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
