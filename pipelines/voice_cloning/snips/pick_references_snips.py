#!/usr/bin/env python3
"""
Build F5-TTS voice references for SNIPS from REAL close-field audio.

Design (mirrors phase3/f5tts_smoke/pick_references.py for Skit-S2I, adapted to
SNIPS's crowdsourced corpus):

  * SNIPS *does* carry speaker identity: speech_corpus/metadata.json has
    worker.id (UUID) per clip. So we take the speaker-ID branch: one reference
    per worker (51 usable workers), rather than the embedding-clustering
    fallback.
  * WRINKLE: no SNIPS clip is ~6 s (p50 = 2.8 s, max 5.0 s). We therefore build
    each ~6 s reference by CONCATENATING that worker's own clips (same voice)
    with a short silence between, until we reach TARGET_SEC (cap MAX_SEC).
    ref_text is the concatenation of the corresponding transcripts.
  * LEAKAGE GUARD: references are drawn ONLY from the real *train* split, using
    the exact same stratified 80/20 split (seed 42) as
    phase3/snips_audio_transfer.py. No test audio is ever touched.
  * Only unlabeled reference *audio* + its transcript is used — no intent/slot
    annotation. (Transcripts are corpus metadata, not SLU labels; same as the
    Skit-S2I reference setup.)

Outputs:
  phase3/snips_f5/references/spk_XX.wav        (16 kHz mono)
  phase3/snips_f5/references/manifest.json
"""
import json
import os
from collections import defaultdict

import numpy as np
import soundfile as sf

from sklearn.model_selection import train_test_split

from slu_gap import paths
from slu_gap.datasets import load_real_snips_close

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(paths.REPO_ROOT)

SNIPS_META = os.path.join(REPO, "SNIPS/smart-lights-en-close-field/speech_corpus/metadata.json")
OUT_DIR = os.path.join(_HERE, "references")
TARGET_SEC = 6.0     # aim for ~6 s, as on Skit-S2I
MAX_SEC = 9.0        # hard cap
MIN_SEC = 5.0        # skip workers that can't reach this
GAP_MS = 150         # silence inserted between concatenated clips
TARGET_SR = 16000
SEED = 42
REF_PEAK = 0.8       # peak-normalize refs (median worker peak was 0.815)


def load_meta():
    m = json.load(open(SNIPS_META, encoding="utf-8"))
    fn2worker, fn2text = {}, {}
    for v in m.values():
        fn = v["filename"]
        fn2worker[fn] = v["worker"]["id"]
        fn2text[fn] = (v.get("text") or v.get("sentence") or "").strip()
    return fn2worker, fn2text


def read_16k(path):
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return audio.astype(np.float32)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fn2worker, fn2text = load_meta()

    real_paths, real_intents = load_real_snips_close()
    items = list(zip(real_paths, real_intents))
    # EXACT same split as snips_audio_transfer.py -> refs come from train only
    real_train, _real_test = train_test_split(
        items, test_size=0.20, stratify=[x[1] for x in items], random_state=SEED)
    print(f"real_train={len(real_train)} (references drawn ONLY from here)")

    by_worker = defaultdict(list)
    for p, _ in real_train:
        by_worker[fn2worker[os.path.basename(p)]].append(p)
    # deterministic order
    for w in by_worker:
        by_worker[w].sort()

    gap = np.zeros(int(GAP_MS * TARGET_SR / 1000), dtype=np.float32)
    manifest = []
    skipped = []
    for spk_idx, worker in enumerate(sorted(by_worker.keys())):
        clips = by_worker[worker]
        chunks, texts, total = [], [], 0.0
        for p in clips:
            a = read_16k(p)
            d = len(a) / TARGET_SR
            if total + d > MAX_SEC and total >= MIN_SEC:
                break
            chunks.append(a)
            texts.append(fn2text[os.path.basename(p)])
            total += d + GAP_MS / 1000.0
            if total >= TARGET_SEC:
                break
        if total < MIN_SEC:
            skipped.append((worker, round(total, 2), len(clips)))
            continue

        # interleave with silence
        out = chunks[0]
        for c in chunks[1:]:
            out = np.concatenate([out, gap, c])
        dur = len(out) / TARGET_SR

        # Peak-normalize. One SNIPS worker recorded at very low gain (peak 0.027,
        # vs median 0.815 across workers); F5-TTS trims silence from the reference
        # and ends up with an empty array -> librosa ParameterError. Gain does not
        # change timbre/speaker identity, so normalizing is safe and necessary.
        peak = float(np.abs(out).max())
        if peak > 1e-4:
            out = (out * (REF_PEAK / peak)).astype(np.float32)

        name = f"spk_{spk_idx:02d}.wav"
        dst = os.path.join(OUT_DIR, name)
        sf.write(dst, out, TARGET_SR, subtype="PCM_16")
        ref_text = " ".join(t.rstrip(".") + "." for t in texts if t)
        manifest.append({
            "ref_id": len(manifest),
            "worker_id": worker,
            "ref_wav": os.path.relpath(dst, REPO).replace("\\", "/"),
            "ref_text": ref_text,
            "duration_sec": round(dur, 2),
            "n_clips_concat": len(chunks),
            "n_clips_available": len(clips),
        })
        print(f"  ref {len(manifest)-1:>2} worker={worker[:8]} dur={dur:.2f}s "
              f"({len(chunks)} clips) '{ref_text[:52]}'")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    durs = [m["duration_sec"] for m in manifest]
    print(f"\nWrote {len(manifest)} references -> {OUT_DIR}")
    print(f"  duration: min={min(durs):.2f}s mean={sum(durs)/len(durs):.2f}s max={max(durs):.2f}s")
    if skipped:
        print(f"  skipped {len(skipped)} worker(s) with <{MIN_SEC}s audio: {skipped}")
    print(f"  utts per reference if 1765 balanced: {1765/len(manifest):.1f}")


if __name__ == "__main__":
    main()
