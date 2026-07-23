#!/usr/bin/env python3
"""
Clone all 1,765 synthetic English SNIPS utterances with F5-TTS, using the 51
real-speaker references built by pick_references_snips.py.

Key differences from the Skit-S2I F5 run:
  * Output 16 kHz (SNIPS close-field), NOT 8 kHz.
  * NO telephony post-process — SNIPS real audio is clean close-field, so
    matching the corpus means staying clean.
  * Filenames mirror the existing MMS-TTS synth audio (cmd_XXXX_en.wav) so the
    dataset is a drop-in for phase3/snips_5fold_whisper.py's load_synth_snips()
    and for snips_audio_transfer.py via --synth-audio-dir.

Reference assignment: ref_id = idx % n_refs  (balanced; ~34.6 utts per voice).

Memory hygiene: F5-TTS leaks across long runs, so the model is reloaded every
CHUNK_SIZE utterances. Resume-safe: re-running skips wavs already on disk.

Outputs:
  data/snips_f5_cloned_audio/cmd_XXXX_en.wav
  data/snips_f5_cloned_audio/manifest.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Bypass torchcodec/ffmpeg (not installed system-wide)
import torchaudio
def _sf_load(path, *args, **kwargs):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav.copy()), sr
torchaudio.load = _sf_load

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(_HERE))
REFS_MANIFEST = os.path.join(_HERE, "references/manifest.json")
SEQ_IN = os.path.join(REPO, "data/snips_multilingual_pipeline/processed_data/en/all/seq.in")
LABEL = os.path.join(REPO, "data/snips_multilingual_pipeline/processed_data/en/all/label")
DEFAULT_OUT = os.path.join(REPO, "data/snips_f5_cloned_audio")

TARGET_SR = 16000
CHUNK_SIZE = 600


def load_refs():
    m = json.load(open(REFS_MANIFEST, encoding="utf-8"))
    return [(os.path.join(REPO, r["ref_wav"]), r["ref_text"]) for r in m]


def load_texts():
    seq = [l.strip() for l in open(SEQ_IN, encoding="utf-8") if l.strip()]
    lab = [l.strip() for l in open(LABEL, encoding="utf-8") if l.strip()]
    assert len(seq) == len(lab), f"{len(seq)} != {len(lab)}"
    return seq, lab


def load_model_fresh():
    from f5_tts.api import F5TTS
    print("Loading F5-TTS model...", flush=True)
    return F5TTS()


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def synth_one(model, ref_wav, ref_text, gen_text, seed):
    out = model.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=gen_text, seed=seed)
    if isinstance(out, tuple):
        wav, sr = out[0], out[1]
    else:
        wav = out["wav"] if isinstance(out, dict) else out
        sr = getattr(model, "target_sample_rate", 24000)
    return np.asarray(wav, dtype=np.float32).squeeze(), int(sr)


def to_16k(wav, sr):
    if sr == TARGET_SR:
        return wav
    import librosa
    return librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--n-total", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    refs = load_refs()
    seq, lab = load_texts()
    n_total = args.n_total or len(seq)
    print(f"refs: {len(refs)}   utterances: {n_total}   -> {n_total/len(refs):.1f} utts/voice")

    def fname(i):
        return f"cmd_{i:04d}_en.wav"

    have = {f for f in os.listdir(args.output_dir) if f.endswith(".wav")}
    missing = [i for i in range(n_total) if fname(i) not in have]
    print(f"already on disk: {len(have)}   to generate: {len(missing)}")
    if not missing:
        print("Nothing to do.")
    meta_path = os.path.join(args.output_dir, "manifest.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else []
    meta_by_id = {m["id"]: m for m in meta}

    t0 = time.time()
    model = None
    for start in range(0, len(missing), CHUNK_SIZE):
        chunk = missing[start:start + CHUNK_SIZE]
        if model is not None:
            print("Releasing F5 model...", flush=True)
            free_model(model)
        model = load_model_fresh()
        for i in tqdm(chunk, desc=f"chunk{start//CHUNK_SIZE + 1}"):
            ref_id = i % len(refs)
            ref_wav, ref_text = refs[ref_id]
            try:
                wav, sr = synth_one(model, ref_wav, ref_text, seq[i], seed=i)
                wav = to_16k(wav, sr)          # 16 kHz, NO telephony post-process
                sf.write(os.path.join(args.output_dir, fname(i)), wav, TARGET_SR,
                         subtype="PCM_16")
                meta_by_id[i] = {"id": i, "file": fname(i), "text": seq[i],
                                 "intent": lab[i], "ref_id": ref_id,
                                 "sampling_rate": TARGET_SR,
                                 "duration": len(wav) / TARGET_SR}
            except torch.cuda.OutOfMemoryError:
                print(f"\n!! OOM at idx {i}", flush=True)
                break
            except Exception as e:
                print(f"\n  Error at idx {i}: {type(e).__name__}", flush=True)
                continue
        meta = [meta_by_id[k] for k in sorted(meta_by_id)]
        json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"  saved manifest: {len(meta)} entries, elapsed {time.time()-t0:.0f}s", flush=True)

    done = len([f for f in os.listdir(args.output_dir) if f.endswith(".wav")])
    print(f"\nDone. {done}/{n_total} wavs. Elapsed {time.time()-t0:.0f}s")
    if meta:
        durs = [m["duration"] for m in meta]
        durs.sort()
        print(f"cloned duration: p50={durs[len(durs)//2]:.2f}s mean={sum(durs)/len(durs):.2f}s max={durs[-1]:.2f}s")


if __name__ == "__main__":
    main()
