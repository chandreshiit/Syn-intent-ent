#!/usr/bin/env python3
"""
Backfill missing F5-TTS audios + rebuild metadata.

Use case: the initial full F5 run crashed midway (LLVM OOM around idx 6750)
and dropped ~24 utterances mid-run due to UnicodeEncodeError on the Indian
Rupee sign (`₹`). This script:

  1. Reads the input command manifest (banking_commands_v2.json) — index N
     -> (command, intent, speaker_id derived by N % 11).
  2. Scans data/skit_s2i_synthesis_pipeline/generated_audio/ for existing wavs.
  3. Builds the full metadata.json from the manifest + which wavs actually
     exist on disk (so we never have phantom metadata entries for missing wavs).
  4. Generates the missing wavs in CHUNK_SIZE chunks, reloading the F5 model
     between chunks so memory does not creep into OOM.

Run with PYTHONIOENCODING=utf-8 so prints involving `₹` don't crash.
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

import torchaudio
def _sf_load(path, *args, **kwargs):
    wav, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if wav.ndim == 1:
        wav = wav[None, :]
    else:
        wav = wav.T
    return torch.from_numpy(wav.copy()), sr
torchaudio.load = _sf_load

from slu_gap import paths
from slu_gap.telephony import TELEPHONE_SR, apply_telephony_postprocess

REPO = str(paths.REPO_ROOT)
SYNTH_DIR = os.path.join(REPO, "data/skit_s2i_synthesis_pipeline")
DEFAULT_INPUT = os.path.join(SYNTH_DIR, "banking_commands_v2.json")
DEFAULT_OUTPUT = os.path.join(SYNTH_DIR, "generated_audio")
REFS_MANIFEST = os.path.join(REPO, "phase3/f5tts_smoke/references/manifest.json")



CHUNK_SIZE = 2000  # reload F5 between chunks to prevent memory creep


def load_refs():
    with open(REFS_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return {int(r["speaker_id"]): (os.path.join(REPO, r["ref_wav"]), r["ref_text"]) for r in manifest}


def load_model_fresh():
    from f5_tts.api import F5TTS
    print("Loading F5-TTS...")
    m = F5TTS()
    return m


def free_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def synthesize_one(model, ref_wav, ref_text, gen_text, seed):
    out = model.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=gen_text, seed=seed)
    if isinstance(out, tuple):
        wav, sr = out[0], out[1]
    else:
        wav = out["wav"] if isinstance(out, dict) else out
        sr = getattr(model, "target_sample_rate", 24000)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    return wav, int(sr)


def rebuild_metadata(input_path, output_dir, refs):
    """Build metadata.json from manifest + wavs actually on disk."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    speaker_ids = sorted(refs.keys())
    n_spk = len(speaker_ids)

    wavs_on_disk = {int(f.replace("audio_", "").replace(".wav", "")): f
                    for f in os.listdir(output_dir) if f.endswith(".wav")}
    print(f"Wavs on disk: {len(wavs_on_disk)}")

    meta = []
    for idx, entry in enumerate(data):
        if idx not in wavs_on_disk:
            continue
        fname = wavs_on_disk[idx]
        wav_path = os.path.join(output_dir, fname)
        try:
            duration = sf.info(wav_path).frames / float(sf.info(wav_path).samplerate)
        except Exception:
            duration = 0.0
        meta.append({
            "id": idx,
            "file": fname,
            "command": entry["command"],
            "intent": entry["intent"],
            "category": entry.get("category", "Banking"),
            "language": entry.get("language", "English"),
            "speaker_id": speaker_ids[idx % n_spk],
            "sampling_rate": TELEPHONE_SR,
            "duration": duration,
        })
    meta_path = os.path.join(output_dir, "audio_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Rebuilt metadata: {len(meta)} entries -> {meta_path}")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    ap.add_argument("--n-total", type=int, default=11900,
                    help="Total commands to ensure covered (default: 11900)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    refs = load_refs()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_total = min(args.n_total, len(data))
    speaker_ids = sorted(refs.keys())
    n_spk = len(speaker_ids)

    # Rebuild metadata to current disk state
    meta = rebuild_metadata(args.input, args.output_dir, refs)
    have = {m["id"] for m in meta}
    missing = sorted(i for i in range(n_total) if i not in have)
    print(f"Missing wavs to generate: {len(missing)}")
    if not missing:
        print("Nothing to do.")
        return

    meta_path = os.path.join(args.output_dir, "audio_metadata.json")
    t0 = time.time()
    chunk_start = 0
    model = None
    while chunk_start < len(missing):
        chunk_end = min(chunk_start + CHUNK_SIZE, len(missing))
        chunk = missing[chunk_start:chunk_end]
        print(f"\n=== Chunk {chunk_start//CHUNK_SIZE + 1} | {len(chunk)} items | range {chunk[0]}..{chunk[-1]} ===", flush=True)

        if model is not None:
            print("Releasing previous F5 model...", flush=True)
            free_model(model)
        model = load_model_fresh()

        for idx in tqdm(chunk, desc=f"chunk{chunk_start//CHUNK_SIZE + 1}"):
            entry = data[idx]
            spk = speaker_ids[idx % n_spk]
            ref_wav, ref_text = refs[spk]
            try:
                wav, sr = synthesize_one(model, ref_wav, ref_text, entry["command"], seed=idx)
                pcm = apply_telephony_postprocess(wav, src_sr=sr, target_sr=TELEPHONE_SR, seed=idx)
                fname = f"audio_{idx:06d}.wav"
                sf.write(os.path.join(args.output_dir, fname), pcm, TELEPHONE_SR, subtype="PCM_16")
                duration = len(pcm) / float(TELEPHONE_SR)
                meta.append({
                    "id": idx,
                    "file": fname,
                    "command": entry["command"],
                    "intent": entry["intent"],
                    "category": entry.get("category", "Banking"),
                    "language": entry.get("language", "English"),
                    "speaker_id": spk,
                    "sampling_rate": TELEPHONE_SR,
                    "duration": duration,
                })
            except torch.cuda.OutOfMemoryError as e:
                print(f"\n!! OOM at idx {idx}, will retry after model reload next chunk", flush=True)
                break
            except Exception as e:
                # log without printing the offending text (may have unicode)
                print(f"\n  Error at idx {idx}: {type(e).__name__}", flush=True)
                continue
        # Persist metadata after each chunk
        meta.sort(key=lambda m: m["id"])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  saved metadata: {len(meta)} entries. elapsed {time.time()-t0:.0f}s", flush=True)
        chunk_start = chunk_end

    # Final stats
    have = {m["id"] for m in meta}
    still_missing = [i for i in range(n_total) if i not in have]
    print(f"\nDone. Have: {len(have)}/{n_total}. Still missing: {len(still_missing)}")
    if still_missing[:20]:
        print(f"  first 20 missing: {still_missing[:20]}")


if __name__ == "__main__":
    main()
