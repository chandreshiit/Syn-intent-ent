#!/usr/bin/env python3
"""
F5-TTS regen v2: 8 references per speaker, rotated deterministically per
utterance to introduce within-speaker acoustic variation.

Reference rotation: for utterance idx with speaker_id S = speaker_ids[idx % 11],
the reference picked is refs_per_speaker[S][(idx // 11) % 8]. This ensures:
  - same (idx -> speaker, ref) deterministic mapping (reproducible)
  - each (speaker, ref) combo gets ~135 utterances across 11,900 commands
  - within a speaker, 8 ref voices alternate as we walk the index space

Outputs (drop-in replacement for parler/F5-v1 layout):
  data/skit_s2i_synthesis_pipeline/generated_audio/audio_XXXXXX.wav
  data/skit_s2i_synthesis_pipeline/generated_audio/audio_metadata.json

Resume + chunked memory hygiene identical to backfill_f5.py (reloads F5 model
every CHUNK_SIZE utts to prevent OOM creep).
"""
import argparse
import gc
import json
import os
import time
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Patch torchaudio.load to bypass torchcodec/ffmpeg
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
REFS_MANIFEST = os.path.join(REPO, "phase3/f5tts_smoke/references_v2/manifest.json")



CHUNK_SIZE = 1000  # smaller chunks: more model reloads but smaller per-cycle memory leak


def load_refs():
    """Return {speaker_id: [(ref_wav_path, ref_text), ...]} ordered by ref_idx."""
    with open(REFS_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    grouped = defaultdict(list)
    for r in manifest:
        grouped[int(r["speaker_id"])].append((int(r["ref_idx"]),
                                                os.path.join(REPO, r["ref_wav"]),
                                                r["ref_text"]))
    out = {}
    for spk, lst in grouped.items():
        lst.sort(key=lambda x: x[0])
        out[spk] = [(wav, txt) for _, wav, txt in lst]
    return out


def pick_ref_for_idx(refs, speaker_ids, idx):
    spk = speaker_ids[idx % len(speaker_ids)]
    n_refs = len(refs[spk])
    ref_idx = (idx // len(speaker_ids)) % n_refs
    ref_wav, ref_text = refs[spk][ref_idx]
    return spk, ref_idx, ref_wav, ref_text


def load_model_fresh():
    from f5_tts.api import F5TTS
    print("Loading F5-TTS model...")
    return F5TTS()


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


def rebuild_metadata(input_path, output_dir, refs, speaker_ids):
    """Build metadata.json from manifest + wavs actually on disk."""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
        spk, ref_idx, _, _ = pick_ref_for_idx(refs, speaker_ids, idx)
        meta.append({
            "id": idx, "file": fname,
            "command": entry["command"], "intent": entry["intent"],
            "category": entry.get("category", "Banking"),
            "language": entry.get("language", "English"),
            "speaker_id": spk, "ref_idx": ref_idx,
            "sampling_rate": TELEPHONE_SR, "duration": duration,
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
    ap.add_argument("--n-total", type=int, default=11900)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    refs = load_refs()
    speaker_ids = sorted(refs.keys())
    print(f"Speakers: {speaker_ids} | refs per speaker: {[len(refs[s]) for s in speaker_ids]}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    n_total = min(args.n_total, len(data))

    meta = rebuild_metadata(args.input, args.output_dir, refs, speaker_ids)
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
            spk, ref_idx, ref_wav, ref_text = pick_ref_for_idx(refs, speaker_ids, idx)
            try:
                wav, sr = synthesize_one(model, ref_wav, ref_text, entry["command"], seed=idx)
                pcm = apply_telephony_postprocess(wav, src_sr=sr, target_sr=TELEPHONE_SR, seed=idx)
                fname = f"audio_{idx:06d}.wav"
                sf.write(os.path.join(args.output_dir, fname), pcm, TELEPHONE_SR, subtype="PCM_16")
                duration = len(pcm) / float(TELEPHONE_SR)
                meta.append({
                    "id": idx, "file": fname,
                    "command": entry["command"], "intent": entry["intent"],
                    "category": entry.get("category", "Banking"),
                    "language": entry.get("language", "English"),
                    "speaker_id": spk, "ref_idx": ref_idx,
                    "sampling_rate": TELEPHONE_SR, "duration": duration,
                })
            except torch.cuda.OutOfMemoryError:
                print(f"\n!! OOM at idx {idx}, will retry after model reload next chunk", flush=True)
                break
            except Exception as e:
                print(f"\n  Error at idx {idx}: {type(e).__name__}", flush=True)
                continue
        meta.sort(key=lambda m: m["id"])
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  saved metadata: {len(meta)} entries. elapsed {time.time()-t0:.0f}s", flush=True)
        chunk_start = chunk_end

    have = {m["id"] for m in meta}
    still_missing = [i for i in range(n_total) if i not in have]
    print(f"\nDone. Have: {len(have)}/{n_total}. Still missing: {len(still_missing)}")


if __name__ == "__main__":
    main()
