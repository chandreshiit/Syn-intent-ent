#!/usr/bin/env python3
"""
Full F5-TTS regen for Skit-S2I: generate all 11,900 commands using F5-TTS
voice-cloned from real Skit-S2I speaker references, then apply our calibrated
telephony post-process (same chain used for parler-v3) so the channel
distribution matches.

Outputs (drop-in replacement for parler-v3 layout):
  data/skit_s2i_synthesis_pipeline/generated_audio/audio_XXXXXX.wav
  data/skit_s2i_synthesis_pipeline/generated_audio/audio_metadata.json

Resume: pass --start-index N to continue from index N (loads existing metadata
when N > 0, matching synthesize_speech.py's contract).
"""
import argparse
import json
import os
import time

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Patch torchaudio.load to bypass torchcodec/ffmpeg (FFmpeg DLLs not installed).
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




def load_refs():
    """Return dict: speaker_id (int) -> (ref_wav_path, ref_text)."""
    with open(REFS_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    out = {}
    for r in manifest:
        out[int(r["speaker_id"])] = (
            os.path.join(REPO, r["ref_wav"]),
            r["ref_text"],
        )
    return out


def load_f5_model():
    from f5_tts.api import F5TTS
    print("Loading F5-TTS model...")
    return F5TTS()


def synthesize_one(model, ref_wav, ref_text, gen_text, seed):
    out = model.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=gen_text, seed=seed)
    if isinstance(out, tuple):
        wav, sr = out[0], out[1]
    else:
        wav = out["wav"] if isinstance(out, dict) else out
        sr = getattr(model, "target_sample_rate", 24000)
    wav = np.asarray(wav, dtype=np.float32).squeeze()
    return wav, int(sr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, default=DEFAULT_INPUT)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--end-index", type=int, default=None)
    ap.add_argument("--num-speakers", type=int, default=11)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} commands")

    refs = load_refs()
    speaker_ids = sorted(refs.keys())[:args.num_speakers]
    print(f"Using {len(speaker_ids)} speakers: {speaker_ids}")

    model = load_f5_model()

    start = args.start_index
    end = args.end_index if args.end_index is not None else len(data)
    print(f"Generating commands {start} to {end}")

    metadata = []
    meta_path = os.path.join(args.output_dir, "audio_metadata.json")
    if os.path.exists(meta_path) and start > 0:
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        print(f"Loaded {len(metadata)} existing metadata entries")

    t0 = time.time()
    for idx in tqdm(range(start, end), desc="F5-Synth"):
        entry = data[idx]
        spk_id = speaker_ids[idx % len(speaker_ids)]
        ref_wav, ref_text = refs[spk_id]
        try:
            wav, sr = synthesize_one(model, ref_wav, ref_text, entry["command"], seed=idx)
            pcm = apply_telephony_postprocess(wav, src_sr=sr, target_sr=TELEPHONE_SR, seed=idx)
            fname = f"audio_{idx:06d}.wav"
            sf.write(os.path.join(args.output_dir, fname), pcm, TELEPHONE_SR, subtype="PCM_16")
            duration = len(pcm) / float(TELEPHONE_SR)
            metadata.append({
                "id": idx,
                "file": fname,
                "command": entry["command"],
                "intent": entry["intent"],
                "category": entry.get("category", "Banking"),
                "language": entry.get("language", "English"),
                "speaker_id": spk_id,
                "sampling_rate": TELEPHONE_SR,
                "duration": duration,
            })
            if (idx + 1) % 100 == 0:
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n!! OOM at idx {idx}: {e}", flush=True)
            raise
        except Exception as e:
            print(f"\n  Error at idx {idx}: {type(e).__name__}: {e}", flush=True)
            # continue on transient errors
            continue

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"\nDone. {len(metadata)} entries in metadata. Elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
