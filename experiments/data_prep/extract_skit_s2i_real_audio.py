#!/usr/bin/env python3
"""
One-time extraction of the real Skit-S2I audio from the HuggingFace parquet
files into plain WAV files.

Writes to $SLU_GAP_DATA/skit_s2i_real_audio/ (override with --out-dir):
  - audio/{split}_{idx:05d}.wav  decoded from the parquet audio bytes
  - metadata.csv                 split, idx, intent_class, template,
                                 speaker_id, audio_path

Run this once after staging the corpus. Downstream experiments then read the
audio straight off disk instead of re-decoding parquet bytes every epoch, which
otherwise dominates the training loop.

Requires the Skit-S2I corpus; see docs/DATA.md. It is CC BY-NC 4.0.
"""

import argparse
import csv
import io
import os

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from slu_gap import paths


PARQUET_FILES = {
    "train": [
        "skit-s2i/data/train-00000-of-00002.parquet",
        "skit-s2i/data/train-00001-of-00002.parquet",
    ],
    "test": ["skit-s2i/data/test-00000-of-00001.parquet"],
}


def extract(out_dir):
    audio_dir = os.path.join(out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    metadata_csv = os.path.join(out_dir, "metadata.csv")

    with open(metadata_csv, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "split", "idx", "intent_class", "template", "speaker_id", "audio_path",
        ])
        writer.writeheader()

        for split, paths in PARQUET_FILES.items():
            for parquet_path in paths:
                print(f"Reading {parquet_path}")
                df = pd.read_parquet(parquet_path, columns=[
                    "intent_class", "template", "speaker_id", "audio",
                ])
                for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"{split}"):
                    audio_meta = row["audio"]
                    if not isinstance(audio_meta, dict) or "bytes" not in audio_meta:
                        continue
                    audio_bytes = audio_meta["bytes"]
                    # Decode + re-encode as plain wav for simplicity.
                    audio_arr, sr = sf.read(io.BytesIO(audio_bytes))
                    wav_name = f"{split}_{idx:05d}.wav"
                    wav_path = os.path.join(audio_dir, wav_name)
                    sf.write(wav_path, audio_arr, sr)
                    writer.writerow({
                        "split": split,
                        "idx": int(idx),
                        "intent_class": int(row["intent_class"]),
                        "template": str(row["template"]).strip(),
                        "speaker_id": int(row["speaker_id"]),
                        "audio_path": os.path.join("audio", wav_name),
                    })
    print(f"\nWrote metadata: {metadata_csv}")
    print(f"Wrote audio dir: {audio_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=str, default=str(paths.SKIT_S2I_REAL))
    args = parser.parse_args()
    extract(args.out_dir)


if __name__ == "__main__":
    main()
