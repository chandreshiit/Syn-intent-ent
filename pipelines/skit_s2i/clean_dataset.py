#!/usr/bin/env python3
"""
Drop manifest rows whose audio file is missing.

TTS generation over ~12k utterances occasionally fails on individual items
(model reloads, encoding errors). This filters the manifest down to rows whose
wav actually exists on disk, so training does not die partway through an epoch.

Path resolution mirrors the training data loader: the manifest's `audio_path`
may be absolute or relative, but only its basename is used, resolved against
`--wav-root`.

Usage:
    python pipelines/skit_s2i/clean_dataset.py \
        --input  $SLU_GAP_DATA/skit_s2i_synthesis_pipeline/output/test_fixed.csv \
        --output $SLU_GAP_DATA/skit_s2i_synthesis_pipeline/output/test_verified.csv
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from slu_gap import paths

DEFAULT_ROOT = paths.SKIT_S2I_SYNTH


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=str(DEFAULT_ROOT / "output" / "test_fixed.csv"))
    parser.add_argument("--output", default=str(DEFAULT_ROOT / "output" / "test_verified.csv"))
    parser.add_argument("--wav-root", default=str(DEFAULT_ROOT / "generated_audio"))
    args = parser.parse_args()

    print(f"Checking files listed in {args.input}")
    df = pd.read_csv(args.input)

    valid_rows = []
    missing = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        raw_path = str(row["audio_path"]).strip().replace("\\", "/")
        filename = os.path.basename(raw_path.rstrip("/"))
        if os.path.exists(os.path.join(args.wav_root, filename)):
            valid_rows.append(row)
        else:
            missing += 1
            if missing <= 5:
                print(f"  missing: {filename}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    pd.DataFrame(valid_rows).to_csv(args.output, index=False)

    print(f"\nOriginal rows: {len(df)}")
    print(f"Kept rows:     {len(valid_rows)}")
    print(f"Dropped:       {missing} (audio file not found)")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
