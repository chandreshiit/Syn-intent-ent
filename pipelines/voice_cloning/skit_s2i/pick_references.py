#!/usr/bin/env python3
"""
Pick one ~6-second reference clip per Skit-S2I speaker (1..11) from the train
split. F5-TTS prefers reference clips in the 5-15 second range with clean
speech and a transcript. We:
  1) read data/skit_s2i_real_audio/metadata.csv,
  2) for each speaker_id, pick a candidate utt whose audio duration is closest
     to 6 seconds (cap at 12s — F5-TTS docs say <=15s is best),
  3) copy the chosen wav + write a sidecar JSON manifest with the transcript
     (the metadata.csv `template` column).

Outputs:
  phase3/f5tts_smoke/references/<speaker_id>.wav
  phase3/f5tts_smoke/references/manifest.json
"""
import csv
import json
import os
import shutil
from collections import defaultdict

import soundfile as sf

from slu_gap import paths

REPO = str(paths.REPO_ROOT)
META = os.path.join(REPO, "data/skit_s2i_real_audio/metadata.csv")
AUDIO_BASE = os.path.join(REPO, "data/skit_s2i_real_audio")
OUT_DIR = os.path.join(REPO, "phase3/f5tts_smoke/references")

TARGET_SEC = 6.0
MAX_SEC = 12.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    by_spk = defaultdict(list)
    with open(META, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "train":
                continue
            by_spk[int(row["speaker_id"])].append(row)
    print(f"Loaded {sum(len(v) for v in by_spk.values())} train rows across {len(by_spk)} speakers")

    manifest = []
    for spk_id in sorted(by_spk.keys()):
        rows = by_spk[spk_id]
        best_row, best_dur, best_diff = None, None, 1e9
        # Sample up to first 60 rows per speaker for speed; pick closest to 6s within cap
        for r in rows[:60]:
            wav_path = os.path.join(AUDIO_BASE, r["audio_path"].replace("\\", "/"))
            try:
                info = sf.info(wav_path)
                dur = info.frames / float(info.samplerate)
            except Exception:
                continue
            if dur > MAX_SEC:
                continue
            diff = abs(dur - TARGET_SEC)
            if diff < best_diff:
                best_diff, best_dur, best_row = diff, dur, r
        if best_row is None:
            print(f"  spk {spk_id}: no suitable clip found")
            continue
        src = os.path.join(AUDIO_BASE, best_row["audio_path"].replace("\\", "/"))
        dst = os.path.join(OUT_DIR, f"speaker_{spk_id:02d}.wav")
        shutil.copy2(src, dst)
        manifest.append({
            "speaker_id": spk_id,
            "ref_wav": os.path.relpath(dst, REPO).replace("\\", "/"),
            "ref_text": best_row["template"],
            "duration_sec": round(best_dur, 2),
            "source_idx": best_row["idx"],
        })
        print(f"  spk {spk_id:>2}: {best_dur:.2f}s  '{best_row['template'][:50]}'")
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(manifest)} references + manifest.json")


if __name__ == "__main__":
    main()
