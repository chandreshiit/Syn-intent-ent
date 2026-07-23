#!/usr/bin/env python3
"""
Pick 8 reference clips per Skit-S2I speaker (1..11) from the train split,
covering different intent contexts and a range of durations (5-12s preferred,
6-10s ideal). Goal: capture within-speaker variation so F5-TTS clones get more
acoustic diversity than the single-ref V1 setup.

Selection per speaker:
  1) Filter train rows to those with audio duration in [5, 12] seconds.
  2) Group by intent_class.
  3) Greedily pick one clip from each intent class until we have 8; if a
     speaker has fewer than 8 intent classes, fill remaining slots with the
     longest unused clips.
  4) Sort the final 8 by intent_class for stable ordering.

Outputs:
  phase3/f5tts_smoke/references_v2/speaker_<id>_ref_<k>.wav  (k = 0..7)
  phase3/f5tts_smoke/references_v2/manifest.json
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
OUT_DIR = os.path.join(REPO, "phase3/f5tts_smoke/references_v2")

N_REFS_PER_SPEAKER = 8
DUR_MIN = 5.0
DUR_MAX = 12.0
SCAN_CAP = 400  # scan up to this many rows per speaker for duration filtering


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
        # Annotate duration + intent for the first SCAN_CAP rows
        candidates = []
        for r in rows[:SCAN_CAP]:
            wav_path = os.path.join(AUDIO_BASE, r["audio_path"].replace("\\", "/"))
            try:
                info = sf.info(wav_path)
                dur = info.frames / float(info.samplerate)
            except Exception:
                continue
            if DUR_MIN <= dur <= DUR_MAX:
                candidates.append({"row": r, "dur": dur, "intent": int(r["intent_class"])})
        # Sort by intent then descending duration so each intent's longest comes first
        candidates.sort(key=lambda c: (c["intent"], -c["dur"]))
        # Greedy: one from each new intent until we have N_REFS_PER_SPEAKER
        chosen = []
        seen_intents = set()
        for c in candidates:
            if c["intent"] in seen_intents:
                continue
            chosen.append(c)
            seen_intents.add(c["intent"])
            if len(chosen) >= N_REFS_PER_SPEAKER:
                break
        # If we're short, fill with longest unused
        if len(chosen) < N_REFS_PER_SPEAKER:
            chosen_set = {id(c) for c in chosen}
            extras = sorted(
                [c for c in candidates if id(c) not in chosen_set],
                key=lambda c: -c["dur"],
            )
            chosen.extend(extras[: N_REFS_PER_SPEAKER - len(chosen)])
        # Final sort by intent_class for stability
        chosen.sort(key=lambda c: c["intent"])

        spk_entries = []
        for k, c in enumerate(chosen):
            src = os.path.join(AUDIO_BASE, c["row"]["audio_path"].replace("\\", "/"))
            dst_name = f"speaker_{spk_id:02d}_ref_{k}.wav"
            dst = os.path.join(OUT_DIR, dst_name)
            shutil.copy2(src, dst)
            spk_entries.append({
                "speaker_id": spk_id,
                "ref_idx": k,
                "ref_wav": os.path.relpath(dst, REPO).replace("\\", "/"),
                "ref_text": c["row"]["template"],
                "duration_sec": round(c["dur"], 2),
                "intent_class": c["intent"],
                "source_idx": c["row"]["idx"],
            })
        manifest.extend(spk_entries)
        durs = [e["duration_sec"] for e in spk_entries]
        intents = sorted({e["intent_class"] for e in spk_entries})
        print(f"  spk {spk_id:>2}: {len(spk_entries)} refs, "
              f"durs={min(durs):.1f}-{max(durs):.1f}s, "
              f"intents={len(intents)} unique ({intents})")

    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(manifest)} refs + manifest.json (target: 11 spk x {N_REFS_PER_SPEAKER} = 88)")


if __name__ == "__main__":
    main()
