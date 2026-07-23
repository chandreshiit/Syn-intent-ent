#!/usr/bin/env python3
"""Compare duration/loudness of real SNIPS vs MMS-TTS synth vs F5-cloned synth."""
import json
import os
import statistics as st

import numpy as np
import soundfile as sf

from slu_gap import paths
from slu_gap.datasets import load_real_snips_close, load_synth_snips

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(paths.REPO_ROOT)

N = 300
CLONED = os.path.join(REPO, "data/snips_f5_cloned_audio")


def stats(paths, label):
    durs, rmss, peaks = [], [], []
    for p in paths[:N]:
        try:
            a, sr = sf.read(p, dtype="float32", always_2d=False)
        except Exception:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        durs.append(len(a) / sr)
        rmss.append(float(np.sqrt(np.mean(a ** 2) + 1e-12)))
        peaks.append(float(np.abs(a).max()))
    return {"label": label, "n": len(durs), "sr": sr,
            "dur_p50": st.median(durs), "dur_mean": st.mean(durs), "dur_p90": sorted(durs)[int(.9*len(durs))],
            "rms_p50": st.median(rmss), "peak_p50": st.median(peaks)}


def main():
    real_p, _ = load_real_snips_close()
    mms_p, _ = load_synth_snips()
    lab = [l.strip() for l in open(os.path.join(REPO, "data/snips_multilingual_pipeline/processed_data/en/all/label"), encoding="utf-8") if l.strip()]
    clone_p = [os.path.join(CLONED, f"cmd_{i:04d}_en.wav") for i in range(len(lab))]
    clone_p = [p for p in clone_p if os.path.exists(p)]

    rows = [stats(real_p, "REAL close-field"), stats(mms_p, "SYNTH MMS-TTS (1 voice)")]
    if clone_p:
        rows.append(stats(clone_p, "SYNTH F5-cloned (51 voices)"))

    print(f"{'source':<28}{'n':>5}{'sr':>7}{'dur_p50':>9}{'dur_mean':>10}{'dur_p90':>9}{'rms_p50':>9}{'peak_p50':>9}")
    print("-" * 86)
    for r in rows:
        print(f"{r['label']:<28}{r['n']:>5}{r['sr']:>7}{r['dur_p50']:>9.2f}{r['dur_mean']:>10.2f}"
              f"{r['dur_p90']:>9.2f}{r['rms_p50']:>9.4f}{r['peak_p50']:>9.3f}")

    if len(rows) == 3:
        real, mms, f5 = rows
        print(f"\nduration_p50 vs real: MMS {mms['dur_p50']-real['dur_p50']:+.2f}s   F5 {f5['dur_p50']-real['dur_p50']:+.2f}s")
    out = os.path.join(REPO, "phase3/results/snips_audio_characterization.json")
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
