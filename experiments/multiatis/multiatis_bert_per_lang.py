#!/usr/bin/env python3
"""
Run MultiATIS++ BERT-only baselines one language at a time, each in a fresh
Python subprocess. This sidesteps the CUDA-fragmentation OOM that hits when
multiple BERT models train in the same process on Windows.

Usage:
    python experiments/multiatis/multiatis_bert_per_lang.py [--langs de fr ...] [--epochs 20]

For each language, the subprocess writes its own JSON under
results/multiatis/multiatis_bert_<lang>_tmp/; this script then merges them into
the canonical multiatis_multilingual_baselines.json.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from slu_gap import paths

LANGS_DEFAULT = ["pt", "de", "fr", "zh", "ja", "hi", "tr"]
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = str(paths.REPO_ROOT)


def run_one(lang, epochs):
    """Spawn a fresh process running multiatis_multilingual_baselines.py for one lang."""
    out_dir = str(paths.results("multiatis", f"multiatis_bert_{lang}_tmp"))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(_HERE, "multiatis_multilingual_baselines.py"),
        "--languages", lang,
        "--models", "bert",
        "--epochs", str(epochs),
        "--out-dir", out_dir,
    ]
    env = os.environ.copy()
    # Keep per-language temp files off the system drive; these runs write
    # several GB of checkpoints each. Override with $SLU_GAP_SCRATCH.
    scratch = str(paths.scratch("multiatis"))
    env["TMP"] = scratch
    env["TEMP"] = scratch
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    os.makedirs(env["TMP"], exist_ok=True)
    print(f"\n{'='*60}\nRunning BERT-only for {lang} in fresh subprocess\n{'='*60}", flush=True)
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=REPO, env=env)
    print(f"  rc={rc}  elapsed={time.time()-t0:.1f}s", flush=True)
    # Read the result
    out_json = os.path.join(out_dir, "multiatis_multilingual_baselines.json")
    if os.path.exists(out_json):
        with open(out_json, "r", encoding="utf-8") as f:
            return json.load(f).get("per_language", [{}])[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=LANGS_DEFAULT)
    ap.add_argument("--epochs", type=int, default=20)
    args = ap.parse_args()

    main_json = os.path.join(REPO, "phase3", "results", "multiatis_multilingual_baselines.json")
    with open(main_json, "r", encoding="utf-8") as f:
        main_data = json.load(f)
    by_lang = {r["lang"]: r for r in main_data["per_language"]}

    for lang in args.langs:
        result = run_one(lang, args.epochs)
        if result is None:
            print(f"  {lang}: NO RESULT FILE", flush=True)
            continue
        # Merge BERT result into the existing entry (preserve LSTM)
        existing = by_lang.get(lang, {})
        existing["lang"] = lang
        if "bert" in result:
            existing["bert"] = result["bert"]
        if "lstm" not in existing and "lstm" in result:
            existing["lstm"] = result["lstm"]
        # Update n_train / n_test in case missing
        for k in ("n_train", "n_dev", "n_test", "intents", "slot_labels"):
            if k in result and k not in existing:
                existing[k] = result[k]
        by_lang[lang] = existing
        # Write after each language so a crash doesn't lose progress
        main_data["per_language"] = [by_lang[l] for l in ["en", "es", "pt", "de", "fr", "zh", "ja", "hi", "tr"] if l in by_lang]
        with open(main_json, "w", encoding="utf-8") as f:
            json.dump(main_data, f, indent=2)

    print(f"\nMerged JSON: {main_json}")


if __name__ == "__main__":
    main()
