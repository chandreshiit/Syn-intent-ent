"""Corpus loaders shared between experiments and pipelines.

Each loader returns parallel lists of (audio_paths, intent_labels) so callers
can index, split, and subsample them freely. None of these corpora ship with
the repository -- see docs/DATA.md for how to obtain and stage them.
"""

import json
import os

from . import paths

# SNIPS smart-lights intents. Matches
# snips_multilingual_pipeline/processed_data/intent_label.txt minus the leading
# UNK token, so the indices here are the label ids the models are trained on.
SNIPS_INTENTS = [
    "decreasebrightness",
    "increasebrightness",
    "setlightbrightness",
    "setlightcolor",
    "switchlightoff",
    "switchlighton",
]
SNIPS_INTENT2IDX = {name: i for i, name in enumerate(SNIPS_INTENTS)}
SNIPS_N_INTENTS = len(SNIPS_INTENTS)


def load_synth_snips(audio_dir=None, label_path=None):
    """Synthetic SNIPS audio and intent labels.

    Audio filenames are positional (`cmd_0000_en.wav`), aligned by row with the
    label file. Three sample points are existence-checked so a partially
    generated corpus fails immediately rather than midway through training.
    """
    root = paths.SNIPS_MULTILINGUAL
    label_path = label_path or root / "processed_data" / "en" / "all" / "label"
    audio_dir = audio_dir or root / "generated_audio"

    with open(label_path, "r", encoding="utf-8") as f:
        intents = [line.strip() for line in f if line.strip()]

    audio_paths = [os.path.join(audio_dir, f"cmd_{i:04d}_en.wav") for i in range(len(intents))]
    for j in (0, len(audio_paths) // 2, len(audio_paths) - 1):
        if not os.path.exists(audio_paths[j]):
            raise FileNotFoundError(
                f"Missing synthetic SNIPS audio: {audio_paths[j]}. "
                "Run pipelines/snips/03_synthesize_multilingual_speech.py first."
            )
    return audio_paths, intents


def load_real_snips_close(root=None):
    """Real SNIPS smart-lights-en-close-field audio and intent labels.

    Requires `audio_index.json` (BIO row -> wav path), built by
    experiments/data_prep/prepare_real_snips.py. Rows whose audio is missing are
    dropped, and the returned lists stay aligned.
    """
    root = paths.SNIPS_REAL if root is None else root

    with open(root / "bio" / "label", "r", encoding="utf-8") as f:
        intents = [line.strip() for line in f if line.strip()]
    with open(root / "audio_index.json", "r", encoding="utf-8") as f:
        audio_index = json.load(f)

    audio_paths, kept = [], []
    for i in range(len(intents)):
        p = audio_index.get(str(i))
        if p and os.path.exists(p):
            audio_paths.append(p)
            kept.append(i)
    return audio_paths, [intents[i] for i in kept]
