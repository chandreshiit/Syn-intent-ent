"""Filesystem layout for datasets, scratch space, and results.

Every path here is overridable by an environment variable so the repository can
be run from any checkout, on any machine, without editing code. Defaults assume
the layout described in docs/DATA.md.

    SLU_GAP_DATA     corpora and generated audio       (default: <repo>/data)
    SLU_GAP_RESULTS  experiment result JSONs           (default: <repo>/results)
    SLU_GAP_SCRATCH  temporary training checkpoints    (default: system temp)
    SLU_GAP_MODELS   downloaded model weights          (default: <scratch>/models)

`SLU_GAP_SCRATCH` and `SLU_GAP_MODELS` matter on machines whose system drive is
small: the speech experiments write multi-gigabyte checkpoints and the Whisper
backbones are downloaded per run, so pointing these at a roomier volume avoids
filling the boot disk.
"""

import os
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _from_env(var, default):
    return Path(os.environ.get(var, default)).resolve()


DATA_ROOT = _from_env("SLU_GAP_DATA", REPO_ROOT / "data")
RESULTS_ROOT = _from_env("SLU_GAP_RESULTS", REPO_ROOT / "results")
SCRATCH_ROOT = _from_env("SLU_GAP_SCRATCH", Path(tempfile.gettempdir()) / "slu_gap")
MODEL_CACHE = _from_env("SLU_GAP_MODELS", SCRATCH_ROOT / "models")
WHISPER_CACHE = MODEL_CACHE / "whisper"

# Corpora. See docs/DATA.md for how to obtain each one -- none are redistributed
# with this repository.
SNIPS_REAL = DATA_ROOT / "snips_real_close"
SNIPS_SYNTH = DATA_ROOT / "snips_synth_for_snipsnlu"
SNIPS_MULTILINGUAL = DATA_ROOT / "snips_multilingual_pipeline"
SNIPS_F5_AUDIO = DATA_ROOT / "snips_f5_cloned_audio"

SKIT_S2I_REAL = DATA_ROOT / "skit_s2i_real_audio"
SKIT_S2I_SYNTH = DATA_ROOT / "skit_s2i_synthesis_pipeline"

MULTIATIS_MULTILINGUAL = DATA_ROOT / "multiatis_multilingual_pipeline"


def scratch(name):
    """Return a created scratch directory for `name`, e.g. scratch("kfold")."""
    path = SCRATCH_ROOT / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def results(*parts):
    """Return a path under results/, creating the parent directory."""
    path = RESULTS_ROOT.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
