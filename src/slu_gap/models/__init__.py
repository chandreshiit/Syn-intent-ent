"""Joint intent-classification / slot-filling baselines.

`bert_alone` (JointBERT) and `lstm_alone` (Bi-LSTM) are adapted from the
official MultiATIS++ repository and share the dataset, vocabulary, and CoNLL
evaluation helpers in `utils` and `conlleval`.

Both are module-configured rather than class-configured: assign `model_dir`
and call `set_seed()` before training. See each module's docstring.
"""

from . import bert_alone, conlleval, lstm_alone, utils

__all__ = ["bert_alone", "lstm_alone", "utils", "conlleval"]
