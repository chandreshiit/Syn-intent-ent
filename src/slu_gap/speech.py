"""Shared speech model for the end-to-end SLU experiments.

All speech results in the paper use the same architecture -- a Whisper encoder
with a mean-pooled linear intent head -- so that comparisons across audio
conditions (generic voices, channel-matched, voice-cloned) isolate the data and
not the model. This module is the single definition; the Skit-S2I and SNIPS
experiments both import it.
"""

import torch
import torch.nn as nn
import whisper


class WhisperIntentClassifier(nn.Module):
    """Whisper encoder + mean pool + linear classifier.

    Matches the Whisper baseline of the Skit-S2I dataset paper. The whole
    encoder is fine-tuned, not frozen.
    """

    FEATURE_DIMS = {
        "tiny.en": 384, "tiny": 384,
        "base.en": 512, "base": 512,
        "small.en": 768, "small": 768,
        "medium.en": 1024, "medium": 1024,
    }

    def __init__(self, backbone="tiny.en", n_class=6, download_root=None):
        super().__init__()
        self.encoder = whisper.load_model(backbone, download_root=download_root).encoder
        for p in self.encoder.parameters():
            p.requires_grad = True
        self.classifier = nn.Linear(self.FEATURE_DIMS.get(backbone, 384), n_class)

    def forward(self, mel):
        # mel: (B, 80, T) -> pooled (B, dim) -> logits (B, n_class)
        pooled = self.encoder(mel).mean(dim=1)
        return self.classifier(pooled)


def collate(batch):
    """Stack (log_mel, intent_id) pairs into a batch."""
    mels = torch.stack([b[0] for b in batch], dim=0)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return mels, labels
