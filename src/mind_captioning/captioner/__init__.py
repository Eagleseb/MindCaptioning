"""Reconstruct captions from cached caption embeddings.

Each input vector is the masked mean of a DeBERTa layer, averaged across the
reference captions of one video. Training and the reported scores measure
whether a frozen language model can read that vector. They do not measure
captioning from pixels or from brain activity.
"""

from __future__ import annotations

LIMITATION = (
    "Each embedding is the masked mean of DeBERTa features of that video's "
    "reference captions, averaged across those captions and left unnormalized. "
    "Loss, perplexity, and caption metrics measure reconstruction from those "
    "caption-derived features. They do not measure captioning of video pixels "
    "or of brain activity."
)

POOLING = "masked_mean_exclude_special"
NORMALIZATION = "none"

__all__ = ["LIMITATION", "NORMALIZATION", "POOLING"]
