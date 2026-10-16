"""Projector shape and independence from the decoder."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mind_captioning.captioner.model import FeatureProjector  # noqa: E402


class ProjectorTest(unittest.TestCase):
    def test_output_shape(self) -> None:
        projector = FeatureProjector(embedding_dim=12, decoder_hidden_size=7, num_prefix_tokens=8, hidden_dim=1024)
        output = projector(torch.randn(4, 12))
        self.assertEqual(output.shape, (4, 8, 7))

    def test_rejects_the_wrong_width(self) -> None:
        projector = FeatureProjector(5, 6, num_prefix_tokens=2, hidden_dim=8)
        with self.assertRaises(ValueError):
            projector(torch.randn(2, 4))


if __name__ == "__main__":
    unittest.main()
