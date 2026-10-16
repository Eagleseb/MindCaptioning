"""Label shift, padding, and per-video loss weights."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mind_captioning.captioner.model import (  # noqa: E402
    IGNORE_INDEX,
    build_labels,
    grouped_caption_loss,
    mean_by_video,
)
from tiny import tiny_captioner  # noqa: E402


class LabelAlignmentTest(unittest.TestCase):
    def test_prefix_and_padding_are_ignored_after_the_shift(self) -> None:
        input_ids = torch.tensor([[7, 8, 9, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        labels = build_labels(input_ids, mask, prefix_length=2)
        self.assertTrue(torch.equal(labels, torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 7, 8, 9, IGNORE_INDEX]])))
        targets = labels[:, 1:]
        self.assertTrue(torch.equal(targets, torch.tensor([[IGNORE_INDEX, 7, 8, 9, IGNORE_INDEX]])))
        self.assertEqual(targets[targets.ne(IGNORE_INDEX)].tolist(), [7, 8, 9])

    def test_video_mean_weights_captions_equally(self) -> None:
        per_caption = torch.tensor([0.0, 2.0, 1.0])
        video_index = torch.tensor([0, 0, 1])
        loss = mean_by_video(per_caption, video_index)
        self.assertAlmostEqual(float(loss), 1.0)

    def test_padding_does_not_change_the_loss(self) -> None:
        torch.manual_seed(0)
        model = tiny_captioner(prefix_tokens=2)
        model.eval()
        embeddings = torch.randn(1, 8)
        input_ids = torch.tensor([[4, 5, 6, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        logits, labels = model(embeddings, input_ids, mask)
        changed = input_ids.clone()
        changed[0, -1] = 9
        logits_changed, labels_changed = model(embeddings, changed, mask)
        self.assertTrue(torch.equal(labels, labels_changed))
        self.assertEqual(int(labels[0, :2].eq(IGNORE_INDEX).sum()), 2)
        self.assertEqual(int(labels[0, -1]), IGNORE_INDEX)
        torch.testing.assert_close(logits[:, :-1], logits_changed[:, :-1])
        video_index = torch.tensor([0])
        first = grouped_caption_loss(logits, labels, video_index)
        second = grouped_caption_loss(logits_changed, labels_changed, video_index)
        self.assertTrue(torch.allclose(first, second))
        self.assertTrue(torch.isfinite(first))

    def test_ignored_positions_do_not_contribute(self) -> None:
        logits = torch.zeros(1, 4, 3)
        labels = torch.tensor([[IGNORE_INDEX, 1, IGNORE_INDEX, IGNORE_INDEX]])
        logits[0, 0, 1] = 20.0
        loss = grouped_caption_loss(logits, labels, torch.tensor([0]))
        self.assertLess(float(loss), 0.01)
        logits[0, 0, 1] = -20.0
        logits[0, 2, 0] = 50.0
        worse = grouped_caption_loss(logits, labels, torch.tensor([0]))
        self.assertGreater(float(worse), 1.0)


if __name__ == "__main__":
    unittest.main()
