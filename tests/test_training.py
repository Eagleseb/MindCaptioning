"""Frozen decoder, finite decreasing loss, and checkpoints."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mind_captioning.captioner.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from mind_captioning.captioner.data import VideoCaptionDataset, VideoExample  # noqa: E402
from mind_captioning.captioner.loop import fit  # noqa: E402
from mind_captioning.captioner.model import attach_lora, grouped_caption_loss  # noqa: E402
from tiny import TinyTokenizer, tiny_captioner  # noqa: E402


def synthetic_examples() -> list[VideoExample]:
    examples = []
    words = ("alpha", "beta", "gamma", "delta")
    for index, word in enumerate(words):
        embedding = np.zeros(8, dtype=np.float32)
        embedding[index] = 1.0
        examples.append(VideoExample(index + 1, embedding, (f"{word} {word}", f"{word} {word}")))
    return examples


class GradientTest(unittest.TestCase):
    def test_projector_updates_and_decoder_stays_fixed(self) -> None:
        torch.manual_seed(0)
        model = tiny_captioner()
        tokenizer = TinyTokenizer()
        for word in ("alpha", "beta"):
            tokenizer.encode(word)
        model.train()
        self.assertTrue(model.projector.training)
        self.assertFalse(model.decoder.training)
        before = [parameter.detach().clone() for parameter in model.decoder.parameters()]
        embeddings = torch.randn(2, 8)
        input_ids = torch.tensor([[tokenizer.bos_token_id, 3, tokenizer.eos_token_id, 0]] * 2)
        mask = torch.tensor([[1, 1, 1, 0]] * 2)
        logits, labels = model(embeddings, input_ids, mask)
        loss = grouped_caption_loss(logits, labels, torch.tensor([0, 1]))
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.AdamW(model.projector.parameters(), lr=1e-2)
        projector_before = model.projector.net[0].weight.detach().clone()
        loss.backward()
        self.assertIsNotNone(model.projector.net[0].weight.grad)
        self.assertGreater(float(model.projector.net[0].weight.grad.abs().sum()), 0.0)
        optimizer.step()
        self.assertFalse(torch.equal(projector_before, model.projector.net[0].weight))
        for old, new in zip(before, model.decoder.parameters()):
            self.assertTrue(torch.equal(old, new))
        self.assertFalse(any(parameter.requires_grad for parameter in model.decoder.parameters()))

    def test_lora_receives_gradients_and_round_trips(self) -> None:
        torch.manual_seed(0)
        model = tiny_captioner()
        tokenizer = TinyTokenizer()
        tokenizer.encode("alpha")
        attach_lora(model.decoder, rank=2, alpha=4, layers=1)
        model.freeze_decoder()
        projection = model.decoder.transformer.h[-1].attn.c_attn
        self.assertFalse(projection.base.weight.requires_grad)
        self.assertTrue(projection.lora_A.requires_grad)
        embeddings = torch.randn(2, 8)
        input_ids = torch.tensor([[tokenizer.bos_token_id, 3, tokenizer.eos_token_id, 0]] * 2)
        mask = torch.tensor([[1, 1, 1, 0]] * 2)
        logits, labels = model(embeddings, input_ids, mask)
        grouped_caption_loss(logits, labels, torch.tensor([0, 1])).backward()
        self.assertIsNotNone(projection.lora_B.grad)
        self.assertGreater(float(projection.lora_B.grad.abs().sum()), 0.0)
        self.assertIsNone(projection.base.weight.grad)
        mean = np.zeros(8, dtype=np.float32)
        scale = np.ones(8, dtype=np.float32)
        config = {
            "decoder_name": "tiny",
            "tokenizer_name": "tiny",
            "max_length": 8,
            "captions_per_video": 1,
            "seed": 0,
            "save_decoder": True,
            "normalization": "zscore",
            "input_norm_arrays": {"mean": mean, "scale": scale},
            "lora": {"rank": 2, "alpha": 4.0, "layers": 1},
        }
        split = {"train": [1], "validation": [2], "test": [3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            save_checkpoint(path, model, config, split, save_decoder=True)
            loaded, _, stored, _ = load_checkpoint(path, torch.device("cpu"), tokenizer=tokenizer)
            self.assertEqual(stored["normalization"], "zscore")
            np.testing.assert_allclose(stored["input_norm_arrays"]["mean"], mean)
            restored = loaded.decoder.transformer.h[-1].attn.c_attn
            self.assertTrue(torch.equal(restored.lora_A.detach().cpu(), projection.lora_A.detach().cpu()))
            self.assertTrue(restored.lora_A.requires_grad)
            self.assertFalse(restored.base.weight.requires_grad)


class FitTest(unittest.TestCase):
    def test_loss_decreases_and_best_checkpoint_is_written(self) -> None:
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        examples = synthetic_examples()
        for example in examples:
            for caption in example.captions:
                tokenizer.encode(caption)
        model = tiny_captioner(vocab_size=len(tokenizer._token_to_id) + 4)
        train = VideoCaptionDataset(examples[:3], mode="all_captions", seed=0)
        val = VideoCaptionDataset(examples[3:], mode="all_captions", seed=0)
        split = {"train": [1, 2, 3], "validation": [4], "test": [5]}
        config = {
            "decoder_name": "tiny",
            "tokenizer_name": "tiny",
            "max_length": 8,
            "captions_per_video": 2,
            "seed": 0,
            "save_decoder": True,
            "feature_model": "synthetic",
            "layer": 1,
        }
        with tempfile.TemporaryDirectory() as directory:
            best = fit(
                model,
                tokenizer,
                train,
                val,
                Path(directory),
                config,
                split,
                epochs=12,
                patience=12,
                batch_size=3,
                learning_rate=3e-2,
                weight_decay=0.0,
                grad_clip=1.0,
                seed=0,
                device=torch.device("cpu"),
                max_length=8,
            )
            history = __import__("json").loads((Path(directory) / "history.json").read_text())
            self.assertGreaterEqual(len(history), 2)
            self.assertTrue(all(np.isfinite(row["train_loss"]) for row in history))
            self.assertLess(history[-1]["train_loss"], history[0]["train_loss"])
            self.assertTrue((best / "projector.pt").exists())
            self.assertTrue((best / "decoder.pt").exists())
            stored = __import__("json").loads((best / "config.json").read_text())
            self.assertIn("best_validation_loss", stored)
            self.assertEqual(stored["normalization"], "none")
            self.assertIn("reference captions", stored["limitation"])


if __name__ == "__main__":
    unittest.main()
