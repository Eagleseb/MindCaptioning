"""Prefix generation, KV-cache alignment, and checkpoint reload."""

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
from mind_captioning.captioner.data import VideoExample  # noqa: E402
from mind_captioning.captioner.generate import decode_caption, generate_caption, load_embedding_file  # noqa: E402
from mind_captioning.captioner.metrics import rouge_l, rouge_l_max  # noqa: E402
from mind_captioning.captioner.reporting import evaluate_examples  # noqa: E402
from tiny import TinyTokenizer, force_token, tiny_captioner  # noqa: E402


class GenerationTest(unittest.TestCase):
    def test_prefix_matches_cached_continuation(self) -> None:
        torch.manual_seed(0)
        model = tiny_captioner()
        model.eval()
        embedding = torch.randn(2, 8)
        prefix = model.projector(embedding)
        mask = torch.ones(2, prefix.shape[1], dtype=torch.long)
        with torch.no_grad():
            prefill = model.decoder(inputs_embeds=prefix, attention_mask=mask, use_cache=True)
            first = prefill.logits[:, -1].argmax(dim=-1)
            longer = torch.ones(2, prefix.shape[1] + 1, dtype=torch.long)
            second_out = model.decoder(
                input_ids=first[:, None],
                attention_mask=longer,
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            second = second_out.logits[:, -1].argmax(dim=-1)
            generated = model.decoder.generate(
                inputs_embeds=prefix,
                attention_mask=mask,
                max_new_tokens=2,
                min_new_tokens=2,
                do_sample=False,
                pad_token_id=0,
                eos_token_id=2,
            )
        self.assertTrue(torch.equal(generated[:, 0], first))
        self.assertTrue(torch.equal(generated[:, 1], second))

    def test_greedy_caption_is_nonempty_and_repeatable(self) -> None:
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        tokenizer.encode("dog")
        model = tiny_captioner(vocab_size=max(tokenizer._token_to_id.values()) + 1)
        force_token(model.decoder, tokenizer.encode("dog")[0])
        embedding = np.ones(8, dtype=np.float32)
        first = generate_caption(model, tokenizer, embedding, max_new_tokens=3)
        second = generate_caption(model, tokenizer, embedding, max_new_tokens=3)
        self.assertEqual(first, second)
        self.assertTrue(str(first).strip())
        self.assertIn("dog", str(first))

    def test_decode_stops_at_eos(self) -> None:
        tokenizer = TinyTokenizer()
        dog = tokenizer.encode("dog")[0]
        self.assertEqual(decode_caption(tokenizer, [dog, tokenizer.eos_token_id, dog]), "dog")

    def test_checkpoint_reloads_the_same_caption(self) -> None:
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        tokenizer.encode("dog")
        model = tiny_captioner(vocab_size=max(tokenizer._token_to_id.values()) + 1)
        embedding = np.linspace(0.1, 0.8, 8, dtype=np.float32)
        config = {
            "decoder_name": "tiny",
            "tokenizer_name": "tiny",
            "max_length": 8,
            "captions_per_video": 1,
            "seed": 42,
            "feature_model": "synthetic",
            "layer": 1,
        }
        split = {"train": [1], "validation": [2], "test": [3]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best"
            save_checkpoint(path, model, config, split, save_decoder=True)
            loaded, _, stored, loaded_split = load_checkpoint(path, torch.device("cpu"), tokenizer=tokenizer)
            original = generate_caption(model, tokenizer, embedding, max_new_tokens=4)
            restored = generate_caption(loaded, tokenizer, embedding, max_new_tokens=4)
            self.assertEqual(original, restored)
            self.assertTrue(torch.equal(model.projector.net[0].weight, loaded.projector.net[0].weight))
            self.assertTrue(
                torch.equal(model.decoder.transformer.wte.weight, loaded.decoder.transformer.wte.weight)
            )
            self.assertEqual(loaded_split["test"], [3])
            self.assertEqual(stored["normalization"], "none")
            array_path = Path(directory) / "embedding.npy"
            np.save(array_path, embedding)
            np.testing.assert_allclose(load_embedding_file(array_path), embedding)

    def test_batch_generation(self) -> None:
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        token = tokenizer.encode("bird")[0]
        model = tiny_captioner(vocab_size=token + 1)
        force_token(model.decoder, token)
        embeddings = np.ones((2, 8), dtype=np.float32)
        texts = generate_caption(model, tokenizer, embeddings, max_new_tokens=2, num_beams=2)
        self.assertEqual(len(texts), 2)
        self.assertTrue(all("bird" in text for text in texts))


class MetricTest(unittest.TestCase):
    def test_rouge_l(self) -> None:
        self.assertAlmostEqual(rouge_l("a cat sat", "a cat sat"), 1.0)
        self.assertEqual(rouge_l("a cat sat", "blue train"), 0.0)
        self.assertAlmostEqual(rouge_l_max("a cat", ["blue train", "a cat sat"]), 0.8)

    def test_report_records_the_limitation(self) -> None:
        torch.manual_seed(0)
        tokenizer = TinyTokenizer()
        tokenizer.encode("cat runs")
        tokenizer.encode("dog sits")
        model = tiny_captioner(vocab_size=len(tokenizer._token_to_id) + 2)
        train = [VideoExample(1, np.zeros(8, dtype=np.float32), ("cat runs",))]
        test = [VideoExample(2, np.ones(8, dtype=np.float32), ("dog sits",))]
        with tempfile.TemporaryDirectory() as directory:
            summary = evaluate_examples(
                model,
                tokenizer,
                test,
                train,
                Path(directory) / "test.jsonl",
                torch.device("cpu"),
                max_length=8,
                batch_size=2,
                seed=0,
                max_new_tokens=4,
            )
            self.assertTrue((Path(directory) / "test.jsonl").exists())
        self.assertIn("reference captions", summary["limitation"])
        self.assertEqual(summary["n_videos"], 1)
        self.assertTrue(np.isfinite(summary["loss"]))


if __name__ == "__main__":
    unittest.main()
