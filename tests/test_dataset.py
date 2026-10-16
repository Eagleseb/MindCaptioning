"""Dataset loading, aggregation, and the perception split."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from mind_captioning.captioner.data import (  # noqa: E402
    VideoCaptionDataset,
    assert_disjoint,
    build_examples,
    collate_captions,
    encode_captions,
    examples_for_ids,
    excluded_video_ids,
    load_examples,
    make_split,
    mean_caption_vectors,
    perception_video_ids,
)
from tiny import TinyTokenizer  # noqa: E402


class AggregationTest(unittest.TestCase):
    def test_mean_caption_vectors(self) -> None:
        vectors = np.arange(40, dtype=np.float64).reshape(20, 2)
        aggregated = mean_caption_vectors(vectors)
        self.assertEqual(aggregated.shape, (2,))
        np.testing.assert_allclose(aggregated, vectors.mean(axis=0))
        np.testing.assert_allclose(aggregated, np.array([19.0, 20.0]))

    def test_mean_rejects_empty_and_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            mean_caption_vectors(np.zeros((0, 4)))
        with self.assertRaises(ValueError):
            mean_caption_vectors(np.array([[1.0], [np.nan]]))


class ExampleTest(unittest.TestCase):
    def test_shared_embedding_and_grouping(self) -> None:
        captions = [["a cat", "a dog"], ["one bird", "two birds"]]
        features = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        examples = build_examples(captions, features, [2, 1], captions_per_video=2)
        self.assertEqual([example.video_id for example in examples], [2, 1])
        np.testing.assert_array_equal(examples[0].embedding, features[1])
        self.assertEqual(examples[0].captions, ("one bird", "two birds"))
        dataset = VideoCaptionDataset(examples, mode="all_captions")
        item = dataset[0]
        self.assertEqual(item["captions"], ("one bird", "two birds"))
        self.assertEqual(item["video_id"], 2)

    def test_duplicate_empty_and_bad_embedding(self) -> None:
        captions = [["ok", " "], ["fine", "also"]]
        features = np.ones((2, 3), dtype=np.float32)
        with self.assertRaises(ValueError):
            build_examples(captions, features, [1], captions_per_video=2)
        captions = [["ok", "yes"], ["fine", "also"]]
        with self.assertRaises(ValueError):
            build_examples(captions, features, [1, 1], captions_per_video=2)
        features[0, 0] = np.nan
        with self.assertRaises(ValueError):
            build_examples(captions, features, [1], captions_per_video=2)

    def test_sample_caption_stays_on_the_video(self) -> None:
        captions = [["red", "blue", "green"]]
        features = np.ones((1, 2), dtype=np.float32)
        examples = build_examples(captions, features, [1], captions_per_video=3)
        dataset = VideoCaptionDataset(examples, mode="sample_caption", seed=42)
        dataset.set_epoch(3)
        first = dataset[0]["captions"]
        second = dataset[0]["captions"]
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1)
        self.assertIn(first[0], captions[0])


class EncodeTest(unittest.TestCase):
    def test_gpt2_style_does_not_start_with_eos(self) -> None:
        tokenizer = TinyTokenizer()
        tokenizer.bos_token_id = tokenizer.eos_token_id
        ids, mask = encode_captions(tokenizer, ["red cat"], max_length=6)
        self.assertEqual(ids[0, 0].item(), tokenizer.encode("red")[0])
        self.assertNotEqual(ids[0, 0].item(), tokenizer.eos_token_id)
        self.assertIn(tokenizer.eos_token_id, ids[0].tolist())
        self.assertEqual(int(mask.sum()), 3)

    def test_distinct_bos_is_kept(self) -> None:
        tokenizer = TinyTokenizer()
        ids, mask = encode_captions(tokenizer, ["red cat"], max_length=6)
        self.assertEqual(ids[0, 0].item(), tokenizer.bos_token_id)
        self.assertEqual(int(mask[0, 0]), 1)
        content = [token for token, keep in zip(ids[0].tolist(), mask[0].tolist()) if keep]
        self.assertEqual(content[0], tokenizer.bos_token_id)
        self.assertEqual(content[-1], tokenizer.eos_token_id)


class SplitTest(unittest.TestCase):
    def test_synthetic_split_is_disjoint_and_stable(self) -> None:
        train = np.arange(1, 21)
        test = np.array([21, 22, 23])
        first = make_split(train, test, 0.25, 42)
        second = make_split(train, test, 0.25, 42)
        self.assertEqual(first, second)
        assert_disjoint(first)
        self.assertEqual(len(first["validation"]), 5)
        self.assertEqual(sorted(first["train"] + first["validation"]), list(range(1, 21)))
        self.assertEqual(first["test"], [21, 22, 23])

    def test_perception_split_matches_the_stimulus_files(self) -> None:
        fmri = ROOT / "data/fmri/data/preprocessed"
        feature = ROOT / "data/feature/video/deberta-large/layer24.mat"
        captions = ROOT / "data/caption/caption_ck20.csv"
        if not feature.exists():
            self.skipTest("cached DeBERTa features are not available")
        train_ids, test_ids = perception_video_ids(fmri, "S1")
        split = make_split(train_ids, test_ids, 0.1, 42)
        library = load_examples(captions, feature, 20)
        split["excluded"] = excluded_video_ids(len(library), split)
        assert_disjoint(split)
        self.assertEqual(len(library), 2196)
        self.assertEqual(library[0].embedding.shape, (1024,))
        self.assertEqual(len(split["train"]) + len(split["validation"]), 2108)
        self.assertEqual(len(split["validation"]), 211)
        self.assertEqual(len(split["test"]), 72)
        self.assertEqual(len(split["excluded"]), 16)
        self.assertEqual(set(split["test"]), set(int(video_id) for video_id in test_ids))
        held_out = examples_for_ids(library, split["test"])
        self.assertEqual(len(held_out[0].captions), 20)
        self.assertTrue(np.isfinite(held_out[0].embedding).all())

    def test_collate_repeats_the_video_embedding(self) -> None:
        tokenizer = TinyTokenizer()
        tokenizer.encode("red house")
        tokenizer.encode("blue house")
        captions = [["red house", "blue house"]]
        features = np.array([[5.0, 6.0]], dtype=np.float32)
        examples = build_examples(captions, features, [1], captions_per_video=2)
        batch = collate_captions([VideoCaptionDataset(examples)[0]], tokenizer, max_length=8)
        self.assertEqual(batch["embeddings"].shape, (2, 2))
        self.assertTrue(torch_equal_rows(batch["embeddings"]))
        self.assertEqual(batch["video_index"].tolist(), [0, 0])
        self.assertEqual(batch["video_ids"], [1])


def torch_equal_rows(values) -> bool:
    return bool((values[0] == values[1]).all())


if __name__ == "__main__":
    unittest.main()
