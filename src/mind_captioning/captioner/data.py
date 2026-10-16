"""Video-level caption examples and the perception split.

Feature files already store one vector per video. Row ``i`` is ``video_id``
``i + 1``. That vector is the mean of the caption vectors, and each caption
vector is a masked mean that excludes padding and special tokens. This module
reads those rows. It does not encode captions again and it does not z-score them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from mind_captioning.io import load_captions, load_stimuli, load_video_features


@dataclass(frozen=True)
class VideoExample:
    """One video, its shared embedding, and the reference captions."""

    video_id: int
    embedding: np.ndarray
    captions: tuple[str, ...]


def model_dirname(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def available_layers(directory: Path) -> list[int]:
    layers = []
    for path in directory.glob("layer*.mat"):
        suffix = path.stem.removeprefix("layer")
        if suffix.isdigit():
            layers.append(int(suffix))
    return sorted(layers)


def feature_matrix_path(feature_root: Path, model_id: str, layer: int | None) -> tuple[Path, int]:
    """Resolve ``data/feature/video/<model>/layerXX.mat``.

    ``layer=None`` selects the last saved layer.
    """
    directory = feature_root / "video" / model_dirname(model_id)
    layers = available_layers(directory)
    if not layers:
        raise FileNotFoundError(
            f"No layer files in {directory}. Run extract-video-features for {model_id} first."
        )
    chosen = layers[-1] if layer is None else layer
    if chosen not in layers:
        joined = ", ".join(f"{index:02d}" for index in layers)
        raise FileNotFoundError(
            f"Layer {chosen} is not in {directory}. Available layers: {joined}"
        )
    return directory / f"layer{chosen:02d}.mat", chosen


def mean_caption_vectors(vectors: np.ndarray) -> np.ndarray:
    """Average caption vectors of one video.

    ``vectors`` has shape ``(n_captions, dim)``. ``extract_features`` applies
    this average after masked-mean pooling and writes the result to the
    feature matrix, so training reads one row per video.
    """
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise ValueError(
            f"Expected caption vectors with shape (n_captions, dim), got {vectors.shape}"
        )
    if not np.isfinite(vectors).all():
        raise ValueError("Caption vectors contain non-finite values")
    return vectors.mean(axis=0)


def build_examples(
    captions: list[list[str]],
    features: np.ndarray,
    video_ids: Iterable[int],
    captions_per_video: int,
) -> list[VideoExample]:
    """Pair each requested video with its feature row and captions.

    ``captions[video_id - 1]`` must contain exactly ``captions_per_video``
    nonempty strings. Every selected row must be finite and share ``features``'
    width. Video ids must be unique.
    """
    if captions_per_video < 1:
        raise ValueError("captions_per_video must be positive")
    if features.ndim != 2:
        raise ValueError(f"Features must have shape (n_videos, dim), got {features.shape}")
    examples: list[VideoExample] = []
    seen: set[int] = set()
    for raw_id in video_ids:
        video_id = int(raw_id)
        if video_id in seen:
            raise ValueError(f"Duplicate video id {video_id}")
        seen.add(video_id)
        row = video_id - 1
        if row < 0 or row >= len(captions) or row >= features.shape[0]:
            raise ValueError(
                f"Video id {video_id} is outside captions 1..{len(captions)} "
                f"and features 1..{features.shape[0]}"
            )
        texts = tuple(text.strip() for text in captions[row])
        if len(texts) != captions_per_video:
            raise ValueError(
                f"Video {video_id} has {len(texts)} captions, expected {captions_per_video}"
            )
        if any(not text for text in texts):
            raise ValueError(f"Video {video_id} has an empty caption")
        embedding = np.array(features[row], dtype=np.float32, copy=True)
        if embedding.shape != (features.shape[1],) or not np.isfinite(embedding).all():
            raise ValueError(f"Video {video_id} has a malformed embedding of shape {embedding.shape}")
        examples.append(VideoExample(video_id, embedding, texts))
    return examples


def load_examples(
    captions_path: Path,
    feature_path: Path,
    captions_per_video: int,
) -> list[VideoExample]:
    """Load every video in the caption table and check it against the feature matrix."""
    captions, n_videos = load_captions(captions_path)
    features = load_video_features(feature_path)
    if features.shape[0] != n_videos:
        raise ValueError(
            f"{feature_path} has {features.shape[0]} videos and {captions_path} has {n_videos}"
        )
    counts = {len(texts) for texts in captions}
    if counts != {captions_per_video}:
        found = ", ".join(str(count) for count in sorted(counts))
        raise ValueError(
            f"Expected {captions_per_video} captions per video, found counts {found}"
        )
    return build_examples(captions, features, range(1, n_videos + 1), captions_per_video)


def perception_video_ids(fmri_dir: Path, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """Return unique 1-based train and test perception stimulus ids."""
    train_path = fmri_dir / f"trainPerception_{subject}.mat"
    test_path = fmri_dir / f"testPerception_{subject}.mat"
    train_ids, _ = load_stimuli(train_path)
    test_ids, _ = load_stimuli(test_path)
    return np.unique(train_ids.astype(np.int64)), np.unique(test_ids.astype(np.int64))


def assert_disjoint(groups: dict[str, Iterable[int]]) -> None:
    """Raise when any two named id lists share a video."""
    seen: dict[int, str] = {}
    for name, video_ids in groups.items():
        for video_id in video_ids:
            video_id = int(video_id)
            previous = seen.get(video_id)
            if previous is not None:
                raise ValueError(f"Video {video_id} is in both {previous} and {name}")
            seen[video_id] = name


def make_split(
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    val_fraction: float,
    seed: int,
) -> dict[str, list[int]]:
    """Hold out validation videos from the train-perception ids.

    Test ids stay intact. The permutation uses ``numpy.random.default_rng(seed)``
    on the sorted unique train ids, so the same seed reproduces the split.
    """
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    train_ids = np.unique(np.asarray(train_ids, dtype=np.int64))
    test_ids = np.unique(np.asarray(test_ids, dtype=np.int64))
    overlap = np.intersect1d(train_ids, test_ids)
    if overlap.size:
        preview = ", ".join(str(int(video_id)) for video_id in overlap[:10])
        raise ValueError(f"Train and test perception videos overlap, including {preview}")
    if train_ids.size < 2:
        raise ValueError("Need at least two training videos to hold out validation")
    n_val = int(round(train_ids.size * val_fraction))
    n_val = min(max(n_val, 1), int(train_ids.size) - 1)
    order = np.random.default_rng(seed).permutation(train_ids.size)
    shuffled = np.sort(train_ids)[order]
    validation = np.sort(shuffled[:n_val])
    train = np.sort(shuffled[n_val:])
    split = {
        "train": [int(video_id) for video_id in train],
        "validation": [int(video_id) for video_id in validation],
        "test": [int(video_id) for video_id in np.sort(test_ids)],
    }
    assert_disjoint(split)
    return split


def excluded_video_ids(n_videos: int, split: dict[str, list[int]]) -> list[int]:
    """Library videos that are neither train, validation, nor test."""
    used = set(split["train"]) | set(split["validation"]) | set(split["test"])
    return [video_id for video_id in range(1, n_videos + 1) if video_id not in used]


def examples_for_ids(examples: list[VideoExample], video_ids: Iterable[int]) -> list[VideoExample]:
    by_id = {example.video_id: example for example in examples}
    selected = []
    for video_id in video_ids:
        video_id = int(video_id)
        if video_id not in by_id:
            raise KeyError(f"Video {video_id} is not in the loaded library")
        selected.append(by_id[video_id])
    return selected


def start_token_id(tokenizer) -> int | None:
    """Return a distinct start-token id, or None when it would also mean end-of-sequence."""
    bos = getattr(tokenizer, "bos_token_id", None)
    eos = getattr(tokenizer, "eos_token_id", None)
    if bos is None or bos == eos:
        return None
    return int(bos)


def encode_captions(tokenizer, captions: list[str], max_length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode captions as ``content + eos``, padded on the right.

    GPT-2 uses one id for the start and end tokens. Prepending that id would
    teach the model to emit end-of-sequence immediately, so a start token is
    included only when it is different from the end token. Padding and the
    end token stay distinguishable via the attention mask: the end token is a
    real target, and padding is not.
    """
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("The tokenizer needs an end-of-sequence id")
    eos = int(eos)
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = eos if pad is None else int(pad)
    bos = start_token_id(tokenizer)
    prefix = [bos] if bos is not None else []
    room = max_length - len(prefix) - 1
    if room < 1:
        raise ValueError(f"max_length must be at least {len(prefix) + 2} to fit a token and eos")
    rows: list[list[int]] = []
    masks: list[list[int]] = []
    for text in captions:
        content = [int(token) for token in tokenizer.encode(text, add_special_tokens=False)]
        if not content:
            raise ValueError(f"Caption has no tokens after encoding: {text!r}")
        ids = prefix + content[:room] + [eos]
        mask = [1] * len(ids)
        pad_count = max_length - len(ids)
        rows.append(ids + [pad] * pad_count)
        masks.append(mask + [0] * pad_count)
    return torch.tensor(rows, dtype=torch.long), torch.tensor(masks, dtype=torch.long)


class VideoCaptionDataset(Dataset):
    """One item per video.

    ``all_captions`` returns every reference. ``sample_caption`` returns one
    reference, chosen from ``seed``, the epoch, and the video id.
    """

    def __init__(
        self,
        examples: list[VideoExample],
        mode: str = "all_captions",
        seed: int = 42,
    ) -> None:
        if mode not in {"all_captions", "sample_caption"}:
            raise ValueError("mode must be all_captions or sample_caption")
        if not examples:
            raise ValueError("The dataset has no videos")
        self.examples = list(examples)
        self.mode = mode
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        captions = example.captions
        if self.mode == "sample_caption":
            rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + example.video_id)
            captions = (captions[int(rng.integers(0, len(captions)))],)
        if not captions:
            raise ValueError(f"Video {example.video_id} has no captions to sample")
        return {
            "video_id": example.video_id,
            "embedding": example.embedding,
            "captions": captions,
        }


def collate_captions(batch: list[dict], tokenizer, max_length: int) -> dict[str, torch.Tensor | list[int]]:
    """Flatten captions and repeat each video embedding once per caption.

    ``video_index`` maps a flattened caption back to its position in the batch
    so the loss can average captions inside a video before averaging videos.
    """
    embeddings = []
    video_index = []
    texts = []
    for index, item in enumerate(batch):
        for caption in item["captions"]:
            embeddings.append(item["embedding"])
            video_index.append(index)
            texts.append(caption)
    input_ids, attention_mask = encode_captions(tokenizer, texts, max_length)
    return {
        "embeddings": torch.tensor(np.stack(embeddings), dtype=torch.float32),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "video_index": torch.tensor(video_index, dtype=torch.long),
        "video_ids": [int(item["video_id"]) for item in batch],
    }
