"""Score one checkpoint on S1 test-perception brain features.

Every condition uses the checkpoint's saved input normalization. Geometric
norm matching, used by the original raw-feature baseline, rescales a decoded
vector to the training-feature norm before that normalization.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from mind_captioning.captioner.data import VideoCaptionDataset, VideoExample
from mind_captioning.captioner.generate import generate_captions
from mind_captioning.captioner.loop import make_loader, run_epoch
from mind_captioning.captioner.metrics import rouge_l_max
from train_decoder import average_by_video


def apply_norm(values: np.ndarray, arrays: dict | None) -> np.ndarray:
    if arrays is None:
        return np.asarray(values, dtype=np.float32)
    return ((values - arrays["mean"]) / arrays["scale"]).astype(np.float32)


def geometric_match(decoded: np.ndarray, train_raw: np.ndarray) -> np.ndarray:
    center = train_raw.mean(axis=0)
    reference = np.linalg.norm(train_raw - center, axis=1).mean()
    centered = decoded - center
    norms = np.maximum(np.linalg.norm(centered, axis=1), 1e-8)
    return (center + centered * (reference / norms)[:, None]).astype(np.float32)


def _score_condition(
    model,
    tokenizer,
    features: np.ndarray,
    video_ids: np.ndarray,
    references: dict[int, tuple[str, ...]],
    test_ids: np.ndarray,
    device,
    max_length: int,
    batch_size: int,
    seed: int,
    generate_kwargs: dict,
) -> dict:
    examples = [
        VideoExample(int(video_id), np.asarray(vector, dtype=np.float32), references[int(video_id)])
        for vector, video_id in zip(features, video_ids)
    ]
    loader = make_loader(VideoCaptionDataset(examples), tokenizer, max_length, 8, False, seed)
    loss = run_epoch(model, loader, device)
    texts: list[str] = []
    for start in range(0, len(features), batch_size):
        texts.extend(generate_captions(model, tokenizer, features[start : start + batch_size], **generate_kwargs))
    column = {int(video_id): index for index, video_id in enumerate(test_ids)}
    cache: dict[str, np.ndarray] = {}
    rows = []
    for text in texts:
        if text not in cache:
            cache[text] = np.array([rouge_l_max(text, references[int(video_id)]) for video_id in test_ids])
        rows.append(cache[text])
    scores = np.vstack(rows)
    own_columns = np.array([column[int(video_id)] for video_id in video_ids])
    own = scores[np.arange(len(texts)), own_columns]
    others = (scores.sum(axis=1) - own) / (scores.shape[1] - 1)
    higher = (scores > own[:, None]).sum(axis=1)
    ties = (scores == own[:, None]).sum(axis=1) - 1
    ranks = 1 + higher + 0.5 * ties
    return {
        "loss": loss,
        "perplexity": float(np.exp(loss)),
        "rouge_own": float(own.mean()),
        "rouge_other": float(others.mean()),
        "accuracy": float((ranks == 1).mean()),
        "median_rank": float(np.median(ranks)),
        "distinct": len(set(texts)) / len(texts),
        "n": len(texts),
        "captions": [
            {"video_id": int(video_id), "caption": text, "rouge_l": float(score)}
            for video_id, text, score in zip(video_ids, texts, own)
        ],
    }


def evaluate_brain(
    model,
    tokenizer,
    config: dict,
    split: dict,
    library,
    ridge_predicted: np.ndarray,
    ridge_video_ids: np.ndarray,
    device,
    output_dir: Path,
    max_new_tokens: int = 40,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    batch_size: int = 16,
    seed: int = 0,
) -> dict:
    from mind_captioning.captioner.data import examples_for_ids

    arrays = config.get("input_norm_arrays")
    test_ids = np.unique(ridge_video_ids.astype(np.int64))
    if set(int(video_id) for video_id in test_ids) != set(split["test"]):
        raise ValueError("Ridge test videos differ from the captioner split")
    decoded = average_by_video(ridge_predicted.astype(np.float32), ridge_video_ids.astype(np.int64)).astype(np.float32)
    test_examples = examples_for_ids(library, test_ids)
    train_examples = examples_for_ids(library, split["train"])
    references = {example.video_id: example.captions for example in test_examples}
    true_raw = np.stack([example.embedding for example in test_examples])
    train_raw = np.stack([example.embedding for example in train_examples])
    matched = geometric_match(decoded, train_raw)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(test_ids))
    while np.any(permutation == np.arange(len(test_ids))):
        permutation = rng.permutation(len(test_ids))
    conditions_in = {
        "caption_features": (apply_norm(true_raw, arrays), test_ids),
        "brain_averaged": (apply_norm(decoded, arrays), test_ids),
        "brain_norm_matched": (apply_norm(matched, arrays), test_ids),
        "shuffled": (apply_norm(decoded[permutation], arrays), test_ids),
        "mean_training": (
            apply_norm(np.repeat(train_raw.mean(axis=0, keepdims=True), len(test_ids), axis=0), arrays),
            test_ids,
        ),
    }
    generate_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "do_sample": False,
    }
    conditions = {}
    for name, (features, video_ids) in conditions_in.items():
        print(f"Scoring {name}")
        conditions[name] = _score_condition(
            model,
            tokenizer,
            features,
            video_ids,
            references,
            test_ids,
            device,
            int(config["max_length"]),
            batch_size,
            seed,
            generate_kwargs,
        )
    public = {}
    captions = {}
    for name, row in conditions.items():
        captions[name] = row.pop("captions")
        public[name] = row
    summary = {
        "normalization": config.get("normalization", "none"),
        "decoding": generate_kwargs,
        "conditions": public,
        "captions": captions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def comparison_condition(summary: dict) -> dict:
    """Brain condition used to decide whether a run beat the raw-feature baseline.

    Z-scored runs are judged on decoded vectors passed through that same
    z-score. Raw-feature runs are judged on the geometrically norm-matched
    decoded vectors.
    """
    name = "brain_averaged" if summary.get("normalization") == "zscore" else "brain_norm_matched"
    row = dict(summary["conditions"][name])
    row["condition"] = name
    return row


def beats_baseline(summary: dict, baseline: dict) -> bool:
    candidate = comparison_condition(summary)
    reference = comparison_condition(baseline)
    shuffled = summary["conditions"]["shuffled"]["rouge_own"]
    return (
        candidate["rouge_own"] > reference["rouge_own"]
        and candidate["accuracy"] > reference["accuracy"]
        and candidate["rouge_own"] > shuffled
    )
