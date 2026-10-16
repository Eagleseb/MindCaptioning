"""Encode captions into one feature vector per video and layer."""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from .device import pick_device


def extract_features(
    captions: list[list[str]],
    model_id: str,
    batch_size: int,
    max_length: int,
    device_name: str,
) -> np.ndarray:
    """Return features with shape (n_layers, n_videos, hidden_size)."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = pick_device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    model.to(device)
    n_layers = int(model.config.num_hidden_layers)
    hidden_size = int(model.config.hidden_size)
    n_videos = len(captions)
    sums = np.zeros((n_layers, n_videos, hidden_size), dtype=np.float64)
    counts = np.zeros(n_videos, dtype=np.int64)
    special_ids = set(tokenizer.all_special_ids)

    samples: list[tuple[int, str]] = []
    for video_index, video_captions in enumerate(captions):
        samples.extend((video_index, text) for text in video_captions)
    if not samples:
        raise ValueError("No captions to encode")

    print(
        f"Encoding {len(samples)} captions for {n_videos} videos "
        f"with {model_id} on {device} ({n_layers} layers, {hidden_size} units)"
    )
    n_batches = (len(samples) + batch_size - 1) // batch_size
    for batch_index in tqdm(range(n_batches), desc="Encoding batches"):
        batch = samples[batch_index * batch_size : (batch_index + 1) * batch_size]
        video_ids = np.array([video_index for video_index, _ in batch], dtype=np.int64)
        texts = [text for _, text in batch]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != n_layers + 1:
            found = 0 if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"{model_id} returned {found} hidden states; expected {n_layers + 1} including embeddings"
            )
        keep = encoded["attention_mask"].bool()
        for token_id in special_ids:
            keep &= encoded["input_ids"].ne(token_id)
        valid = keep.any(dim=1).detach().cpu().numpy()
        weights = keep.unsqueeze(-1)
        for layer_index, states in enumerate(hidden_states[1:]):
            pooled = (states * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
            vectors = pooled.detach().float().cpu().numpy()
            selected = video_ids[valid]
            np.add.at(sums[layer_index], selected, vectors[valid])
        np.add.at(counts, video_ids[valid], 1)

    missing = np.flatnonzero(counts == 0) + 1
    if missing.size:
        preview = ", ".join(str(video_id) for video_id in missing[:10])
        raise ValueError(
            f"{missing.size} videos have no usable captions, including {preview}"
        )
    return sums / counts[None, :, None]
