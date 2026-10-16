"""Generate captions from a projected prefix.

The prefix is passed as ``inputs_embeds``. Generation then uses the decoder's
cache: the first new token is predicted from the prefix, and later tokens are
predicted from the growing caption.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mind_captioning.captioner.model import PrefixCaptioner, context_length


def decode_caption(tokenizer, token_ids: list[int]) -> str:
    """Decode generated ids, stopping at the first end-of-sequence token."""
    eos = getattr(tokenizer, "eos_token_id", None)
    kept: list[int] = []
    for token_id in token_ids:
        token_id = int(token_id)
        if eos is not None and token_id == int(eos):
            break
        kept.append(token_id)
    return tokenizer.decode(kept, skip_special_tokens=True).strip()


def _as_batch(embedding: np.ndarray | torch.Tensor, embedding_dim: int) -> tuple[torch.Tensor, bool]:
    if isinstance(embedding, np.ndarray):
        tensor = torch.from_numpy(np.array(embedding, dtype=np.float32, copy=True))
    else:
        tensor = embedding.detach()
    single = tensor.ndim == 1
    if single:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2 or tensor.shape[1] != embedding_dim:
        raise ValueError(
            f"Expected embedding shape [{embedding_dim}] or [batch, {embedding_dim}], got {tuple(tensor.shape)}"
        )
    if not torch.isfinite(tensor).all():
        raise ValueError("The embedding contains non-finite values")
    return tensor, single


def generate_captions(
    model: PrefixCaptioner,
    tokenizer,
    embedding: np.ndarray | torch.Tensor,
    max_new_tokens: int = 64,
    num_beams: int = 1,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> list[str]:
    """Generate one caption per row of ``embedding``."""
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if num_beams < 1:
        raise ValueError("num_beams must be positive")
    if repetition_penalty <= 0:
        raise ValueError("repetition_penalty must be positive")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size must be non-negative")
    if do_sample and (temperature <= 0 or not 0 < top_p <= 1):
        raise ValueError("Sampling needs temperature > 0 and top_p in (0, 1]")
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is None:
        raise ValueError("The tokenizer needs an end-of-sequence id")
    pad = getattr(tokenizer, "pad_token_id", None)
    pad = int(eos) if pad is None else int(pad)

    values, _ = _as_batch(embedding, model.projector.embedding_dim)
    device = next(model.projector.parameters()).device
    dtype = next(model.projector.parameters()).dtype
    values = values.to(device=device, dtype=dtype)
    model.eval()
    prefix = model.projector(values)
    limit = context_length(model.decoder.config)
    if prefix.shape[1] + max_new_tokens > limit:
        raise ValueError(
            f"Prefix length {prefix.shape[1]} plus max_new_tokens {max_new_tokens} "
            f"exceeds the decoder context of {limit}"
        )
    attention_mask = torch.ones(prefix.shape[0], prefix.shape[1], dtype=torch.long, device=device)
    kwargs = {
        "inputs_embeds": prefix,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "num_beams": num_beams,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": pad,
        "eos_token_id": int(eos),
    }
    if no_repeat_ngram_size > 0:
        kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size
    if do_sample:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = top_p
    with torch.inference_mode():
        sequences = model.decoder.generate(**kwargs)
    return [decode_caption(tokenizer, row) for row in sequences.tolist()]


def generate_caption(
    model: PrefixCaptioner,
    tokenizer,
    embedding: np.ndarray | torch.Tensor,
    **kwargs,
) -> str | list[str]:
    """Generate a caption for one embedding, or a list when given a batch."""
    texts = generate_captions(model, tokenizer, embedding, **kwargs)
    single = not (isinstance(embedding, (np.ndarray, torch.Tensor)) and embedding.ndim == 2)
    if single:
        return texts[0]
    return texts


def load_embedding_file(path: Path) -> np.ndarray:
    """Load one vector or a batch from ``.npy``, ``.npz``, or ``.pt``."""
    if path.suffix == ".npy":
        array = np.load(path)
    elif path.suffix == ".npz":
        with np.load(path) as handle:
            for key in ("embedding", "feat", "predicted"):
                if key in handle:
                    array = np.array(handle[key])
                    break
            else:
                raise KeyError(f"{path} needs an embedding, feat, or predicted array")
    elif path.suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, dict):
            for key in ("embedding", "feat", "predicted"):
                if key in loaded:
                    loaded = loaded[key]
                    break
            else:
                raise KeyError(f"{path} needs an embedding, feat, or predicted tensor")
        array = loaded.detach().cpu().numpy() if isinstance(loaded, torch.Tensor) else np.asarray(loaded)
    else:
        raise ValueError(f"Unsupported embedding file {path}. Use .npy, .npz, or .pt")
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim == 2 and array.shape[0] == 1:
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"{path} has shape {array.shape}; expected a vector or a batch of vectors")
