"""Prefix projector and frozen causal decoder."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

IGNORE_INDEX = -100


class FeatureProjector(nn.Module):
    """Map a caption embedding to a short sequence of decoder vectors.

    ``forward`` accepts ``[batch, embedding_dim]`` and returns
    ``[batch, num_prefix_tokens, decoder_hidden_size]``.
    """

    def __init__(
        self,
        embedding_dim: int,
        decoder_hidden_size: int,
        num_prefix_tokens: int = 8,
        hidden_dim: int = 1024,
    ) -> None:
        super().__init__()
        if min(embedding_dim, decoder_hidden_size, num_prefix_tokens, hidden_dim) < 1:
            raise ValueError("Projector sizes must be positive")
        self.embedding_dim = int(embedding_dim)
        self.decoder_hidden_size = int(decoder_hidden_size)
        self.num_prefix_tokens = int(num_prefix_tokens)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(
                self.hidden_dim, self.num_prefix_tokens * self.decoder_hidden_size
            ),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        if embedding.ndim != 2 or embedding.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected embedding shape [batch, {self.embedding_dim}], got {tuple(embedding.shape)}"
            )
        flat = self.net(embedding)
        return flat.reshape(
            embedding.shape[0], self.num_prefix_tokens, self.decoder_hidden_size
        )


class LoRAConv1D(nn.Module):
    """Low-rank update added to a frozen GPT-2 Conv1D projection."""

    def __init__(self, base: nn.Module, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not hasattr(base, "nx") or not hasattr(base, "nf"):
            raise TypeError("LoRA wrapping expects a GPT-2 Conv1D projection")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.scale = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.randn(base.nx, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, base.nf))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        update = hidden.reshape(-1, hidden.size(-1)) @ self.lora_A @ self.lora_B
        return self.base(hidden) + (self.scale * update).view(*hidden.shape[:-1], -1)


def attach_lora(decoder: nn.Module, rank: int, alpha: float, layers: int) -> int:
    """Add LoRA to attention and MLP projections in the last decoder blocks.

    Returns the number of adapted blocks. The base projection stays frozen.
    """
    blocks = decoder.transformer.h
    if layers < 1 or layers > len(blocks):
        raise ValueError(f"LoRA layers must be in 1..{len(blocks)}")
    adapted = 0
    for block in blocks[-layers:]:
        for parent, names in (
            (block.attn, ("c_attn", "c_proj")),
            (block.mlp, ("c_fc", "c_proj")),
        ):
            for name in names:
                module = getattr(parent, name)
                if not isinstance(module, LoRAConv1D):
                    setattr(parent, name, LoRAConv1D(module, rank, alpha))
        adapted += 1
    return adapted


def lora_parameters(module: nn.Module) -> list[nn.Parameter]:
    return [
        parameter
        for name, parameter in module.named_parameters()
        if "lora_" in name and parameter.requires_grad
    ]


def lora_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in module.named_parameters()
        if "lora_" in name
    }


def context_length(config) -> int:
    for name in ("n_positions", "max_position_embeddings"):
        value = getattr(config, name, None)
        if value:
            return int(value)
    return 10**9


def hidden_size(config) -> int:
    value = getattr(config, "hidden_size", None)
    if value is None:
        value = getattr(config, "n_embd", None)
    if not value:
        raise ValueError("The decoder config has no hidden size")
    return int(value)


class PrefixCaptioner(nn.Module):
    """Condition a frozen causal language model on projected prefix tokens.

    Decoder dropout stays off. Only the projector is trained.
    """

    def __init__(self, projector: FeatureProjector, decoder: nn.Module) -> None:
        super().__init__()
        self.projector = projector
        self.decoder = decoder
        self.freeze_decoder()

    def freeze_decoder(self) -> None:
        self.decoder.eval()
        for name, parameter in self.decoder.named_parameters():
            parameter.requires_grad_("lora_" in name)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def forward(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prefix = self.projector(embeddings)
        if prefix.shape[0] != input_ids.shape[0]:
            raise ValueError(
                f"Prefix batch {prefix.shape[0]} does not match captions {input_ids.shape[0]}"
            )
        token_embeddings = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix, token_embeddings], dim=1)
        prefix_mask = attention_mask.new_ones(prefix.shape[0], prefix.shape[1])
        full_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_mask,
            use_cache=False,
        )
        labels = build_labels(input_ids, attention_mask, prefix.shape[1])
        return outputs.logits, labels


def build_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prefix_length: int,
) -> torch.Tensor:
    """Labels aligned with ``prefix + caption`` positions.

    Prefix positions and padding are ``-100``. Caption tokens, including the
    end-of-sequence token, keep their ids. The language-model shift is applied
    in ``grouped_caption_loss``: position ``t`` predicts ``labels[t + 1]``.
    """
    token_labels = input_ids.masked_fill(attention_mask.eq(0), IGNORE_INDEX)
    prefix = input_ids.new_full((input_ids.shape[0], prefix_length), IGNORE_INDEX)
    return torch.cat([prefix, token_labels], dim=1)


def mean_by_video(per_caption: torch.Tensor, video_index: torch.Tensor) -> torch.Tensor:
    """Average caption losses within each video, then across videos."""
    if video_index.ndim != 1 or video_index.shape[0] != per_caption.shape[0]:
        raise ValueError(
            f"video_index shape {tuple(video_index.shape)} does not match {per_caption.shape[0]} captions"
        )
    video_index = video_index.to(device=per_caption.device, dtype=torch.long)
    n_videos = int(video_index.max().item()) + 1
    present = torch.zeros(n_videos, dtype=torch.bool, device=per_caption.device)
    present[video_index] = True
    if not bool(present.all()):
        raise ValueError(
            "video_index skips a video, so the batch cannot give every video equal weight"
        )
    totals = per_caption.new_zeros(n_videos)
    counts = per_caption.new_zeros(n_videos)
    totals.index_add_(0, video_index, per_caption)
    counts.index_add_(0, video_index, torch.ones_like(per_caption))
    return (totals / counts).mean()


def grouped_caption_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    video_index: torch.Tensor,
) -> torch.Tensor:
    """Length-normalized token loss, averaged within videos and then across videos.

    The shift makes ``logits[:, t]`` predict ``labels[:, t + 1]``. Ignored
    labels (prefix and padding) contribute nothing. A caption with no remaining
    targets is an error rather than a silent zero.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(shift_labels)
    valid = shift_labels.ne(IGNORE_INDEX)
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("A caption has no prediction targets")
    per_caption = (per_token * valid).sum(dim=1) / counts.to(per_token.dtype)
    return mean_by_video(per_caption, video_index)
