"""Tiny offline tokenizer and GPT-2 used by the captioner tests."""

from __future__ import annotations

import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel

from mind_captioning.captioner.model import FeatureProjector, PrefixCaptioner


class TinyTokenizer:
    """Word tokenizer with a separate end token and no hub access."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.bos_token = "<bos>"
        self.eos_token = "<eos>"
        self.name_or_path = "tiny"
        self._token_to_id = {"<pad>": 0, "<bos>": 1, "<eos>": 2}
        self._id_to_token = {0: "<pad>", 1: "<bos>", 2: "<eos>"}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("Special tokens are added by encode_captions")
        ids = []
        for word in text.split():
            if word not in self._token_to_id:
                index = len(self._token_to_id)
                self._token_to_id[word] = index
                self._id_to_token[index] = word
            ids.append(self._token_to_id[word])
        return ids

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        words = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id in specials:
                continue
            words.append(self._id_to_token.get(token_id, f"t{token_id}"))
        return " ".join(words)


def tiny_decoder(vocab_size: int = 32, hidden: int = 16) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=64,
        n_embd=hidden,
        n_layer=1,
        n_head=4,
        n_inner=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
    )
    decoder = GPT2LMHeadModel(config)
    decoder.eval()
    return decoder


def tiny_captioner(
    embedding_dim: int = 8,
    prefix_tokens: int = 2,
    hidden: int = 16,
    vocab_size: int = 32,
) -> PrefixCaptioner:
    decoder = tiny_decoder(vocab_size=vocab_size, hidden=hidden)
    projector = FeatureProjector(
        embedding_dim=embedding_dim,
        decoder_hidden_size=hidden,
        num_prefix_tokens=prefix_tokens,
        hidden_dim=32,
    )
    return PrefixCaptioner(projector, decoder)


def force_token(decoder: GPT2LMHeadModel, token_id: int) -> None:
    """Replace the output layer so greedy decoding emits one content token."""

    class Fixed(nn.Module):
        def __init__(self, vocab: int, chosen: int) -> None:
            super().__init__()
            self.bias = nn.Parameter(torch.full((vocab,), -100.0))
            self.bias.data[chosen] = 100.0

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return self.bias.expand(*hidden_states.shape[:-1], self.bias.shape[0])

    decoder.lm_head = Fixed(decoder.config.vocab_size, token_id)
