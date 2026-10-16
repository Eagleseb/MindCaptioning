"""Save and load a projector checkpoint.

Frozen hub decoders are reloaded from their repository. A checkpoint records
the identifiers needed to do that, plus the projector, the video split, and
the training settings. Decoder weights are stored only when the model cannot
be retrieved from a hub id, which is the case for the randomly initialized
models used in tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from mind_captioning.captioner import LIMITATION, NORMALIZATION, POOLING
from mind_captioning.captioner.model import FeatureProjector, PrefixCaptioner, hidden_size


def save_checkpoint(
    directory: Path,
    model: PrefixCaptioner,
    config: dict,
    split: dict[str, list[int]],
    save_decoder: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    stored = dict(config)
    stored["pooling"] = POOLING
    stored.setdefault("normalization", NORMALIZATION)
    stored["limitation"] = LIMITATION
    arrays = stored.pop("input_norm_arrays", None)
    if arrays is not None:
        import numpy as np

        np.savez(directory / "input_norm.npz", mean=arrays["mean"], scale=arrays["scale"])
    stored["embedding_dim"] = model.projector.embedding_dim
    stored["decoder_hidden_size"] = model.projector.decoder_hidden_size
    stored["num_prefix_tokens"] = model.projector.num_prefix_tokens
    stored["projector_hidden_dim"] = model.projector.hidden_dim
    if save_decoder:
        stored["decoder_origin"] = "weights"
        stored["decoder_config"] = model.decoder.config.to_dict()
        torch.save(model.decoder.state_dict(), directory / "decoder.pt")
    else:
        stored["decoder_origin"] = "hub"
    from mind_captioning.captioner.model import lora_state_dict

    adapters = lora_state_dict(model.decoder)
    if adapters:
        torch.save(adapters, directory / "lora.pt")
    torch.save(model.projector.state_dict(), directory / "projector.pt")
    (directory / "config.json").write_text(json.dumps(stored, indent=2) + "\n")
    (directory / "split.json").write_text(json.dumps(split, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_checkpoint(directory: Path, device: torch.device, tokenizer=None):
    """Rebuild the captioner.

    Hub checkpoints load the decoder and tokenizer from ``decoder_name``.
    Weight checkpoints rebuild the decoder from the saved config and require
    the caller to pass the matching tokenizer.
    """
    config = read_json(directory / "config.json")
    origin = config.get("decoder_origin", "hub")
    if origin == "hub":
        decoder, tokenizer = load_hub_decoder(
            config["decoder_name"],
            device,
            revision=config.get("decoder_revision"),
        )
    elif origin == "weights":
        if tokenizer is None:
            raise ValueError(f"{directory} stores decoder weights and needs the training tokenizer")
        from transformers import AutoModelForCausalLM
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        payload = config["decoder_config"]
        decoder_config = CONFIG_MAPPING[payload["model_type"]].from_dict(payload)
        decoder = AutoModelForCausalLM.from_config(decoder_config)
    else:
        raise ValueError(f"Unknown decoder_origin {origin!r} in {directory}")

    if hidden_size(decoder.config) != int(config["decoder_hidden_size"]):
        raise ValueError(
            f"Loaded decoder hidden size {hidden_size(decoder.config)} "
            f"does not match the checkpoint ({config['decoder_hidden_size']})"
        )
    lora = config.get("lora")
    if lora:
        from mind_captioning.captioner.model import attach_lora

        attach_lora(decoder, int(lora["rank"]), float(lora["alpha"]), int(lora["layers"]))
    if origin == "weights":
        state = torch.load(directory / "decoder.pt", map_location="cpu", weights_only=True)
        decoder.load_state_dict(state)
        if hasattr(decoder, "tie_weights"):
            decoder.tie_weights()
    adapter_path = directory / "lora.pt"
    if lora and adapter_path.exists():
        decoder.load_state_dict(torch.load(adapter_path, map_location="cpu", weights_only=True), strict=False)
    decoder.to(device)
    projector = FeatureProjector(
        embedding_dim=int(config["embedding_dim"]),
        decoder_hidden_size=int(config["decoder_hidden_size"]),
        num_prefix_tokens=int(config["num_prefix_tokens"]),
        hidden_dim=int(config["projector_hidden_dim"]),
    )
    state = torch.load(directory / "projector.pt", map_location="cpu", weights_only=True)
    projector.load_state_dict(state)
    model = PrefixCaptioner(projector, decoder)
    model.to(device)
    model.eval()
    norm_path = directory / "input_norm.npz"
    if norm_path.exists():
        import numpy as np

        with np.load(norm_path) as handle:
            config["input_norm_arrays"] = {
                "mean": np.array(handle["mean"], dtype=np.float32),
                "scale": np.array(handle["scale"], dtype=np.float32),
            }
    split = read_json(directory / "split.json")
    return model, tokenizer, config, split


def load_hub_decoder(name: str, device: torch.device, revision: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"revision": revision} if revision else {}
    tokenizer = AutoTokenizer.from_pretrained(name, **kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    decoder = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if tokenizer.pad_token_id is not None:
        decoder.config.pad_token_id = tokenizer.pad_token_id
        decoder.generation_config.pad_token_id = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        decoder.generation_config.eos_token_id = tokenizer.eos_token_id
    decoder.to(device)
    decoder.eval()
    return decoder, tokenizer


def hub_revisions(decoder, tokenizer) -> tuple[str | None, str | None]:
    decoder_revision = getattr(decoder.config, "_commit_hash", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", None) or {}
    tokenizer_revision = getattr(tokenizer, "_commit_hash", None) or init_kwargs.get("revision")
    return decoder_revision, tokenizer_revision
