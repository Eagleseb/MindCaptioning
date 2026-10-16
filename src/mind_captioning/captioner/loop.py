"""Train the projector with video-level early stopping.

Validation loss selects the checkpoint. The test split is not read here.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mind_captioning.captioner.checkpoint import save_checkpoint
from mind_captioning.captioner.data import collate_captions
from mind_captioning.captioner.model import PrefixCaptioner, grouped_caption_loss


def set_seed(seed: int) -> None:
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        "embeddings": batch["embeddings"].to(device),
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "video_index": batch["video_index"].to(device),
        "video_ids": batch["video_ids"],
    }


def batch_loss(model: PrefixCaptioner, batch: dict) -> torch.Tensor:
    logits, labels = model(batch["embeddings"], batch["input_ids"], batch["attention_mask"])
    return grouped_caption_loss(logits, labels, batch["video_index"])


def trainable_parameters(model: PrefixCaptioner) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def run_epoch(
    model: PrefixCaptioner,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float = 1.0,
    progress: str | None = None,
    transform=None,
) -> float:
    training = optimizer is not None
    model.train(training)
    use_amp = training and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    total = 0.0
    count = 0
    batches = tqdm(loader, desc=progress, leave=False) if progress else loader
    for batch in batches:
        batch = move_batch(batch, device)
        if transform is not None:
            batch["embeddings"] = transform(batch["embeddings"], training)
        n_videos = len(batch["video_ids"])
        trainable = trainable_parameters(model)
        if training:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Training loss is not finite")
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
                optimizer.step()
        else:
            with torch.no_grad():
                loss = batch_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Evaluation loss is not finite")
        total += float(loss.detach()) * n_videos
        count += n_videos
    if count == 0:
        raise RuntimeError("The loader produced no videos")
    return total / count


def make_loader(
    dataset,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        collate_fn=lambda batch: collate_captions(batch, tokenizer, max_length),
    )


def fit(
    model: PrefixCaptioner,
    tokenizer,
    train_data,
    val_data,
    output_dir: Path,
    config: dict,
    split: dict[str, list[int]],
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    seed: int,
    device: torch.device,
    max_length: int,
    transform=None,
    lora_learning_rate: float | None = None,
) -> Path:
    """Train until validation loss stops improving and return the best checkpoint."""
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    train_loader = make_loader(train_data, tokenizer, max_length, batch_size, True, seed)
    val_loader = make_loader(val_data, tokenizer, max_length, batch_size, False, seed)
    projector_params = list(model.projector.parameters())
    projector_ids = {id(parameter) for parameter in projector_params}
    adapter_params = [parameter for parameter in trainable_parameters(model) if id(parameter) not in projector_ids]
    groups = [{"params": projector_params, "lr": learning_rate}]
    if adapter_params:
        groups.append({"params": adapter_params, "lr": learning_rate if lora_learning_rate is None else lora_learning_rate})
    optimizer = torch.optim.AdamW(groups, lr=learning_rate, weight_decay=weight_decay)
    best_loss = float("inf")
    waits = 0
    history: list[dict] = []
    best_dir = output_dir / "best"
    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
            progress=f"epoch {epoch} train",
            transform=transform,
        )
        val_loss = run_epoch(
            model, val_loader, device, progress=f"epoch {epoch} validation", transform=transform
        )
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(f"epoch {epoch}  train {train_loss:.4f}  validation {val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            waits = 0
            stored = dict(config)
            stored["best_validation_loss"] = best_loss
            stored["best_epoch"] = epoch
            save_checkpoint(best_dir, model, stored, split, save_decoder=bool(config.get("save_decoder")))
            print(f"Saved {best_dir} (validation loss {best_loss:.4f})")
        else:
            waits += 1
            if waits >= patience:
                print(f"Early stopping at epoch {epoch} after {patience} epochs without improvement")
                break
    if not best_dir.exists():
        raise RuntimeError("Training finished without a validation checkpoint")
    best_state = torch.load(best_dir / "projector.pt", map_location=device, weights_only=True)
    model.projector.load_state_dict(best_state)
    adapter_path = best_dir / "lora.pt"
    if adapter_path.exists():
        model.decoder.load_state_dict(torch.load(adapter_path, map_location=device, weights_only=True), strict=False)
    model.eval()
    return best_dir
