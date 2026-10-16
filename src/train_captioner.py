"""Train a captioner on cached DeBERTa video embeddings.

The embedding of a video is the masked mean of its reference captions, averaged
across those captions. The model learns to reconstruct a caption from that
vector. It does not see pixels or brain activity, and the 72 test-perception
videos are held out from training and from early stopping.

Validation videos are 10% of the train-perception videos, chosen with seed 42.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mind_captioning.captioner import LIMITATION, POOLING
from mind_captioning.captioner.checkpoint import hub_revisions, load_hub_decoder
from mind_captioning.captioner.data import (
    VideoCaptionDataset,
    examples_for_ids,
    excluded_video_ids,
    feature_matrix_path,
    load_examples,
    make_split,
    model_dirname,
    perception_video_ids,
)
from mind_captioning.captioner.loop import fit, set_seed
from mind_captioning.captioner.model import (
    FeatureProjector,
    PrefixCaptioner,
    attach_lora,
    context_length,
    hidden_size,
)
from mind_captioning.captioner.neighborhood import (
    CellAugmentedDataset,
    fit_cell_geometry,
    probe_examples,
)
from mind_captioning.device import pick_device

ROOT = Path(__file__).resolve().parents[1]


def input_transform(train_examples, kind: str, shrink_min: float, shrink_max: float, noise_std: float, device):
    """Return a batch transform and the arrays that must be saved with the checkpoint.

    Z-scoring uses the training videos only. Shrinkage and noise are applied
    on training batches, after that z-score, and are absent from validation.
    """
    arrays = None
    mean = scale = None
    if kind == "zscore":
        values = np.stack([example.embedding for example in train_examples]).astype(np.float64)
        raw_mean = values.mean(axis=0)
        raw_scale = values.std(axis=0, ddof=1)
        raw_scale = np.where(raw_scale > 0, raw_scale, 1.0)
        arrays = {"mean": raw_mean.astype(np.float32), "scale": raw_scale.astype(np.float32)}
        mean = torch.tensor(arrays["mean"], device=device)
        scale = torch.tensor(arrays["scale"], device=device)
    use_noise = shrink_min < shrink_max or noise_std > 0

    def transform(embeddings: torch.Tensor, training: bool) -> torch.Tensor:
        values = embeddings
        if mean is not None:
            values = (values - mean) / scale
        if training and use_noise:
            draw = torch.empty(values.shape[0], 1, device=values.device, dtype=values.dtype)
            draw.uniform_(shrink_min, shrink_max)
            values = values * draw
            if noise_std > 0:
                values = values + torch.randn_like(values) * noise_std
        return values

    return (transform if kind == "zscore" or use_noise else None), arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-model", default="deberta-large", help="Cached feature directory or model id")
    parser.add_argument("--layer", type=int, help="1-based feature layer. Defaults to the last saved layer")
    parser.add_argument("--decoder", default="distilgpt2", help="Pretrained causal language model")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/caption/caption_ck20.csv")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/feature")
    parser.add_argument("--fmri-dir", type=Path, default=ROOT / "data/fmri/data/preprocessed")
    parser.add_argument("--subject", default="S1", help="Subject whose perception stimulus ids define the split")
    parser.add_argument("--captions-per-video", type=int, default=20)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8, help="Videos per batch")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--prefix-tokens", type=int, default=8)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--max-length", type=int, default=64, help="Caption tokens, including the end token")
    parser.add_argument("--caption-mode", choices=("all_captions", "sample_caption"), default="all_captions")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, help="Checkpoint directory. Defaults under res/captioner")
    parser.add_argument("--run-id", help="Name recorded in the checkpoint config")
    parser.add_argument("--normalize", choices=("none", "zscore"), default="none")
    parser.add_argument("--shrink-min", type=float, default=1.0, help="Lower draw for training-time feature shrinkage")
    parser.add_argument("--shrink-max", type=float, default=1.0, help="Upper draw for training-time feature shrinkage")
    parser.add_argument("--noise-std", type=float, default=0.0, help="Gaussian noise added to normalized training inputs")
    parser.add_argument("--lora-rank", type=int, default=0, help="LoRA rank. Zero leaves the decoder frozen")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-layers", type=int, default=2, help="How many final decoder blocks receive LoRA")
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument(
        "--cell-augmentation",
        action="store_true",
        help="Train on clean and Voronoi-cell-preserving angular/radial probes",
    )
    parser.add_argument("--cell-clean-probability", type=float, default=0.25)
    parser.add_argument("--cell-max-fraction", type=float, default=0.9)
    parser.add_argument("--cell-validation-fraction", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing best checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.epochs < 1 or args.patience < 1:
        raise ValueError("--batch-size, --epochs, and --patience must be positive")
    if args.lr <= 0 or args.prefix_tokens < 1 or args.projector_hidden < 1 or args.max_length < 2:
        raise ValueError("Learning rate, prefix length, hidden size, and max length must be valid")
    if args.grad_clip < 0 or args.noise_std < 0 or args.lora_rank < 0:
        raise ValueError("--grad-clip, --noise-std, and --lora-rank must be non-negative")
    if not 0 < args.shrink_min <= args.shrink_max <= 1:
        raise ValueError("--shrink-min and --shrink-max must satisfy 0 < min <= max <= 1")
    if args.cell_augmentation:
        if args.normalize != "none" or args.noise_std or args.shrink_min != 1 or args.shrink_max != 1:
            raise ValueError("--cell-augmentation is a raw-feature experiment; do not combine it with normalization/noise")
        if not 0 <= args.cell_clean_probability <= 1:
            raise ValueError("--cell-clean-probability must be in [0, 1]")
        if not 0 < args.cell_max_fraction < 1 or not 0 < args.cell_validation_fraction < 1:
            raise ValueError("Cell fractions must be in (0, 1)")

    feature_path, layer = feature_matrix_path(args.feature_root, args.feature_model, args.layer)
    library = load_examples(args.captions, feature_path, args.captions_per_video)
    train_ids, test_ids = perception_video_ids(args.fmri_dir, args.subject)
    split = make_split(train_ids, test_ids, args.val_fraction, args.seed)
    split["excluded"] = excluded_video_ids(len(library), split)
    output = args.output
    if output is None:
        output = (
            ROOT
            / "res"
            / "captioner"
            / model_dirname(args.decoder)
            / model_dirname(args.feature_model)
            / f"layer{layer:02d}"
        )
    best_weights = output / "best" / "projector.pt"
    if best_weights.exists() and not args.overwrite:
        raise FileExistsError(f"{best_weights} already exists. Pass --overwrite to replace it")

    print(LIMITATION)
    print(
        f"Library: {len(library)} videos, {args.captions_per_video} captions, "
        f"{library[0].embedding.shape[0]} feature units from {feature_path}"
    )
    print(
        f"Split: {len(split['train'])} train, {len(split['validation'])} validation, "
        f"{len(split['test'])} test, {len(split['excluded'])} excluded"
    )
    print(f"Pooling: {POOLING}. Normalization: {args.normalize}.")

    set_seed(args.seed)
    device = pick_device(args.device)
    decoder, tokenizer = load_hub_decoder(args.decoder, device)
    decoder_revision, tokenizer_revision = hub_revisions(decoder, tokenizer)
    width = hidden_size(decoder.config)
    if args.prefix_tokens + args.max_length > context_length(decoder.config):
        raise ValueError(
            f"Prefix tokens plus --max-length exceed the decoder context ({context_length(decoder.config)})"
        )
    projector = FeatureProjector(
        embedding_dim=int(library[0].embedding.shape[0]),
        decoder_hidden_size=width,
        num_prefix_tokens=args.prefix_tokens,
        hidden_dim=args.projector_hidden,
    )
    model = PrefixCaptioner(projector, decoder).to(device)
    lora = None
    if args.lora_rank:
        adapted = attach_lora(decoder, args.lora_rank, args.lora_alpha, args.lora_layers)
        lora = {"rank": args.lora_rank, "alpha": args.lora_alpha, "layers": adapted}
        print(f"LoRA: rank {args.lora_rank}, alpha {args.lora_alpha}, last {adapted} blocks, lr {args.lora_lr:g}")
    model.to(device)
    train_examples = examples_for_ids(library, split["train"])
    transform, norm_arrays = input_transform(
        train_examples,
        args.normalize,
        args.shrink_min,
        args.shrink_max,
        args.noise_std,
        device,
    )
    validation_examples = examples_for_ids(library, split["validation"])
    cell_summary = None
    if args.cell_augmentation:
        print("Fitting train and validation cell geometry.")
        train_geometry = fit_cell_geometry(train_examples)
        validation_geometry = fit_cell_geometry(validation_examples)
        train_data = CellAugmentedDataset(
            train_examples,
            train_geometry,
            mode=args.caption_mode,
            seed=args.seed,
            clean_probability=args.cell_clean_probability,
            max_fraction=args.cell_max_fraction,
        )
        validation_probes = probe_examples(
            validation_examples,
            validation_geometry,
            args.cell_validation_fraction,
        )
        val_data = VideoCaptionDataset(validation_probes, mode="all_captions", seed=args.seed)
        cell_summary = {
            "clean_probability": args.cell_clean_probability,
            "max_fraction": args.cell_max_fraction,
            "validation_fraction": args.cell_validation_fraction,
            "train_nearest_distance_mean": float(train_geometry.nearest_distance.mean()),
            "train_nearest_distance_median": float(np.median(train_geometry.nearest_distance)),
        }
        print(
            f"Cell distances: mean {cell_summary['train_nearest_distance_mean']:.3f}, "
            f"median {cell_summary['train_nearest_distance_median']:.3f}; "
            f"validation has {len(validation_probes)} probes."
        )
    else:
        train_data = VideoCaptionDataset(
            train_examples,
            mode=args.caption_mode,
            seed=args.seed,
        )
        val_data = VideoCaptionDataset(
            validation_examples,
            mode="all_captions",
            seed=args.seed,
        )
    config = {
        "feature_model": model_dirname(args.feature_model),
        "layer": layer,
        "feature_path": str(feature_path),
        "decoder_name": args.decoder,
        "decoder_revision": decoder_revision,
        "tokenizer_name": tokenizer.name_or_path,
        "tokenizer_revision": tokenizer_revision,
        "captions_per_video": args.captions_per_video,
        "caption_mode": args.caption_mode,
        "max_length": args.max_length,
        "seed": args.seed,
        "subject": args.subject,
        "run_id": args.run_id,
        "save_decoder": False,
        "normalization": args.normalize,
        "input_norm_arrays": norm_arrays,
        "lora": lora,
        "cell_augmentation": cell_summary,
        "hyperparameters": {
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "patience": args.patience,
            "grad_clip": args.grad_clip,
            "val_fraction": args.val_fraction,
            "prefix_tokens": args.prefix_tokens,
            "projector_hidden": args.projector_hidden,
            "normalize": args.normalize,
            "shrink_min": args.shrink_min,
            "shrink_max": args.shrink_max,
            "noise_std": args.noise_std,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_layers": args.lora_layers,
            "lora_lr": args.lora_lr,
            "cell_augmentation": args.cell_augmentation,
            "cell_clean_probability": args.cell_clean_probability,
            "cell_max_fraction": args.cell_max_fraction,
            "cell_validation_fraction": args.cell_validation_fraction,
        },
    }
    print(
        f"Decoder: {args.decoder} ({width} units, frozen) on {device}. "
        f"Projector: {args.prefix_tokens} prefix tokens, hidden {args.projector_hidden}."
    )
    best = fit(
        model,
        tokenizer,
        train_data,
        val_data,
        output,
        config,
        split,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        seed=args.seed,
        device=device,
        max_length=args.max_length,
        transform=transform,
        lora_learning_rate=args.lora_lr if args.lora_rank else None,
    )
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
