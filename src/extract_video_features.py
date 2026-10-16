"""Extract one feature vector per video from its captions.

Each caption is encoded with a Hugging Face encoder. Token states are
averaged, then the caption vectors for a video are averaged. One MATLAB
file is written per transformer layer:

    data/feature/video/<model>/layerXX.mat       feat: n_videos x n_units
    data/feature/norm_param/<model>/layerXX.mat  mu, sd, mu_val, sd_val, mu_tr, sd_tr

Row order follows video_id, so row i is video_id i + 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mind_captioning.encoding import extract_features
from mind_captioning.io import load_captions, load_stimuli, write_matlab
from mind_captioning.normalization import normalization_parameters

ROOT = Path(__file__).resolve().parents[1]


def model_dirname(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def layer_paths(directory: Path, n_layers: int) -> list[Path]:
    return [directory / f"layer{index:02d}.mat" for index in range(1, n_layers + 1)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode video captions and save per-layer MATLAB features.")
    parser.add_argument("--model", default="FacebookAI/roberta-base", help="Hugging Face encoder checkpoint")
    parser.add_argument("--captions", type=Path, default=ROOT / "data/caption/caption_ck20.csv")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/feature")
    parser.add_argument("--fmri-dir", type=Path, default=ROOT / "data/fmri/data/preprocessed")
    parser.add_argument("--subject", default="S1", help="Subject whose train and test stimulus ids define normalization")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--overwrite", action="store_true", help="Replace layer files that are already present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    captions, n_videos = load_captions(args.captions)
    counts = np.array([len(texts) for texts in captions])
    print(f"Loaded {int(counts.sum())} captions for {n_videos} videos from {args.captions}")

    train_path = args.fmri_dir / f"trainPerception_{args.subject}.mat"
    test_path = args.fmri_dir / f"testPerception_{args.subject}.mat"
    train_ids, train_runs = load_stimuli(train_path)
    test_ids, _ = load_stimuli(test_path)
    print(f"Normalization uses {len(np.unique(train_ids))} training videos and {len(np.unique(test_ids))} test videos from {args.subject}")

    dirname = model_dirname(args.model)
    video_dir = args.feature_root / "video" / dirname
    norm_dir = args.feature_root / "norm_param" / dirname
    from transformers import AutoConfig

    n_layers = int(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    targets = layer_paths(video_dir, n_layers) + layer_paths(norm_dir, n_layers)
    present = [path for path in targets if path.exists()]
    if len(present) == len(targets) and not args.overwrite:
        print(f"All {n_layers} layers already exist in {video_dir} and {norm_dir}")
        return
    if present and not args.overwrite:
        raise FileExistsError(
            f"{len(present)} of {len(targets)} layer files already exist under {video_dir} and {norm_dir}. Pass --overwrite to replace them."
        )
    features = extract_features(captions, args.model, args.batch_size, args.max_length, args.device)
    if features.shape[0] != n_layers:
        raise RuntimeError(f"Expected {n_layers} layers from {args.model}, got {features.shape[0]}")

    for layer_index in range(n_layers):
        layer_features = features[layer_index]
        parameters = normalization_parameters(layer_features, train_ids, train_runs, test_ids)
        write_matlab(video_dir / f"layer{layer_index + 1:02d}.mat", {"feat": layer_features})
        write_matlab(norm_dir / f"layer{layer_index + 1:02d}.mat", parameters)
    print(f"Wrote {n_layers} layers to {video_dir} and {norm_dir}")


if __name__ == "__main__":
    main()
