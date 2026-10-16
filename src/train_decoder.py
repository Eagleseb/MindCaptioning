"""Learn one subject's linear map from fMRI activity to caption features.

The script loads perception runs for one subject, pairs each sample with the
caption features of the video that was shown, and z-scores both sides with
the training mean and unbiased standard deviation. The ridge weights
themselves are left unimplemented in fit_ridge.

    brain activity  --W-->  caption features

Run folds follow the paper: runs 1-10, 11-20, 21-30, 31-40, 41-50, and 51-58.
Pass --select-alpha to choose the penalty on that split. The search grid is
logspace(1, 6, 10).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mind_captioning.io import load_brain_samples, load_video_features
from mind_captioning.normalization import RUN_FOLDS, mean_and_sd

ROOT = Path(__file__).resolve().parents[1]
# Penalties used for the decoding models in the paper.
ALPHAS = np.logspace(1, 6, 10)


def fit_ridge(brain: np.ndarray, text: np.ndarray, alpha: float) -> np.ndarray:
    """Learn the linear map from z-scored brain activity to z-scored text features.

    brain: (n_samples, n_voxels), zero mean and unit scale per voxel.
    text: (n_samples, n_units), zero mean and unit scale per feature unit.
    Return weights of shape (n_voxels + 1, n_units). The last row is the
    intercept. predict() applies that layout.
    """
    X = np.hstack([brain, np.ones((brain.shape[0], 1))])
    return X.T @ np.linalg.solve(
        X @ X.T + alpha * np.eye(X.shape[0]),
        text,
    )


@dataclass(frozen=True)
class DecodingData:
    """One subject, one feature layer, still in the original feature scale."""

    brain_train: np.ndarray
    text_train: np.ndarray
    runs_train: np.ndarray
    brain_test: np.ndarray
    text_test: np.ndarray
    video_ids_test: np.ndarray


def model_dirname(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def available_layers(directory: Path) -> list[int]:
    layers = []
    for path in directory.glob("layer*.mat"):
        suffix = path.stem.removeprefix("layer")
        if suffix.isdigit():
            layers.append(int(suffix))
    return sorted(layers)


def features_for_videos(features: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    """Pick caption-feature rows. video_ids are 1-based."""
    if video_ids.size == 0:
        raise ValueError("No samples to align with caption features")
    if int(video_ids.min()) < 1 or int(video_ids.max()) > features.shape[0]:
        raise ValueError(
            f"Video id outside 1..{features.shape[0]}: {int(video_ids.min())}..{int(video_ids.max())}"
        )
    return features[video_ids.astype(np.int64) - 1]


def column_statistics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and unbiased standard deviation. Constant columns get scale 1."""
    mean, scale = mean_and_sd(np.asarray(values, dtype=np.float64))
    return mean, np.where(scale > 0, scale, 1.0)


def apply_standardizer(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64) - mean) / scale


def restore_standardizer(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    return values * scale + mean


def predict(brain: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply weights from fit_ridge. The last row of weights is the intercept."""
    expected_rows = brain.shape[1] + 1
    if weights.ndim != 2 or weights.shape[0] != expected_rows:
        raise ValueError(
            f"weights must have shape ({expected_rows}, n_units), got {weights.shape}"
        )
    return brain @ weights[:-1] + weights[-1]


def _centered(values: np.ndarray, axis: int) -> np.ndarray:
    return values - values.mean(axis=axis, keepdims=True)


def mean_correlation(true: np.ndarray, predicted: np.ndarray, axis: int) -> float:
    """Mean Pearson correlation along one axis.

    axis 0 correlates each feature unit across samples (profile correlation).
    axis 1 correlates each sample across feature units (pattern correlation).
    """
    if true.shape != predicted.shape:
        raise ValueError(
            f"Shape mismatch: true {true.shape}, predicted {predicted.shape}"
        )
    left = _centered(true, axis)
    right = _centered(predicted, axis)
    numerator = (left * right).sum(axis=axis)
    denominator = np.sqrt((left**2).sum(axis=axis) * (right**2).sum(axis=axis))
    valid = denominator > 0
    if not np.any(valid):
        return float("nan")
    return float((numerator[valid] / denominator[valid]).mean())


def average_by_video(values: np.ndarray, video_ids: np.ndarray) -> np.ndarray:
    """Average repeated presentations of the same video."""
    unique_ids = np.unique(video_ids)
    groups = [values[video_ids == video_id].mean(axis=0) for video_id in unique_ids]
    return np.vstack(groups)


def fold_validation_masks(runs: np.ndarray) -> list[np.ndarray]:
    masks = [(runs >= start) & (runs <= stop) for start, stop in RUN_FOLDS]
    covered = np.zeros(runs.shape[0], dtype=bool)
    for mask in masks:
        if not np.any(mask):
            raise ValueError(
                f"A run fold has no samples. Runs present: {np.unique(runs)}"
            )
        covered |= mask
    missing = np.unique(runs[~covered])
    if missing.size:
        joined = ", ".join(str(int(run)) for run in missing)
        raise ValueError(f"Runs outside the six decoding folds: {joined}")
    return masks


def select_alpha(data: DecodingData, alphas: np.ndarray) -> float:
    """Pick the penalty with the highest validation profile correlation."""
    masks = fold_validation_masks(data.runs_train)
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        scores = []
        for mask in masks:
            brain_mean, brain_scale = column_statistics(data.brain_train[~mask])
            text_mean, text_scale = column_statistics(data.text_train[~mask])
            weights = fit_ridge(
                apply_standardizer(data.brain_train[~mask], brain_mean, brain_scale),
                apply_standardizer(data.text_train[~mask], text_mean, text_scale),
                float(alpha),
            )
            predicted = predict(
                apply_standardizer(data.brain_train[mask], brain_mean, brain_scale),
                weights,
            )
            held_out = apply_standardizer(data.text_train[mask], text_mean, text_scale)
            scores.append(mean_correlation(held_out, predicted, axis=0))
        score = float(np.mean(scores))
        print(f"alpha={float(alpha):.6g}  validation profile r={score:.4f}")
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
    print(f"Selected alpha={best_alpha:.6g} (validation profile r={best_score:.4f})")
    return best_alpha


def score_predictions(
    true: np.ndarray, predicted: np.ndarray, video_ids: np.ndarray
) -> None:
    true_mean = average_by_video(true, video_ids)
    predicted_mean = average_by_video(predicted, video_ids)
    print(
        "Profile r: "
        f"{mean_correlation(true, predicted, axis=0):.4f} "
        f"(repetition-averaged {mean_correlation(true_mean, predicted_mean, axis=0):.4f})"
    )
    print(
        "Pattern r: "
        f"{mean_correlation(true, predicted, axis=1):.4f} "
        f"(repetition-averaged {mean_correlation(true_mean, predicted_mean, axis=1):.4f})"
    )


def load_decoding_data(
    fmri_dir: Path, feature_path: Path, subject: str, test_split: str
) -> DecodingData:
    train_path = fmri_dir / f"trainPerception_{subject}.mat"
    test_path = fmri_dir / f"{test_split}_{subject}.mat"
    brain_train, video_ids_train, runs_train = load_brain_samples(train_path)
    brain_test, video_ids_test, _ = load_brain_samples(test_path)
    if brain_train.shape[1] != brain_test.shape[1]:
        raise ValueError(
            f"Train and test voxel counts differ: {brain_train.shape[1]} and {brain_test.shape[1]}"
        )
    features = load_video_features(feature_path)
    finite = (
        np.isfinite(brain_train).all()
        and np.isfinite(brain_test).all()
        and np.isfinite(features).all()
    )
    if not finite:
        raise ValueError("Brain activity or caption features contain non-finite values")
    return DecodingData(
        brain_train=brain_train,
        text_train=features_for_videos(features, video_ids_train),
        runs_train=runs_train,
        brain_test=brain_test,
        text_test=features_for_videos(features, video_ids_test),
        video_ids_test=video_ids_test,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare one subject's brain-to-text ridge regression and score the learned map."
    )
    parser.add_argument("--subject", default="S1", help="Subject id, for example S1")
    parser.add_argument(
        "--model",
        default="deberta-large",
        help="Feature directory name or Hugging Face model id",
    )
    parser.add_argument(
        "--layer",
        type=int,
        help="1-based transformer layer. Defaults to the last saved layer",
    )
    parser.add_argument(
        "--test-split",
        default="testPerception",
        choices=("testPerception", "testImagery"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e4,
        help="Ridge penalty used when --select-alpha is off",
    )
    parser.add_argument(
        "--select-alpha",
        action="store_true",
        help="Choose the penalty by six-fold cross-validation on the training runs",
    )
    parser.add_argument(
        "--fmri-dir", type=Path, default=ROOT / "data/fmri/data/preprocessed"
    )
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/feature")
    parser.add_argument(
        "--output", type=Path, help="Where to write weights and test predictions"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an output file that already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.alpha < 0:
        raise ValueError("--alpha must be non-negative")
    dirname = model_dirname(args.model)
    feature_dir = args.feature_root / "video" / dirname
    layers = available_layers(feature_dir)
    if not layers:
        raise FileNotFoundError(f"No layer files in {feature_dir}")
    layer = layers[-1] if args.layer is None else args.layer
    if layer not in layers:
        joined = ", ".join(f"{index:02d}" for index in layers)
        raise FileNotFoundError(
            f"Layer {layer} is not in {feature_dir}. Available layers: {joined}"
        )

    output = args.output
    if output is None:
        output = (
            ROOT / "res" / "decoding" / args.subject / dirname / f"layer{layer:02d}.npz"
        )
    if output.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output} already exists. Pass --overwrite to replace it"
        )

    feature_path = feature_dir / f"layer{layer:02d}.mat"
    data = load_decoding_data(
        args.fmri_dir, feature_path, args.subject, args.test_split
    )
    n_repetitions = data.video_ids_test.size / np.unique(data.video_ids_test).size
    print(
        f"{args.subject} train: {data.brain_train.shape[0]} samples, "
        f"{data.brain_train.shape[1]} voxels, {len(np.unique(data.runs_train))} runs"
    )
    print(
        f"{args.subject} {args.test_split}: {data.brain_test.shape[0]} samples, "
        f"{len(np.unique(data.video_ids_test))} videos, {n_repetitions:.0f} repetitions"
    )
    print(
        f"Text features: {dirname} layer {layer:02d}, {data.text_train.shape[1]} units from {feature_path}"
    )
    for start, stop in RUN_FOLDS:
        count = int(((data.runs_train >= start) & (data.runs_train <= stop)).sum())
        print(f"  runs {start}-{stop}: {count} validation samples")

    alpha = select_alpha(data, ALPHAS) if args.select_alpha else args.alpha
    print(f"Fitting ridge with alpha={alpha:g}")
    brain_mean, brain_scale = column_statistics(data.brain_train)
    text_mean, text_scale = column_statistics(data.text_train)
    weights = fit_ridge(
        apply_standardizer(data.brain_train, brain_mean, brain_scale),
        apply_standardizer(data.text_train, text_mean, text_scale),
        alpha,
    )
    predicted_z = predict(
        apply_standardizer(data.brain_test, brain_mean, brain_scale), weights
    )
    predicted = restore_standardizer(predicted_z, text_mean, text_scale)
    score_predictions(data.text_test, predicted, data.video_ids_test)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        weights=np.asarray(weights, dtype=np.float32),
        alpha=np.float64(alpha),
        brain_mean=brain_mean.astype(np.float32),
        brain_scale=brain_scale.astype(np.float32),
        text_mean=text_mean.astype(np.float32),
        text_scale=text_scale.astype(np.float32),
        predicted=np.asarray(predicted, dtype=np.float32),
        video_id=data.video_ids_test.astype(np.int32),
        subject=np.array(args.subject),
        model=np.array(dirname),
        layer=np.int32(layer),
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
