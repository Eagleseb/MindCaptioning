"""Mean and standard deviation of video features for each split."""

from __future__ import annotations

import numpy as np

# Run ranges from Mind Captioning mcap_setParams.m (run2FoldAssignIdx).
RUN_FOLDS = ((1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 58))


def mean_and_sd(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column mean and unbiased standard deviation of a (samples, units) matrix."""
    if features.shape[0] < 2:
        raise ValueError(
            f"Need at least two videos to compute a standard deviation, got {features.shape[0]}"
        )
    return features.mean(axis=0), features.std(axis=0, ddof=1)


def normalization_parameters(
    features: np.ndarray,
    train_ids: np.ndarray,
    train_runs: np.ndarray,
    test_ids: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build the six normalization arrays for one layer.

    features is (n_videos, n_units) in video_id order. mu and sd use every
    training video. Each row of mu_tr and sd_tr leaves out one run fold.
    mu_val and sd_val use the test-perception videos.
    """

    def rows(video_ids: np.ndarray) -> np.ndarray:
        unique_ids = np.unique(video_ids.astype(np.int64))
        if unique_ids.min() < 1 or unique_ids.max() > features.shape[0]:
            raise ValueError(
                f"Stimulus id outside 1..{features.shape[0]}: {unique_ids.min()}..{unique_ids.max()}"
            )
        return unique_ids - 1

    train_rows = rows(train_ids)
    mu, sd = mean_and_sd(features[train_rows])
    mu_val, sd_val = mean_and_sd(features[rows(test_ids)])
    fold_mu = []
    fold_sd = []
    for start, stop in RUN_FOLDS:
        in_fold = (train_runs >= start) & (train_runs <= stop)
        fold_mu_i, fold_sd_i = mean_and_sd(features[rows(train_ids[~in_fold])])
        fold_mu.append(fold_mu_i)
        fold_sd.append(fold_sd_i)
    return {
        "mu": mu.reshape(1, -1),
        "sd": sd.reshape(1, -1),
        "mu_val": mu_val.reshape(1, -1),
        "sd_val": sd_val.reshape(1, -1),
        "mu_tr": np.vstack(fold_mu),
        "sd_tr": np.vstack(fold_sd),
    }
