"""Read MATLAB v7.3 brain and feature files, and write feature files."""

from __future__ import annotations

import platform
import time
from pathlib import Path

import h5py
import numpy as np


def _matlab_chars(array: np.ndarray) -> str:
    return "".join(chr(int(code)) for code in np.asarray(array).reshape(-1))


def _label_matrix(labels: np.ndarray, n_names: int) -> np.ndarray:
    """Return labels as (n_types, n_samples)."""
    if labels.ndim != 2:
        raise ValueError(f"Expected a label matrix, got shape {labels.shape}")
    if labels.shape[0] <= n_names and labels.shape[1] > labels.shape[0]:
        return labels
    if labels.shape[1] <= n_names and labels.shape[0] > labels.shape[1]:
        return labels.T
    raise ValueError(f"Could not orient label matrix of shape {labels.shape}")


def _label_names(handle: h5py.File) -> list[str]:
    references = handle["metainf/label_type"][()]
    return [_matlab_chars(np.array(handle[references[index, 0]])) for index in range(references.shape[0])]


def _label_column(handle: h5py.File, name: str, path: Path) -> np.ndarray:
    names = _label_names(handle)
    labels = _label_matrix(np.array(handle["metainf/Label"]), len(names))
    if name not in names or names.index(name) >= labels.shape[0]:
        available = ", ".join(names[: labels.shape[0]])
        raise ValueError(f"{path} has no {name} label. Available labels: {available}")
    return np.rint(labels[names.index(name)]).astype(np.int64)


def _run_column(handle: h5py.File) -> np.ndarray:
    return np.rint(np.array(handle["metainf/Run"], dtype=np.float64).reshape(-1)).astype(np.int64)


def load_stimuli(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return 1-based stimID and run for each sample."""
    with h5py.File(path, "r") as handle:
        stim_ids = _label_column(handle, "stimID", path)
        runs = _run_column(handle)
    if stim_ids.shape != runs.shape:
        raise ValueError(f"{path} stimID and Run lengths differ")
    return stim_ids, runs


def load_matlab_matrix(path: Path, name: str) -> np.ndarray:
    """Load one numeric array in its MATLAB shape.

    HDF5 stores each MATLAB array transposed.
    """
    with h5py.File(path, "r") as handle:
        if name not in handle:
            raise KeyError(f"{path} has no dataset {name}")
        stored = np.array(handle[name])
    if stored.ndim < 2:
        return stored
    return np.ascontiguousarray(stored.T)


def load_brain_samples(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return brain activity, video ids, and run numbers.

    Activity has shape (n_samples, n_voxels). Video ids are the 1-based
    Condition labels. Those labels index caption-feature rows, and on the
    perception runs they match stimID.
    """
    with h5py.File(path, "r") as handle:
        if "braindat" not in handle:
            raise KeyError(f"{path} has no dataset braindat")
        activity = np.ascontiguousarray(np.array(handle["braindat"]).T)
        video_ids = _label_column(handle, "Condition", path)
        runs = _run_column(handle)
    if activity.shape[0] != video_ids.shape[0] or activity.shape[0] != runs.shape[0]:
        raise ValueError(
            f"{path} sample axis does not match labels: "
            f"brain {activity.shape[0]}, videos {video_ids.shape[0]}, runs {runs.shape[0]}"
        )
    return activity, video_ids, runs


def load_video_features(path: Path) -> np.ndarray:
    """Return caption features with shape (n_videos, n_units).

    Row i is video_id i + 1.
    """
    features = load_matlab_matrix(path, "feat")
    if features.ndim != 2:
        raise ValueError(f"{path} feat has shape {features.shape}; expected a video-by-unit matrix")
    return features


def _matlab_header() -> bytes:
    stamp = time.strftime("%a %b %d %H:%M:%S %Y")
    text = f"MATLAB 7.3 MAT-file, Platform: {platform.system()}, Created on: {stamp} HDF5 schema 1.00 ."
    padded = text.encode("ascii")[:116].ljust(116)
    return padded + b"\x00" * 8 + b"\x00\x02IM"


def write_matlab(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write float32 arrays as a MATLAB v7.3 file.

    Array shapes are MATLAB shapes. HDF5 stores each array transposed, which
    is how MATLAB load and h5py readers of these files expect them.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w", userblock_size=512) as handle:
        string_type = h5py.string_dtype(encoding="ascii", length=6)
        for name, value in arrays.items():
            matlab = np.ascontiguousarray(value, dtype=np.float32)
            dataset = handle.create_dataset(name, data=np.ascontiguousarray(matlab.T))
            dataset.attrs.create("MATLAB_class", "single", dtype=string_type)
    with path.open("r+b") as handle:
        handle.write(_matlab_header())
