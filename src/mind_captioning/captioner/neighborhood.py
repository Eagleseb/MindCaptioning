"""Cell-preserving augmentation around caption-derived video embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mind_captioning.captioner.data import VideoCaptionDataset, VideoExample


@dataclass(frozen=True)
class CellGeometry:
    """Nearest-neighbour and radial boundaries for one set of source vectors."""

    source: np.ndarray
    nearest_index: np.ndarray
    nearest_distance: np.ndarray
    radial_limit: np.ndarray
    center: np.ndarray


def fit_cell_geometry(examples: list[VideoExample], chunk_size: int = 256) -> CellGeometry:
    """Fit Euclidean Voronoi-cell geometry without using another data split."""
    if len(examples) < 2:
        raise ValueError("Cell geometry needs at least two source embeddings")
    source = np.stack([example.embedding for example in examples]).astype(np.float32)
    if not np.isfinite(source).all():
        raise ValueError("Source embeddings contain non-finite values")
    n = source.shape[0]
    norms = np.einsum("ij,ij->i", source, source)
    center = source.mean(axis=0)
    nearest_index = np.empty(n, dtype=np.int64)
    nearest_distance = np.empty(n, dtype=np.float32)
    radial_limit = np.ones(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        rows = source[start:stop]
        distance2 = norms[start:stop, None] + norms[None, :] - 2.0 * rows @ source.T
        np.maximum(distance2, 0.0, out=distance2)
        distance2[np.arange(stop - start), np.arange(start, stop)] = np.inf
        nearest = distance2.argmin(axis=1)
        nearest_index[start:stop] = nearest
        nearest_distance[start:stop] = np.sqrt(
            distance2[np.arange(stop - start), nearest]
        )

        direction = center[None, :] - rows
        # Boundary along p(t)=x+t(center-x):
        # 2t(center-x).(y-x) = ||y-x||^2.
        direction_dot_source = direction @ source.T
        direction_dot_x = np.einsum("ij,ij->i", direction, rows)
        denominator = 2.0 * (direction_dot_source - direction_dot_x[:, None])
        valid = denominator > 1e-8
        bounds = np.full_like(distance2, np.inf)
        np.divide(distance2, denominator, out=bounds, where=valid)
        limit = bounds.min(axis=1)
        radial_limit[start:stop] = np.minimum(limit, 1.0)

    if np.any(nearest_distance <= 0) or not np.isfinite(radial_limit).all():
        raise ValueError("Duplicate embeddings or invalid cell boundaries")
    return CellGeometry(
        source=source,
        nearest_index=nearest_index,
        nearest_distance=nearest_distance,
        radial_limit=radial_limit,
        center=center.astype(np.float32),
    )


def perturb(
    geometry: CellGeometry,
    index: int,
    axis: str,
    fraction: float,
) -> np.ndarray:
    """Move a source vector along one axis while remaining in its cell."""
    if axis not in {"angular", "radial"}:
        raise ValueError("axis must be angular or radial")
    if not 0 <= fraction < 1:
        raise ValueError("fraction must be in [0, 1)")
    source = geometry.source[index]
    if axis == "angular":
        neighbor = geometry.source[geometry.nearest_index[index]]
        # The nearest-neighbour midpoint is the first boundary on this ray.
        return (source + (0.5 * fraction) * (neighbor - source)).astype(np.float32)
    limit = float(geometry.radial_limit[index])
    return (source + (fraction * limit) * (geometry.center - source)).astype(np.float32)


class CellAugmentedDataset(VideoCaptionDataset):
    """Return clean or cell-preserving probes, deterministically by epoch."""

    def __init__(
        self,
        examples: list[VideoExample],
        geometry: CellGeometry,
        mode: str = "all_captions",
        seed: int = 42,
        clean_probability: float = 0.25,
        max_fraction: float = 0.9,
    ) -> None:
        super().__init__(examples, mode=mode, seed=seed)
        if geometry.source.shape[0] != len(examples):
            raise ValueError("Geometry and examples have different lengths")
        if not 0 <= clean_probability <= 1:
            raise ValueError("clean_probability must be in [0, 1]")
        if not 0 < max_fraction < 1:
            raise ValueError("max_fraction must be in (0, 1)")
        self.geometry = geometry
        self.clean_probability = float(clean_probability)
        self.max_fraction = float(max_fraction)

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        rng = np.random.default_rng(
            self.seed + self.epoch * 1_000_003 + self.examples[index].video_id
        )
        draw = float(rng.random())
        if draw >= self.clean_probability:
            axis = "angular" if rng.random() < 0.5 else "radial"
            fraction = float(rng.uniform(0.0, self.max_fraction))
            item["embedding"] = perturb(self.geometry, index, axis, fraction)
        return item


def probe_examples(
    examples: list[VideoExample],
    geometry: CellGeometry,
    fraction: float,
) -> list[VideoExample]:
    """Create one angular and one radial deterministic probe per source."""
    probes: list[VideoExample] = []
    for index, example in enumerate(examples):
        for axis in ("angular", "radial"):
            probes.append(
                VideoExample(
                    video_id=example.video_id,
                    embedding=perturb(geometry, index, axis, fraction),
                    captions=example.captions,
                )
            )
    return probes


def nearest_source_indices(points: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Return Euclidean nearest-source indices in bounded-memory chunks."""
    points = np.asarray(points, dtype=np.float32)
    sources = np.asarray(sources, dtype=np.float32)
    source_norms = np.einsum("ij,ij->i", sources, sources)
    result = []
    for start in range(0, len(points), 256):
        rows = points[start : start + 256]
        distance2 = (
            np.einsum("ij,ij->i", rows, rows)[:, None]
            + source_norms[None, :]
            - 2.0 * rows @ sources.T
        )
        result.append(distance2.argmin(axis=1))
    return np.concatenate(result)
