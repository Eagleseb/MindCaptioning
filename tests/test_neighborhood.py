"""Cell-preserving caption-feature augmentation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mind_captioning.captioner.data import VideoExample  # noqa: E402
from mind_captioning.captioner.neighborhood import (  # noqa: E402
    CellAugmentedDataset,
    fit_cell_geometry,
    nearest_source_indices,
    perturb,
    probe_examples,
)


def examples() -> list[VideoExample]:
    points = ([0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [3.0, 3.0])
    return [
        VideoExample(index + 1, np.array(point, dtype=np.float32), (f"video {index}",))
        for index, point in enumerate(points)
    ]


class NeighborhoodTest(unittest.TestCase):
    def test_probes_remain_in_their_source_cells(self) -> None:
        items = examples()
        geometry = fit_cell_geometry(items, chunk_size=2)
        expected = np.arange(len(items))
        for axis in ("angular", "radial"):
            points = np.stack(
                [perturb(geometry, index, axis, 0.9) for index in range(len(items))]
            )
            np.testing.assert_array_equal(
                nearest_source_indices(points, geometry.source), expected
            )

    def test_dataset_is_reproducible_and_validation_has_both_axes(self) -> None:
        items = examples()
        geometry = fit_cell_geometry(items)
        left = CellAugmentedDataset(
            items, geometry, seed=7, clean_probability=0.0, max_fraction=0.9
        )
        right = CellAugmentedDataset(
            items, geometry, seed=7, clean_probability=0.0, max_fraction=0.9
        )
        left.set_epoch(3)
        right.set_epoch(3)
        for index in range(len(items)):
            np.testing.assert_allclose(
                left[index]["embedding"], right[index]["embedding"]
            )
        probes = probe_examples(items, geometry, 0.5)
        self.assertEqual(len(probes), 2 * len(items))
        self.assertEqual([probe.video_id for probe in probes[:4]], [1, 1, 2, 2])


if __name__ == "__main__":
    unittest.main()
