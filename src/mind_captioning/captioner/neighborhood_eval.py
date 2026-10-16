"""Evaluate caption stability on controlled shells inside embedding cells."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mind_captioning.captioner.generate import generate_captions
from mind_captioning.captioner.metrics import rouge_l_max
from mind_captioning.captioner.neighborhood import (
    fit_cell_geometry,
    nearest_source_indices,
    perturb,
)


def evaluate_shells(
    model,
    tokenizer,
    examples,
    output: Path,
    fractions: tuple[float, ...] = (0.0, 0.5, 0.9),
    batch_size: int = 16,
) -> dict:
    """Score angular and radial probes at fixed fractions of the cell boundary."""
    geometry = fit_cell_geometry(examples)
    rows = []
    for fraction in fractions:
        axes = ("clean",) if fraction == 0 else ("angular", "radial")
        for axis in axes:
            points = (
                geometry.source.copy()
                if axis == "clean"
                else np.stack(
                    [perturb(geometry, index, axis, fraction) for index in range(len(examples))]
                )
            )
            captions: list[str] = []
            for start in range(0, len(points), batch_size):
                captions.extend(
                    generate_captions(
                        model,
                        tokenizer,
                        points[start : start + batch_size],
                        max_new_tokens=32,
                        repetition_penalty=1.2,
                        no_repeat_ngram_size=3,
                    )
                )
            rouge = np.array(
                [
                    rouge_l_max(caption, example.captions)
                    for caption, example in zip(captions, examples)
                ]
            )
            nearest = nearest_source_indices(points, geometry.source)
            retention = float((nearest == np.arange(len(examples))).mean())
            rows.append(
                {
                    "fraction": fraction,
                    "axis": axis,
                    "rouge_l": float(rouge.mean()),
                    "cell_retention": retention,
                    "distinct": len(set(captions)) / len(captions),
                    "n": len(captions),
                }
            )
            print(
                f"shell {fraction:.1f} {axis}: ROUGE-L {rouge.mean():.3f}, "
                f"cell retention {retention:.1%}"
            )
    result = {"rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def decoded_cell_diagnostic(
    decoded: np.ndarray,
    true_features: np.ndarray,
    norm_matched: np.ndarray,
) -> dict:
    """Measure whether decoded test vectors remain in their true caption cell."""
    target = np.arange(len(true_features))
    result = {}
    for name, points in (("brain_averaged", decoded), ("brain_norm_matched", norm_matched)):
        nearest = nearest_source_indices(points, true_features)
        distance_true = np.linalg.norm(points - true_features, axis=1)
        result[name] = {
            "true_cell_fraction": float((nearest == target).mean()),
            "median_true_distance": float(np.median(distance_true)),
            "mean_true_distance": float(distance_true.mean()),
        }
    return result
