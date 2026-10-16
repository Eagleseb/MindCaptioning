"""Train and evaluate the cell-preserving captioner experiment."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mind_captioning.captioner.brain_eval import geometric_match
from mind_captioning.captioner.checkpoint import load_checkpoint
from mind_captioning.captioner.data import examples_for_ids, feature_matrix_path, load_examples
from mind_captioning.captioner.ledger import run_record, upsert_run
from mind_captioning.captioner.neighborhood_eval import (
    decoded_cell_diagnostic,
    evaluate_shells,
)
from mind_captioning.device import pick_device
from run_experiments import ROOT, evaluate_run, relative, train_run
from train_decoder import average_by_video

RUN_ID = "margin-cells"
HYPOTHESIS = (
    "Clean plus cell-preserving angular/radial probes improve nearby OOD caption "
    "stability without changing the frozen decoder or prefix capacity."
)


def main() -> None:
    train_run(RUN_ID, HYPOTHESIS, ["--cell-augmentation"])
    checkpoint = ROOT / "res" / "captioner" / "runs" / RUN_ID / "best"
    brain_summary = evaluate_run(RUN_ID, checkpoint, HYPOTHESIS)

    device = pick_device("mps")
    model, tokenizer, config, split = load_checkpoint(checkpoint, device)
    feature_path, _ = feature_matrix_path(
        ROOT / "data" / "feature", config["feature_model"], int(config["layer"])
    )
    library = load_examples(
        ROOT / "data" / "caption" / "caption_ck20.csv",
        feature_path,
        int(config["captions_per_video"]),
    )
    validation = examples_for_ids(library, split["validation"])
    output = ROOT / "res" / "captioner" / "runs" / RUN_ID / "evaluations"
    shells = evaluate_shells(model, tokenizer, validation, output / "shells.json")

    with np.load(
        ROOT / "res" / "decoding" / "S1" / "deberta-large" / "layer24.npz"
    ) as ridge:
        test_ids = np.unique(ridge["video_id"].astype(np.int64))
        decoded = average_by_video(
            ridge["predicted"].astype(np.float32),
            ridge["video_id"].astype(np.int64),
        ).astype(np.float32)
    test = examples_for_ids(library, test_ids)
    true_features = np.stack([example.embedding for example in test])
    train_features = np.stack(
        [example.embedding for example in examples_for_ids(library, split["train"])]
    )
    matched = geometric_match(decoded, train_features)
    diagnostic = decoded_cell_diagnostic(decoded, true_features, matched)
    (output / "cell_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2) + "\n"
    )
    print(json.dumps(diagnostic, indent=2))

    record = run_record(ROOT, RUN_ID) or {"id": RUN_ID}
    record.update(
        {
            "shells": relative(output / "shells.json"),
            "cell_diagnostic": relative(output / "cell_diagnostic.json"),
            "validation_shells": shells["rows"],
            "true_test_cell_fraction": diagnostic["brain_norm_matched"][
                "true_cell_fraction"
            ],
            "brain_decoding": brain_summary["decoding"],
        }
    )
    upsert_run(ROOT, record)


if __name__ == "__main__":
    main()
