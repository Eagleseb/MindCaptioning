"""Run the captioner queue and record each result in the run ledger.

The finished raw-feature checkpoint is left where it is. Decoding-only and
later training runs write under res/captioner/runs/<id>/. The combined run
starts only when both training runs beat that baseline on decoded brain
features.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from mind_captioning.captioner.brain_eval import beats_baseline, comparison_condition, evaluate_brain
from mind_captioning.captioner.checkpoint import load_checkpoint
from mind_captioning.captioner.data import feature_matrix_path, load_examples
from mind_captioning.captioner.ledger import upsert_run
from mind_captioning.device import pick_device

ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "res" / "captioner" / "distilgpt2" / "deberta-large" / "layer24"
BASELINE_CHECKPOINT = BASELINE_DIR / "best"
PYTHON = sys.executable


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def condition(loss: float, rouge: float, other: float, accuracy: float, rank: float, distinct: float, n: int = 72) -> dict:
    return {
        "loss": loss,
        "perplexity": float(np.exp(loss)),
        "rouge_own": rouge,
        "rouge_other": other,
        "accuracy": accuracy,
        "median_rank": rank,
        "distinct": distinct,
        "n": n,
    }


def register_baseline() -> dict:
    """Record the greedy evaluation already stored in the brain-caption notebook."""
    summary = {
        "normalization": "none",
        "decoding": {"max_new_tokens": 40, "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "do_sample": False},
        "source": "notebooks/brain_captioning.ipynb",
        "conditions": {
            "caption_features": condition(2.433, 0.546, 0.292, 44 / 72, 1.0, 1.0),
            "brain_averaged": condition(3.187, 0.292, 0.246, 7 / 72, 21.2, 0.99),
            "brain_norm_matched": condition(3.100, 0.370, 0.295, 8 / 72, 12.0, 1.0),
            "shuffled": condition(3.845, 0.245, 0.247, 1 / 72, 33.8, 0.99),
            "mean_training": condition(3.688, 0.150, 0.150, 1 / 72, 39.5, 1 / 72),
        },
    }
    evaluation = BASELINE_DIR / "evaluations"
    evaluation.mkdir(parents=True, exist_ok=True)
    (evaluation / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    record = {
        "id": "baseline",
        "status": "done",
        "hypothesis": "Frozen distilgpt2, 8 prefix tokens, raw caption features, greedy decoding.",
        "checkpoint": relative(BASELINE_CHECKPOINT),
        "best_epoch": 9,
        "best_validation_loss": 2.6394,
        **ledger_metrics(summary),
    }
    upsert_run(ROOT, record)
    return summary


def ledger_metrics(summary: dict) -> dict:
    chosen = comparison_condition(summary)
    return {
        "comparison_condition": chosen["condition"],
        "brain_loss": chosen["loss"],
        "brain_rouge_l": chosen["rouge_own"],
        "brain_identification": chosen["accuracy"],
        "brain_distinct": chosen["distinct"],
        "shuffled_rouge_l": summary["conditions"]["shuffled"]["rouge_own"],
        "caption_rouge_l": summary["conditions"]["caption_features"]["rouge_own"],
        "caption_identification": summary["conditions"]["caption_features"]["accuracy"],
    }


def train_run(run_id: str, hypothesis: str, args: list[str]) -> None:
    output = ROOT / "res" / "captioner" / "runs" / run_id
    upsert_run(ROOT, {"id": run_id, "status": "running", "hypothesis": hypothesis, "output": relative(output)})
    command = [
        PYTHON,
        "-m",
        "train_captioner",
        "--run-id",
        run_id,
        "--output",
        str(output),
        "--device",
        "mps",
        "--overwrite",
        *args,
    ]
    print(" ".join(command), flush=True)
    try:
        subprocess.run(command, cwd=ROOT, check=True)
    except Exception as error:
        upsert_run(ROOT, {"id": run_id, "status": "failed", "hypothesis": hypothesis, "error": str(error)})
        raise


def evaluate_run(run_id: str, checkpoint: Path, hypothesis: str, **decoding) -> dict:
    device = pick_device("mps")
    model, tokenizer, config, split = load_checkpoint(checkpoint, device)
    feature_path, _ = feature_matrix_path(ROOT / "data" / "feature", config["feature_model"], int(config["layer"]))
    library = load_examples(ROOT / "data" / "caption" / "caption_ck20.csv", feature_path, int(config["captions_per_video"]))
    ridge = np.load(ROOT / "res" / "decoding" / "S1" / "deberta-large" / "layer24.npz")
    output = ROOT / "res" / "captioner" / "runs" / run_id / "evaluations"
    try:
        summary = evaluate_brain(
            model,
            tokenizer,
            config,
            split,
            library,
            ridge["predicted"],
            ridge["video_id"],
            device,
            output,
            **decoding,
        )
    except Exception as error:
        upsert_run(ROOT, {"id": run_id, "status": "failed", "hypothesis": hypothesis, "error": str(error)})
        raise
    history_path = checkpoint.parent / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    record = {
        "id": run_id,
        "status": "done",
        "hypothesis": hypothesis,
        "checkpoint": relative(checkpoint),
        "summary": relative(output / "summary.json"),
        "best_epoch": None if history is None else config.get("best_epoch"),
        "best_validation_loss": None if history is None else config.get("best_validation_loss"),
        **ledger_metrics(summary),
    }
    upsert_run(ROOT, record)
    chosen = comparison_condition(summary)
    print(
        f"{run_id}: {chosen['condition']} ROUGE-L {chosen['rouge_own']:.3f}, "
        f"identification {chosen['accuracy']:.1%}, distinct {chosen['distinct']:.0%}",
        flush=True,
    )
    return summary


def main() -> None:
    baseline = register_baseline()
    decode_hypothesis = "Same baseline checkpoint with repetition penalty 1.2, no repeated trigram, and 32 new tokens."
    upsert_run(ROOT, {"id": "decode-penalty", "status": "running", "hypothesis": decode_hypothesis})
    decode = evaluate_run(
        "decode-penalty",
        BASELINE_CHECKPOINT,
        decode_hypothesis,
        max_new_tokens=32,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
    )
    decode["normalization"] = "none"

    norm_hypothesis = "Z-score training features and shrink or perturb them so decoded vectors are in range."
    train_run(
        "norm-noise",
        norm_hypothesis,
        ["--normalize", "zscore", "--shrink-min", "0.25", "--shrink-max", "1", "--noise-std", "0.1"],
    )
    norm = evaluate_run("norm-noise", ROOT / "res" / "captioner" / "runs" / "norm-noise" / "best", norm_hypothesis)

    lora_hypothesis = "Z-scored features, 16 prefix tokens, and LoRA on the last two DistilGPT-2 blocks."
    train_run(
        "prefix16-lora",
        lora_hypothesis,
        ["--normalize", "zscore", "--prefix-tokens", "16", "--lora-rank", "8", "--lora-layers", "2"],
    )
    lora = evaluate_run("prefix16-lora", ROOT / "res" / "captioner" / "runs" / "prefix16-lora" / "best", lora_hypothesis)

    combined_id = "norm-noise-prefix16-lora"
    combined_hypothesis = "Z-scoring, shrinkage noise, 16 prefix tokens, and LoRA on the last two blocks."
    if beats_baseline(norm, baseline) and beats_baseline(lora, baseline):
        train_run(
            combined_id,
            combined_hypothesis,
            [
                "--normalize",
                "zscore",
                "--shrink-min",
                "0.25",
                "--shrink-max",
                "1",
                "--noise-std",
                "0.1",
                "--prefix-tokens",
                "16",
                "--lora-rank",
                "8",
                "--lora-layers",
                "2",
            ],
        )
        evaluate_run(combined_id, ROOT / "res" / "captioner" / "runs" / combined_id / "best", combined_hypothesis)
    else:
        upsert_run(
            ROOT,
            {
                "id": combined_id,
                "status": "skipped",
                "hypothesis": combined_hypothesis,
                "reason": "norm-noise and prefix16-lora did not both beat the baseline brain scores.",
                "norm_noise_beats_baseline": beats_baseline(norm, baseline),
                "prefix16_lora_beats_baseline": beats_baseline(lora, baseline),
            },
        )
        print("Skipped norm-noise-prefix16-lora", flush=True)
    print("decode-penalty comparison condition", comparison_condition(decode)["condition"], flush=True)


if __name__ == "__main__":
    main()
