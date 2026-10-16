"""Score a captioner on a held-out video split.

The checkpoint's saved split decides which videos are used. Test videos are
not required for training, and this script is the first place they are read
for scoring. Reported metrics compare generated text with the captions that
produced the embedding.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mind_captioning.captioner import LIMITATION
from mind_captioning.captioner.checkpoint import load_checkpoint
from mind_captioning.captioner.data import feature_matrix_path, load_examples
from mind_captioning.captioner.reporting import evaluate_examples, examples_from_split
from mind_captioning.device import pick_device

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--captions", type=Path, default=ROOT / "data/caption/caption_ck20.csv")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/feature")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, help="JSONL path. Defaults beside the checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1 or args.num_beams < 1:
        raise ValueError("--batch-size, --max-new-tokens, and --num-beams must be positive")
    device = pick_device(args.device)
    model, tokenizer, config, split = load_checkpoint(args.checkpoint, device)
    if args.split not in split:
        raise KeyError(f"{args.checkpoint} has no {args.split} split")
    feature_model = config["feature_model"]
    feature_path, layer = feature_matrix_path(args.feature_root, feature_model, int(config["layer"]))
    if layer != int(config["layer"]):
        raise ValueError(f"Checkpoint layer {config['layer']} resolved to layer {layer}")
    library = load_examples(args.captions, feature_path, int(config["captions_per_video"]))
    if library[0].embedding.shape[0] != int(config["embedding_dim"]):
        raise ValueError(
            f"Feature width {library[0].embedding.shape[0]} does not match the checkpoint "
            f"({config['embedding_dim']})"
        )
    print(LIMITATION)
    print(f"Scoring {args.split} ({len(split[args.split])} videos) from {feature_path}")
    summary = evaluate_examples(
        model,
        tokenizer,
        examples_from_split(library, split, args.split),
        examples_from_split(library, split, "train"),
        args.output or args.checkpoint.parent / "evaluations" / f"{args.split}.jsonl",
        device,
        max_length=int(config["max_length"]),
        batch_size=args.batch_size,
        seed=int(config.get("seed", 42)),
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    print(
        f"loss {summary['loss']:.4f}  perplexity {summary['perplexity']:.2f}  "
        f"ROUGE-L {summary['rouge_l']:.4f}  baseline ROUGE-L {summary['baseline_rouge_l']:.4f}"
    )
    print(f"Wrote {summary['jsonl']}")
    print(json.dumps({key: summary[key] for key in ("loss", "perplexity", "rouge_l", "baseline_rouge_l", "bert_score_f1")}))


if __name__ == "__main__":
    main()
