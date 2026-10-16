"""Generate a caption from a cached video embedding or a saved vector.

The vector must have the width stored in the checkpoint. A video id reads the
same feature row the captioner was trained on. A saved vector can instead be
a restored brain-decoded feature of that width; this script does not check
where the vector came from.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mind_captioning.captioner import LIMITATION
from mind_captioning.captioner.checkpoint import load_checkpoint
from mind_captioning.captioner.data import feature_matrix_path
from mind_captioning.captioner.generate import generate_caption, load_embedding_file
from mind_captioning.device import pick_device
from mind_captioning.io import load_video_features

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--video-id", type=int, action="append", dest="video_ids")
    parser.add_argument("--embedding-path", type=Path, help="A .npy, .npz, or .pt vector")
    parser.add_argument("--feature-root", type=Path, default=ROOT / "data/feature")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--sample", action="store_true", help="Sample instead of greedy decoding")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--output", type=Path, help="Optional JSONL file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.video_ids and args.embedding_path is None:
        raise ValueError("Pass --video-id or --embedding-path")
    if args.video_ids and args.embedding_path is not None:
        raise ValueError("Pass either --video-id or --embedding-path")
    device = pick_device(args.device)
    model, tokenizer, config, _ = load_checkpoint(args.checkpoint, device)
    if args.embedding_path is not None:
        embedding = load_embedding_file(args.embedding_path)
        video_ids = None
    else:
        feature_path, _ = feature_matrix_path(
            args.feature_root, config["feature_model"], int(config["layer"])
        )
        features = load_video_features(feature_path)
        rows = []
        for video_id in args.video_ids:
            if video_id < 1 or video_id > features.shape[0]:
                raise ValueError(f"Video id {video_id} is outside 1..{features.shape[0]}")
            rows.append(features[video_id - 1])
        embedding = np.stack(rows).astype(np.float32)
        video_ids = args.video_ids
    texts = generate_caption(
        model,
        tokenizer,
        embedding,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    if isinstance(texts, str):
        texts = [texts]
    rows = []
    for index, text in enumerate(texts):
        row = {"caption": text}
        if video_ids is not None:
            row["video_id"] = video_ids[index]
        rows.append(row)
        label = f"video {row['video_id']}" if "video_id" in row else f"row {index}"
        print(f"{label}: {text}")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        print(f"Wrote {args.output}")
    print(LIMITATION)


if __name__ == "__main__":
    main()
