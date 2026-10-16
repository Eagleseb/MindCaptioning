"""Load the per-video caption table."""

from __future__ import annotations

import csv
from pathlib import Path


def load_captions(path: Path) -> tuple[list[list[str]], int]:
    grouped: dict[int, list[str]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"video_id", "description"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must have video_id and description columns")
        for row in reader:
            video_id = int(row["video_id"])
            grouped.setdefault(video_id, []).append(row["description"])
    if not grouped:
        raise ValueError(f"No captions in {path}")
    n_videos = max(grouped)
    captions = [grouped.get(video_id, []) for video_id in range(1, n_videos + 1)]
    return captions, n_videos
