"""Append-only index of captioner runs."""

from __future__ import annotations

import json
from pathlib import Path


def ledger_path(root: Path) -> Path:
    return root / "res" / "captioner" / "runs" / "index.jsonl"


def load_runs(root: Path) -> list[dict]:
    path = ledger_path(root)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def upsert_run(root: Path, record: dict) -> None:
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in load_runs(root) if row.get("id") != record["id"]]
    rows.append(record)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_record(root: Path, run_id: str) -> dict | None:
    for row in load_runs(root):
        if row.get("id") == run_id:
            return row
    return None
