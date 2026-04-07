"""
Append-only JSONL writer for serving observability artifacts.

- Creates parent directories if missing.
- Writes one JSON object per line.
- Preserves append-only semantics (opens in append mode).
- No threading, no background scheduler, no runtime daemon.
"""

import json
from pathlib import Path
from typing import Any


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append a single JSON record as one line to the given JSONL file.

    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")


def append_jsonl_batch(path: Path, records: list[dict[str, Any]]) -> None:
    """Append multiple JSON records to the given JSONL file.

    Each record is written as a separate line. Creates parent directories
    if they do not exist.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str, sort_keys=True) + "\n")
