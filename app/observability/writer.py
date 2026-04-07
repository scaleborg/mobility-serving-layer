"""
Append-only JSONL writer for serving observability artifacts.

- Creates parent directories if missing.
- Writes one JSON object per line via pydantic serialization.
- Preserves append-only semantics (opens in append mode).
- No threading, no background scheduler, no runtime daemon.
"""

from pathlib import Path

from pydantic import BaseModel


def append_jsonl(path: Path, record: BaseModel) -> None:
    """Append a single pydantic model as one JSON line to the given JSONL file.

    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")


def append_jsonl_batch(path: Path, records: list[BaseModel]) -> None:
    """Append multiple pydantic models to the given JSONL file.

    Each record is written as a separate JSON line. Creates parent directories
    if they do not exist.
    """
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json() + "\n")
