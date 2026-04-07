"""
Loads and validates model_metadata.json produced by P4.

Expected format:
{
  "model_version": "v1.2.3",
  "trained_at": "2025-01-01T00:00:00Z",
  "target": "demand_next_hour",
  ...
}
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = {"model_version"}


def load_model_metadata(metadata_path: Path) -> dict:
    logger.info("Loading model metadata from %s", metadata_path)
    with metadata_path.open() as fh:
        metadata = json.load(fh)

    missing = _REQUIRED_KEYS - metadata.keys()
    if missing:
        raise ValueError(
            f"model_metadata.json is missing required keys: {sorted(missing)}"
        )

    logger.info("Model metadata loaded: version=%s", metadata.get("model_version"))
    return metadata
