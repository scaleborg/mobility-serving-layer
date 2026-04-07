"""
Extracts the entity timestamp from the raw snapshot row.
Centralises the column-name contract so other modules stay decoupled.
"""

from app.features.loader import TIMESTAMP_COLUMN


def extract_timestamp(raw_row: dict) -> str:
    """
    Returns the ISO-8601 UTC timestamp string for the snapshot row.
    Falls back to 'unknown' rather than crashing — callers may log a warning.
    """
    ts = raw_row.get(TIMESTAMP_COLUMN)
    if ts is None:
        return "unknown"
    return str(ts)
