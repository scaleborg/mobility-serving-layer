"""
Deterministic artifact path builders for serving observability.

All paths are derived from UTC timestamps. Pure functions with no side effects.
"""

from datetime import datetime
from pathlib import Path


def deployment_event_path(base_dir: Path, event_time: datetime) -> Path:
    """Build the JSONL file path for a deployment event.

    Layout: {base_dir}/artifacts/serving/events/{YYYY-MM-DD}/deployment_events.jsonl
    """
    date_str = event_time.strftime("%Y-%m-%d")
    return base_dir / "artifacts" / "serving" / "events" / date_str / "deployment_events.jsonl"


def metrics_window_path(base_dir: Path, window_start: datetime) -> Path:
    """Build the JSONL file path for a metrics window record.

    Layout: {base_dir}/artifacts/serving/metrics/{YYYY-MM-DD}/{HH}/metrics_{MM}.jsonl
    """
    date_str = window_start.strftime("%Y-%m-%d")
    hour_str = window_start.strftime("%H")
    minute_str = window_start.strftime("%M")
    return (
        base_dir
        / "artifacts"
        / "serving"
        / "metrics"
        / date_str
        / hour_str
        / f"metrics_{minute_str}.jsonl"
    )
