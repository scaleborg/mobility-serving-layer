"""
Deterministic artifact path builders for serving observability.

All paths are derived from UTC timestamps. Pure functions with no side effects.
Naive datetimes are rejected; aware datetimes are normalized to UTC.
"""

from datetime import UTC, datetime
from pathlib import Path


def _to_utc(dt: datetime, param_name: str) -> datetime:
    """Reject naive datetimes; normalize aware datetimes to UTC."""
    if dt.tzinfo is None:
        raise ValueError(f"{param_name} must be timezone-aware (got naive datetime)")
    return dt.astimezone(UTC)


def deployment_event_path(base_dir: Path, event_time: datetime) -> Path:
    """Build the JSONL file path for a deployment event.

    Layout: {base_dir}/artifacts/serving/events/{YYYY-MM-DD}/deployment_events.jsonl
    """
    event_time = _to_utc(event_time, "event_time")
    date_str = event_time.strftime("%Y-%m-%d")
    return base_dir / "artifacts" / "serving" / "events" / date_str / "deployment_events.jsonl"


def metrics_window_path(base_dir: Path, window_start: datetime) -> Path:
    """Build the JSONL file path for a metrics window record.

    Layout: {base_dir}/artifacts/serving/metrics/{YYYY-MM-DD}/{HH}/metrics_{MM}.jsonl
    """
    window_start = _to_utc(window_start, "window_start")
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
