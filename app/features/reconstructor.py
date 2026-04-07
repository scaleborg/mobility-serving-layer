"""
Point-in-time feature reconstruction from raw upstream data.

Replaces the pre-materialized P3 snapshot lookup with full feature
computation using the same SQL as P2 training (score_features.sql).
Guarantees train/serve feature parity: same features, same order,
same transformations, strictly point-in-time correct.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from app.schemas.errors import EntityNotFoundError, FeatureParityError
from app.schemas.feature_schema import FeatureEntry

logger = logging.getLogger(__name__)

_SQL_DIR = Path(__file__).resolve().parent.parent / "sql"
_SCORE_SQL = (_SQL_DIR / "score_features.sql").read_text()

# Serving domain constants (must match P2 training filters)
MIN_CAPACITY = 5.0
MAX_SNAPSHOT_STALENESS_MIN = 15

_DTYPE_MAP: dict[str, type] = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": bool,
    "object": object,
    "str": str,
}


def open_upstream_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Open a read-only connection to the upstream raw metrics DuckDB."""
    conn = duckdb.connect(str(db_path), read_only=True)
    logger.info("Upstream DuckDB opened (read-only): %s", db_path)
    return conn


def probe_upstream_db(conn: duckdb.DuckDBPyConnection) -> None:
    """Verify the raw_station_metrics_1min table is accessible."""
    conn.execute("SELECT COUNT(*) FROM raw_station_metrics_1min").fetchone()


def reconstruct_features(
    conn: duckdb.DuckDBPyConnection,
    station_id: str,
    obs_ts: datetime,
    schema: list[FeatureEntry],
) -> tuple[np.ndarray, str]:
    """
    Reconstruct the full feature vector for (station_id, obs_ts) from raw data.

    Returns (feature_vector, snapshot_source_ts_iso).
    feature_vector is shaped (1, n_features) in schema-declared order.

    Raises:
        EntityNotFoundError: station not found or no data at/before obs_ts
        FeatureParityError: domain violation, staleness, or dtype issue
    """
    result = conn.execute(
        _SCORE_SQL, {"station_id": station_id, "obs_ts": obs_ts}
    ).fetchdf()

    if result.empty:
        raise EntityNotFoundError(
            message=f"No raw data found for station '{station_id}'."
        )

    row = result.iloc[0]

    # Snapshot source NULL means station exists but no data at/before obs_ts
    snapshot_source_ts = row["snapshot_source_ts"]
    if pd.isna(snapshot_source_ts):
        raise EntityNotFoundError(
            message=f"No data at or before {obs_ts} for station '{station_id}'."
        )

    snapshot_source_ts = pd.Timestamp(snapshot_source_ts).to_pydatetime()

    # Domain check: capacity must meet training threshold
    capacity = row.get("ft_capacity")
    if capacity is not None and not pd.isna(capacity) and capacity < MIN_CAPACITY:
        raise FeatureParityError(
            message=(
                f"Station capacity ({capacity:.1f}) is below training threshold "
                f"({MIN_CAPACITY}); prediction would be out-of-domain."
            )
        )

    # Staleness check: reject if snapshot source is too far from obs_ts
    obs_ts_pd = pd.Timestamp(obs_ts)
    source_ts_pd = pd.Timestamp(snapshot_source_ts)
    staleness = obs_ts_pd - source_ts_pd
    max_staleness = timedelta(minutes=MAX_SNAPSHOT_STALENESS_MIN)
    if staleness > max_staleness:
        raise FeatureParityError(
            message=(
                f"Latest data for station is from {snapshot_source_ts.isoformat()}, "
                f"which is >{MAX_SNAPSHOT_STALENESS_MIN} min before obs_ts {obs_ts}."
            )
        )

    # Build feature vector in schema-declared order with dtype coercion
    schema_names = [e.name for e in schema]
    missing = [name for name in schema_names if name not in row.index]
    if missing:
        raise FeatureParityError(
            message=f"Features in schema but missing from reconstructed row: {missing}"
        )

    values: list[Any] = []
    for entry in schema:
        raw_value = row[entry.name]
        np_dtype = _DTYPE_MAP.get(entry.dtype)

        if np_dtype is None:
            raise FeatureParityError(
                message=f"Unknown dtype '{entry.dtype}' for feature '{entry.name}'."
            )

        # Handle NaN/None: pass through as NaN for float types (LightGBM handles natively)
        if pd.isna(raw_value):
            if np_dtype in (np.float32, np.float64):
                values.append(float("nan"))
            elif np_dtype in (np.int32, np.int64):
                values.append(float("nan"))
            else:
                values.append(None)
            continue

        try:
            if np_dtype in (object, str):
                coerced = str(raw_value)
            else:
                coerced = np_dtype(raw_value)
        except (ValueError, TypeError) as exc:
            raise FeatureParityError(
                message=(
                    f"Cannot coerce feature '{entry.name}' value {raw_value!r} "
                    f"to dtype '{entry.dtype}': {exc}"
                )
            ) from exc

        values.append(coerced)

    feature_vector = np.array(values, dtype=object).reshape(1, -1)
    source_ts_iso = snapshot_source_ts.isoformat()

    logger.debug(
        "Reconstructed %d features for station '%s' at %s (source: %s)",
        len(values),
        station_id,
        obs_ts,
        source_ts_iso,
    )

    return feature_vector, source_ts_iso
