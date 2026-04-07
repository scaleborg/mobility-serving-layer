"""
Shared test fixtures.

Creates a minimal but complete set of P4 artifacts + upstream raw metrics DB
in a temp directory, boots the FastAPI app against them, and yields an httpx
TestClient.

The upstream DB contains raw_station_metrics_1min rows that allow the
feature reconstruction SQL to compute all 22 features.
"""

import json
import pickle
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import duckdb


class _StubModel:
    """Minimal sklearn-compatible estimator for testing."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([42.0] * X.shape[0])


FEATURE_SCHEMA = {
    "version": "test-schema-v1",
    "entity_key": "station_id",
    "features": [
        {"name": "ft_bikes_available", "dtype": "float64"},
        {"name": "ft_docks_available", "dtype": "float64"},
        {"name": "ft_availability_ratio", "dtype": "float64"},
        {"name": "ft_bikes_available_lag_15m", "dtype": "float64"},
        {"name": "ft_bikes_available_lag_30m", "dtype": "float64"},
        {"name": "ft_bikes_available_lag_60m", "dtype": "float64"},
        {"name": "ft_bikes_available_lag_24h", "dtype": "float64"},
        {"name": "ft_avg_bikes_60m", "dtype": "float64"},
        {"name": "ft_min_bikes_60m", "dtype": "float64"},
        {"name": "ft_max_bikes_60m", "dtype": "float64"},
        {"name": "ft_avg_ratio_60m", "dtype": "float64"},
        {"name": "ft_avg_bikes_24h", "dtype": "float64"},
        {"name": "ft_min_bikes_24h", "dtype": "float64"},
        {"name": "ft_max_bikes_24h", "dtype": "float64"},
        {"name": "ft_low_avail_freq_24h", "dtype": "float64"},
        {"name": "ft_hour_of_day", "dtype": "int32"},
        {"name": "ft_day_of_week", "dtype": "int32"},
        {"name": "ft_is_weekend", "dtype": "int32"},
        {"name": "ft_capacity", "dtype": "float64"},
        {"name": "ft_pct_bikes_of_capacity", "dtype": "float64"},
        {"name": "ft_pct_docks_of_capacity", "dtype": "float64"},
        {"name": "ft_bikes_delta_60m", "dtype": "float64"},
    ],
    "target": "target_empty_next_hour",
}

MODEL_METADATA = {
    "model_version": "test-v1",
    "trained_at": "2025-01-01T00:00:00Z",
    "target": "target_empty_next_hour",
}

# Observation timestamp used in tests
OBS_TS = datetime(2025, 6, 1, 12, 0, 0)


def _build_upstream_db(db_path: Path) -> None:
    """Create a DuckDB with raw_station_metrics_1min rows for two stations.

    Populates 25 hours of 1-minute data so lag/rolling features resolve.
    """
    conn = duckdb.connect(str(db_path))

    conn.execute("""
        CREATE TABLE raw_station_metrics_1min (
            station_id VARCHAR,
            window_start TIMESTAMP,
            window_end TIMESTAMP,
            avg_bikes_available DOUBLE,
            avg_docks_available DOUBLE,
            avg_capacity DOUBLE,
            avg_availability_ratio DOUBLE,
            low_availability_events BIGINT,
            event_count BIGINT,
            emitted_at TIMESTAMP
        )
    """)

    rows = []
    for station in ["station_1", "station_2"]:
        bikes = 10.0 if station == "station_1" else 5.0
        capacity = 20.0
        for minute_offset in range(25 * 60 + 1):  # 25 hours of 1-min data
            ts = OBS_TS - timedelta(hours=25) + timedelta(minutes=minute_offset)
            rows.append({
                "station_id": station,
                "window_start": ts,
                "window_end": ts + timedelta(minutes=1),
                "avg_bikes_available": bikes,
                "avg_docks_available": capacity - bikes,
                "avg_capacity": capacity,
                "avg_availability_ratio": bikes / capacity,
                "low_availability_events": 0,
                "event_count": 1,
                "emitted_at": ts + timedelta(seconds=5),
            })

    df = pd.DataFrame(rows)
    conn.execute("INSERT INTO raw_station_metrics_1min SELECT * FROM df")
    conn.close()


def _build_artifacts(tmp: Path) -> None:
    """Write P4 artifacts + P3 snapshot + upstream DB into *tmp*."""
    # Versioned artifact directory: models/mobility/v1/
    model_dir = tmp / "models" / "mobility" / "v1"
    model_dir.mkdir(parents=True)

    (model_dir / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA))
    (model_dir / "model_metadata.json").write_text(json.dumps(MODEL_METADATA))

    with (model_dir / "model.pkl").open("wb") as fh:
        pickle.dump(_StubModel(), fh)

    # P3 parquet snapshot (retained for startup check)
    data_dir = tmp / "data"
    data_dir.mkdir()
    # Build a snapshot with all 22 feature columns so the snapshot probe passes
    df = pd.DataFrame({
        "station_id": ["station_1", "station_2"],
        "snapshot_ts": ["2025-06-01T12:00:00Z", "2025-06-01T12:00:00Z"],
        "ft_bikes_available": [10.0, 5.0],
        "ft_docks_available": [10.0, 15.0],
        "ft_availability_ratio": [0.5, 0.25],
        "ft_bikes_available_lag_15m": [10.0, 5.0],
        "ft_bikes_available_lag_30m": [10.0, 5.0],
        "ft_bikes_available_lag_60m": [10.0, 5.0],
        "ft_bikes_available_lag_24h": [10.0, 5.0],
        "ft_avg_bikes_60m": [10.0, 5.0],
        "ft_min_bikes_60m": [10.0, 5.0],
        "ft_max_bikes_60m": [10.0, 5.0],
        "ft_avg_ratio_60m": [0.5, 0.25],
        "ft_avg_bikes_24h": [10.0, 5.0],
        "ft_min_bikes_24h": [10.0, 5.0],
        "ft_max_bikes_24h": [10.0, 5.0],
        "ft_low_avail_freq_24h": [0.0, 0.0],
        "ft_hour_of_day": [12, 12],
        "ft_day_of_week": [0, 0],
        "ft_is_weekend": [1, 1],
        "ft_capacity": [20.0, 20.0],
        "ft_pct_bikes_of_capacity": [0.5, 0.25],
        "ft_pct_docks_of_capacity": [0.5, 0.75],
        "ft_bikes_delta_60m": [0.0, 0.0],
    })
    df.to_parquet(data_dir / "online_features.parquet", index=False)

    # Upstream raw metrics DB
    _build_upstream_db(tmp / "mobility.duckdb")


@pytest.fixture(scope="session")
def artifact_dir():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _build_artifacts(tmp_path)
        yield tmp_path


@pytest.fixture(scope="session")
def client(artifact_dir: Path):
    import os

    model_dir = artifact_dir / "models" / "mobility" / "v1"
    os.environ["SERVING_MODEL_PATH"] = str(model_dir / "model.pkl")
    os.environ["SERVING_FEATURE_SCHEMA_PATH"] = str(model_dir / "feature_schema.json")
    os.environ["SERVING_MODEL_METADATA_PATH"] = str(model_dir / "model_metadata.json")
    os.environ["SERVING_FEATURE_SNAPSHOT_PATH"] = str(
        artifact_dir / "data" / "online_features.parquet"
    )
    os.environ["SERVING_UPSTREAM_DB_PATH"] = str(artifact_dir / "mobility.duckdb")

    from importlib import reload

    import app.core.config as cfg_mod

    reload(cfg_mod)

    from app.core.config import settings
    from app.core.startup import run_startup
    from app.core.state import app_state

    run_startup(app_state, settings)

    from app.main import app

    with TestClient(app) as tc:
        yield tc
