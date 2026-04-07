"""
Shared test fixtures.

Creates a minimal but complete set of P3/P4 artifacts in a temp directory,
boots the FastAPI app against them, and yields an httpx TestClient.
"""

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


class _StubModel:
    """Minimal sklearn-compatible estimator for testing."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([42.0] * X.shape[0])


FEATURE_SCHEMA = {
    "version": "test-schema-v1",
    "entity_key": "station_id",
    "features": [
        {"name": "feat_a", "dtype": "float64"},
        {"name": "feat_b", "dtype": "float64"},
        {"name": "feat_c", "dtype": "int64"},
    ],
    "target": "demand_next_hour",
}

MODEL_METADATA = {
    "model_version": "test-v1",
    "trained_at": "2025-01-01T00:00:00Z",
    "target": "demand_next_hour",
}


def _build_artifacts(tmp: Path) -> None:
    """Write fake P4 artifacts + P3 snapshot into *tmp*."""
    # Versioned artifact directory: models/mobility/v1/
    model_dir = tmp / "models" / "mobility" / "v1"
    model_dir.mkdir(parents=True)

    (model_dir / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA))
    (model_dir / "model_metadata.json").write_text(json.dumps(MODEL_METADATA))

    with (model_dir / "model.pkl").open("wb") as fh:
        pickle.dump(_StubModel(), fh)

    # P3 parquet snapshot
    data_dir = tmp / "data"
    data_dir.mkdir()
    df = pd.DataFrame(
        {
            "station_id": ["station_1", "station_2"],
            "snapshot_ts": ["2025-06-01T12:00:00Z", "2025-06-01T12:00:00Z"],
            "feat_a": [1.1, 2.2],
            "feat_b": [3.3, 4.4],
            "feat_c": [10, 20],
        }
    )
    df.to_parquet(data_dir / "online_features.parquet", index=False)


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
