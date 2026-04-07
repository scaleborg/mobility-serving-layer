"""
Tests for observability wiring: BundleLineageContext from real metadata
and deployment event emission on startup.
"""

import json
import pickle
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.config import Settings
from app.core.state import AppState
from app.observability.context import (
    _derive_bundle_id,
    _derive_bundle_uri,
    _derive_model_name,
    _parse_trained_at,
    build_context_from_metadata,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REAL_METADATA = {
    "model_version": "v1-lgbm-20260404",
    "trained_at": "2026-04-04T01:22:26Z",
    "target": "target_empty_next_hour",
    "model_type": "lightgbm",
    "input_dataset_name": "rides-v3",
    "input_dataset_version": "ds-2026-04-01",
    "started_at": "2026-04-04T00:45:00Z",
    "completed_at": "2026-04-04T01:22:26Z",
}

FEATURE_SCHEMA = {
    "version": "v1",
    "entity_key": "station_id",
    "features": [
        {"name": "feat_a", "dtype": "float64"},
        {"name": "feat_b", "dtype": "float64"},
    ],
    "target": "target_empty_next_hour",
}


class _StubModel:
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([0.0] * X.shape[0])


def _make_state(**overrides) -> AppState:
    defaults = dict(
        model="fake-model",
        model_metadata=REAL_METADATA.copy(),
    )
    defaults.update(overrides)
    state = AppState()
    for k, v in defaults.items():
        setattr(state, k, v)
    return state


def _make_settings(**overrides) -> Settings:
    defaults = dict(
        model_path=Path("/srv/models/mobility/v1/model.pkl"),
        feature_schema_path=Path("/srv/models/mobility/v1/feature_schema.json"),
        model_metadata_path=Path("/srv/models/mobility/v1/model_metadata.json"),
        feature_snapshot_path=Path("/srv/data/online_features.parquet"),
        service_name="mobility-serving-layer",
        service_version="0.1.0",
        environment="production",
        deployment_id="deploy-abc",
        instance_id="pod-xyz",
        artifact_base_dir=Path("/srv"),
    )
    defaults.update(overrides)
    return Settings(**defaults)


def _build_test_artifacts(tmp: Path) -> None:
    """Write minimal P4 artifacts + P3 snapshot for run_startup()."""
    model_dir = tmp / "models" / "mobility" / "v1"
    model_dir.mkdir(parents=True)

    (model_dir / "feature_schema.json").write_text(json.dumps(FEATURE_SCHEMA))
    (model_dir / "model_metadata.json").write_text(json.dumps(REAL_METADATA))

    with (model_dir / "model.pkl").open("wb") as fh:
        pickle.dump(_StubModel(), fh)

    data_dir = tmp / "data"
    data_dir.mkdir()
    df = pd.DataFrame({
        "station_id": ["s1"],
        "snapshot_ts": ["2026-04-07T12:00:00Z"],
        "feat_a": [1.0],
        "feat_b": [2.0],
    })
    df.to_parquet(data_dir / "online_features.parquet", index=False)


# ---------------------------------------------------------------------------
# Context building — success
# ---------------------------------------------------------------------------


class TestBuildContextFromMetadata:
    def test_builds_context_with_real_fields(self):
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)

        assert ctx.service_name == "mobility-serving-layer"
        assert ctx.service_version == "0.1.0"
        assert ctx.environment == "production"
        assert ctx.deployment_id == "deploy-abc"
        assert ctx.instance_id == "pod-xyz"
        assert ctx.model_version == "v1-lgbm-20260404"
        assert ctx.model_name == "mobility"
        assert ctx.input_dataset_name == "rides-v3"
        assert ctx.input_dataset_version == "ds-2026-04-01"

    def test_dataset_lineage_from_metadata_not_settings(self):
        """Dataset lineage fields are read from bundle metadata, not settings."""
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)

        assert ctx.input_dataset_name == "rides-v3"
        assert ctx.input_dataset_version == "ds-2026-04-01"
        assert ctx.started_at == datetime(2026, 4, 4, 0, 45, 0, tzinfo=UTC)
        assert ctx.completed_at == datetime(2026, 4, 4, 1, 22, 26, tzinfo=UTC)

    def test_started_at_and_completed_at_from_metadata(self):
        """started_at and completed_at are distinct timestamps from metadata."""
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)

        expected_started = datetime(2026, 4, 4, 0, 45, 0, tzinfo=UTC)
        expected_completed = datetime(2026, 4, 4, 1, 22, 26, tzinfo=UTC)
        assert ctx.started_at == expected_started
        assert ctx.completed_at == expected_completed
        assert ctx.started_at < ctx.completed_at

    def test_trained_at_mapped_to_bundle_created_at(self):
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)

        expected = datetime(2026, 4, 4, 1, 22, 26, tzinfo=UTC)
        assert ctx.bundle_created_at == expected

    def test_bundle_uri_is_file_uri(self):
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)

        assert ctx.bundle_uri.startswith("file://")

    def test_bundle_id_is_deterministic(self):
        state = _make_state()
        settings = _make_settings()
        ctx1 = build_context_from_metadata(state, settings)
        ctx2 = build_context_from_metadata(state, settings)
        assert ctx1.bundle_id == ctx2.bundle_id

    def test_timestamps_are_utc(self):
        state = _make_state()
        settings = _make_settings()
        ctx = build_context_from_metadata(state, settings)
        assert ctx.bundle_created_at.tzinfo is UTC
        assert ctx.started_at.tzinfo is UTC
        assert ctx.completed_at.tzinfo is UTC


# ---------------------------------------------------------------------------
# Context building — failure
# ---------------------------------------------------------------------------


class TestBuildContextFailures:
    def test_empty_metadata_raises(self):
        state = _make_state(model_metadata={})
        settings = _make_settings()
        with pytest.raises(ValueError, match="model_metadata is empty"):
            build_context_from_metadata(state, settings)

    def test_missing_model_version_raises(self):
        meta = REAL_METADATA.copy()
        del meta["model_version"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="model_version"):
            build_context_from_metadata(state, settings)

    def test_missing_trained_at_raises(self):
        meta = REAL_METADATA.copy()
        del meta["trained_at"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="trained_at"):
            build_context_from_metadata(state, settings)

    def test_naive_trained_at_raises(self):
        meta = REAL_METADATA.copy()
        meta["trained_at"] = "2026-04-04T01:22:26"  # no timezone
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="timezone-aware"):
            build_context_from_metadata(state, settings)

    def test_missing_input_dataset_name_raises(self):
        meta = REAL_METADATA.copy()
        del meta["input_dataset_name"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="input_dataset_name"):
            build_context_from_metadata(state, settings)

    def test_missing_input_dataset_version_raises(self):
        meta = REAL_METADATA.copy()
        del meta["input_dataset_version"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="input_dataset_version"):
            build_context_from_metadata(state, settings)

    def test_missing_started_at_raises(self):
        meta = REAL_METADATA.copy()
        del meta["started_at"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="started_at"):
            build_context_from_metadata(state, settings)

    def test_missing_completed_at_raises(self):
        meta = REAL_METADATA.copy()
        del meta["completed_at"]
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="completed_at"):
            build_context_from_metadata(state, settings)

    def test_empty_input_dataset_name_raises(self):
        meta = REAL_METADATA.copy()
        meta["input_dataset_name"] = ""
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="input_dataset_name"):
            build_context_from_metadata(state, settings)

    def test_whitespace_input_dataset_version_raises(self):
        meta = REAL_METADATA.copy()
        meta["input_dataset_version"] = "   "
        state = _make_state(model_metadata=meta)
        settings = _make_settings()
        with pytest.raises(ValueError, match="input_dataset_version"):
            build_context_from_metadata(state, settings)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_derive_model_name_from_path(self):
        p = Path("/srv/models/mobility/v1/model.pkl")
        assert _derive_model_name(p) == "mobility"

    def test_derive_bundle_uri_is_file_uri(self):
        p = Path("/srv/models/mobility/v1/model.pkl")
        uri = _derive_bundle_uri(p)
        assert uri.startswith("file:///")
        assert "v1" in uri

    def test_derive_bundle_id_deterministic(self):
        p = Path("/srv/models/mobility/v1/model.pkl")
        assert _derive_bundle_id(p) == _derive_bundle_id(p)

    def test_parse_trained_at_z_suffix(self):
        dt = _parse_trained_at({"trained_at": "2026-04-04T01:22:26Z"})
        assert dt.tzinfo is UTC
        assert dt.year == 2026

    def test_parse_trained_at_offset_suffix(self):
        dt = _parse_trained_at({"trained_at": "2026-04-04T01:22:26+00:00"})
        assert dt.tzinfo is UTC


# ---------------------------------------------------------------------------
# Startup emission — end-to-end artifact write
# ---------------------------------------------------------------------------


class TestStartupEmission:
    def test_startup_emits_one_deployment_event(self):
        from app.observability.context import build_context_from_metadata
        from app.observability.emission import build_deployment_event
        from app.observability.paths import deployment_event_path
        from app.observability.writer import append_jsonl

        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(artifact_base_dir=Path(tmp))
            context = build_context_from_metadata(state, settings)
            event = build_deployment_event(context, "startup", "serving")
            path = deployment_event_path(settings.artifact_base_dir, event.event_time)
            append_jsonl(path, event)

            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1

    def test_emitted_json_contains_real_lineage(self):
        from app.observability.context import build_context_from_metadata
        from app.observability.emission import build_deployment_event
        from app.observability.paths import deployment_event_path
        from app.observability.writer import append_jsonl

        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(artifact_base_dir=Path(tmp))
            context = build_context_from_metadata(state, settings)
            event = build_deployment_event(context, "startup", "serving")
            path = deployment_event_path(settings.artifact_base_dir, event.event_time)
            append_jsonl(path, event)

            record = json.loads(path.read_text().strip())

            assert record["event_type"] == "serving_deployment_activated"
            assert record["schema_version"] == "v1"
            assert record["model_version"] == "v1-lgbm-20260404"
            assert record["service_name"] == "mobility-serving-layer"
            assert record["environment"] == "production"
            assert record["deployment_id"] == "deploy-abc"
            assert record["activation_reason"] == "startup"
            assert record["traffic_status"] == "serving"
            assert record["input_dataset_name"] == "rides-v3"
            assert record["input_dataset_version"] == "ds-2026-04-01"

    def test_emitted_json_timestamps_are_utc_iso8601(self):
        from app.observability.context import build_context_from_metadata
        from app.observability.emission import build_deployment_event
        from app.observability.paths import deployment_event_path
        from app.observability.writer import append_jsonl

        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(artifact_base_dir=Path(tmp))
            context = build_context_from_metadata(state, settings)
            event = build_deployment_event(context, "startup", "serving")
            path = deployment_event_path(settings.artifact_base_dir, event.event_time)
            append_jsonl(path, event)

            record = json.loads(path.read_text().strip())
            for ts_field in ("event_time", "bundle_created_at", "started_at", "completed_at"):
                ts = record[ts_field]
                assert ts.endswith("+00:00") or ts.endswith("Z"), (
                    f"{ts_field} not UTC: {ts}"
                )

    def test_second_startup_appends_not_overwrites(self):
        from app.observability.context import build_context_from_metadata
        from app.observability.emission import build_deployment_event
        from app.observability.paths import deployment_event_path
        from app.observability.writer import append_jsonl

        state = _make_state()

        with tempfile.TemporaryDirectory() as tmp:
            settings = _make_settings(artifact_base_dir=Path(tmp))
            context = build_context_from_metadata(state, settings)

            event1 = build_deployment_event(context, "startup", "serving")
            path = deployment_event_path(settings.artifact_base_dir, event1.event_time)
            append_jsonl(path, event1)

            event2 = build_deployment_event(context, "restart", "serving")
            path2 = deployment_event_path(settings.artifact_base_dir, event2.event_time)
            # Same date → same file
            assert path == path2
            append_jsonl(path2, event2)

            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["activation_reason"] == "startup"
            assert json.loads(lines[1])["activation_reason"] == "restart"


# ---------------------------------------------------------------------------
# run_startup() integration — real startup hook
# ---------------------------------------------------------------------------


class TestRunStartupEmission:
    def test_run_startup_emits_deployment_event_artifact(self):
        """Call run_startup() with real artifacts and verify the deployment
        event JSONL file is written with correct lineage fields."""
        from app.core.startup import run_startup

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            _build_test_artifacts(tmp_path)

            model_dir = tmp_path / "models" / "mobility" / "v1"
            test_settings = Settings(
                model_path=model_dir / "model.pkl",
                feature_schema_path=model_dir / "feature_schema.json",
                model_metadata_path=model_dir / "model_metadata.json",
                feature_snapshot_path=tmp_path / "data" / "online_features.parquet",
                service_name="mobility-serving-layer",
                service_version="0.1.0",
                environment="production",
                deployment_id="test-deploy-001",
                artifact_base_dir=tmp_path,
            )

            state = AppState()
            run_startup(state, test_settings)

            # Find the emitted artifact
            event_dir = tmp_path / "artifacts" / "serving" / "events"
            assert event_dir.exists(), "No events directory created"

            jsonl_files = list(event_dir.rglob("*.jsonl"))
            assert len(jsonl_files) == 1, f"Expected 1 JSONL file, found {len(jsonl_files)}"

            lines = jsonl_files[0].read_text().strip().split("\n")
            assert len(lines) == 1

            record = json.loads(lines[0])
            assert record["event_type"] == "serving_deployment_activated"
            assert record["schema_version"] == "v1"
            assert record["model_version"] == "v1-lgbm-20260404"
            assert record["model_name"] == "mobility"
            assert record["service_name"] == "mobility-serving-layer"
            assert record["deployment_id"] == "test-deploy-001"
            assert record["activation_reason"] == "startup"
            assert record["traffic_status"] == "serving"
            assert record["input_dataset_name"] == "rides-v3"
            assert record["input_dataset_version"] == "ds-2026-04-01"
            assert record["bundle_created_at"].endswith("+00:00") or \
                record["bundle_created_at"].endswith("Z")

            # Cleanup DuckDB connection
            if state.db_conn is not None:
                state.db_conn.close()
