"""
Tests for P5 serving observability scaffolding.

Covers schema validation, path generation, JSONL writer, and emission helpers.
"""

import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.observability.emission import build_deployment_event, build_metrics_window
from app.observability.paths import deployment_event_path, metrics_window_path
from app.observability.writer import append_jsonl, append_jsonl_batch
from app.schemas.observability import (
    BundleLineageContext,
    ServingDeploymentEvent,
    ServingMetricsWindow,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NOW = datetime(2026, 4, 7, 14, 35, 22)
LATER = NOW + timedelta(minutes=5)


def _make_context(**overrides) -> BundleLineageContext:
    defaults = dict(
        service_name="mobility-serving-layer",
        service_version="0.1.0",
        environment="production",
        deployment_id="deploy-001",
        model_name="mobility",
        model_version="v1",
        bundle_id="bundle-abc",
        bundle_uri="s3://bucket/bundle-abc",
        bundle_created_at=NOW,
        input_dataset_name="rides-dataset",
        input_dataset_version="ds-v3",
        started_at=NOW - timedelta(hours=1),
        completed_at=NOW,
    )
    defaults.update(overrides)
    return BundleLineageContext(**defaults)


def _metrics_kwargs(**overrides) -> dict:
    defaults = dict(
        window_start=NOW,
        window_end=LATER,
        endpoint_name="/predict",
        request_count=100,
        success_count=95,
        failure_count=3,
        rejected_count=1,
        timeout_count=1,
        latency_p50_ms=12.5,
        latency_p95_ms=45.0,
        latency_p99_ms=120.0,
        validation_error_count=1,
        feature_lookup_error_count=0,
        model_load_error_count=0,
        inference_runtime_error_count=2,
        dependency_error_count=0,
        internal_error_count=0,
        input_schema_failure_count=0,
        missing_required_field_count=0,
        invalid_type_count=0,
        domain_violation_count=1,
        prediction_count=95,
        prediction_null_count=0,
        prediction_non_finite_count=0,
        prediction_out_of_range_count=0,
        fallback_prediction_count=0,
        heartbeat_emitted_at=LATER,
    )
    defaults.update(overrides)
    return defaults


# ---------------------------------------------------------------------------
# Schema validation — success
# ---------------------------------------------------------------------------


class TestSchemaValidationSuccess:
    def test_deployment_event_valid(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        assert event.event_type == "serving_deployment_activated"
        assert event.schema_version == "v1"
        assert event.activation_reason == "startup"

    def test_metrics_window_valid(self):
        ctx = _make_context()
        window = build_metrics_window(ctx, **_metrics_kwargs())
        assert window.schema_version == "v1"
        assert window.request_count == 100

    def test_bundle_context_valid(self):
        ctx = _make_context()
        assert ctx.service_name == "mobility-serving-layer"

    def test_bundle_context_placeholder(self):
        ctx = BundleLineageContext.placeholder()
        assert ctx.service_name == "mobility-serving-layer"
        assert ctx.environment == "development"


# ---------------------------------------------------------------------------
# Schema validation — failure
# ---------------------------------------------------------------------------


class TestSchemaValidationFailure:
    def test_window_end_before_start(self):
        ctx = _make_context()
        with pytest.raises(ValidationError, match="window_end"):
            build_metrics_window(ctx, **_metrics_kwargs(window_end=NOW - timedelta(minutes=1)))

    def test_window_end_equal_start(self):
        ctx = _make_context()
        with pytest.raises(ValidationError, match="window_end"):
            build_metrics_window(ctx, **_metrics_kwargs(window_end=NOW))

    def test_negative_request_count(self):
        ctx = _make_context()
        with pytest.raises(ValidationError):
            build_metrics_window(ctx, **_metrics_kwargs(request_count=-1))

    def test_negative_latency(self):
        ctx = _make_context()
        with pytest.raises(ValidationError):
            build_metrics_window(ctx, **_metrics_kwargs(latency_p50_ms=-1.0))

    def test_negative_optional_latency(self):
        ctx = _make_context()
        with pytest.raises(ValidationError):
            build_metrics_window(ctx, **_metrics_kwargs(feature_lookup_p95_ms=-0.1))

    def test_negative_optional_count(self):
        ctx = _make_context()
        with pytest.raises(ValidationError):
            build_metrics_window(ctx, **_metrics_kwargs(missing_feature_count=-1))

    def test_empty_service_name_deployment_event(self):
        with pytest.raises(ValidationError, match="service_name"):
            ServingDeploymentEvent(
                event_time=NOW,
                service_name="",
                service_version="v1",
                environment="prod",
                deployment_id="d1",
                model_name="m",
                model_version="v1",
                bundle_id="b1",
                bundle_uri="s3://x",
                bundle_created_at=NOW,
                input_dataset_name="ds",
                input_dataset_version="v1",
                started_at=NOW,
                completed_at=NOW,
                activation_reason="startup",
                traffic_status="serving",
            )

    def test_empty_lineage_field_context(self):
        with pytest.raises(ValidationError, match="input_dataset_name"):
            _make_context(input_dataset_name="")

    def test_empty_endpoint_name(self):
        ctx = _make_context()
        with pytest.raises(ValidationError, match="endpoint_name"):
            build_metrics_window(ctx, **_metrics_kwargs(endpoint_name=""))


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------


class TestPathGeneration:
    def test_deployment_event_path_deterministic(self):
        base = Path("/artifacts")
        t = datetime(2026, 4, 7, 14, 35, 22)
        result = deployment_event_path(base, t)
        assert result == base / "artifacts/serving/events/2026-04-07/deployment_events.jsonl"

    def test_deployment_event_path_date_changes(self):
        base = Path("/out")
        t1 = datetime(2026, 1, 1)
        t2 = datetime(2026, 12, 31)
        assert deployment_event_path(base, t1) != deployment_event_path(base, t2)

    def test_metrics_window_path_deterministic(self):
        base = Path("/artifacts")
        t = datetime(2026, 4, 7, 14, 35, 22)
        result = metrics_window_path(base, t)
        assert result == base / "artifacts/serving/metrics/2026-04-07/14/metrics_35.jsonl"

    def test_metrics_window_path_hour_minute(self):
        base = Path("/out")
        t = datetime(2026, 6, 15, 3, 7, 0)
        result = metrics_window_path(base, t)
        assert "03" in str(result)
        assert "metrics_07.jsonl" in str(result)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


class TestJsonlWriter:
    def test_append_creates_file_and_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "dir" / "out.jsonl"
            append_jsonl(path, {"key": "value"})
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            assert json.loads(lines[0]) == {"key": "value"}

    def test_append_preserves_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, {"a": 1})
            append_jsonl(path, {"b": 2})
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0]) == {"a": 1}
            assert json.loads(lines[1]) == {"b": 2}

    def test_append_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl_batch(path, [{"x": 1}, {"x": 2}, {"x": 3}])
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_append_batch_empty_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl_batch(path, [])
            assert not path.exists()

    def test_each_line_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, {"ts": datetime(2026, 1, 1)})
            line = path.read_text().strip()
            parsed = json.loads(line)
            assert "ts" in parsed


# ---------------------------------------------------------------------------
# Emission helpers
# ---------------------------------------------------------------------------


class TestEmissionHelpers:
    def test_build_deployment_event_from_context(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        assert event.service_name == ctx.service_name
        assert event.model_version == ctx.model_version
        assert event.bundle_id == ctx.bundle_id
        assert event.input_dataset_name == ctx.input_dataset_name
        assert event.activation_reason == "startup"
        assert event.traffic_status == "serving"

    def test_build_deployment_event_default_time(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "rollback", "draining")
        assert event.event_time is not None

    def test_build_metrics_window_from_context(self):
        ctx = _make_context()
        window = build_metrics_window(ctx, **_metrics_kwargs())
        assert window.service_name == ctx.service_name
        assert window.bundle_id == ctx.bundle_id
        assert window.endpoint_name == "/predict"

    def test_deployment_event_serializable(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        data = json.loads(event.model_dump_json())
        assert data["event_type"] == "serving_deployment_activated"
        assert data["schema_version"] == "v1"

    def test_metrics_window_serializable(self):
        ctx = _make_context()
        window = build_metrics_window(ctx, **_metrics_kwargs())
        data = json.loads(window.model_dump_json())
        assert data["request_count"] == 100
