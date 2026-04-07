"""
Tests for P5 serving observability scaffolding.

Covers schema validation, path generation, JSONL writer, and emission helpers.
"""

import json
import tempfile
from datetime import UTC, datetime, timedelta, timezone
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

NOW = datetime(2026, 4, 7, 14, 35, 22, tzinfo=UTC)
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

    def test_non_utc_aware_datetime_normalized_to_utc(self):
        eastern = timezone(timedelta(hours=-5))
        t = datetime(2026, 4, 7, 9, 35, 22, tzinfo=eastern)  # same instant as NOW
        ctx = _make_context(bundle_created_at=t, started_at=t, completed_at=t)
        assert ctx.bundle_created_at.tzinfo is UTC


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

    def test_naive_datetime_rejected_in_context(self):
        naive = datetime(2026, 4, 7, 14, 0, 0)
        with pytest.raises(ValidationError, match="timezone-aware"):
            _make_context(bundle_created_at=naive)

    def test_naive_datetime_rejected_in_deployment_event(self):
        with pytest.raises(ValidationError, match="timezone-aware"):
            ServingDeploymentEvent(
                event_time=datetime(2026, 4, 7, 14, 0, 0),  # naive
                service_name="svc",
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

    def test_naive_datetime_rejected_in_metrics_window(self):
        ctx = _make_context()
        naive = datetime(2026, 4, 7, 14, 0, 0)
        with pytest.raises(ValidationError, match="timezone-aware"):
            build_metrics_window(ctx, **_metrics_kwargs(window_start=naive))

    def test_invalid_event_type_rejected(self):
        with pytest.raises(ValidationError):
            ServingDeploymentEvent(
                event_type="wrong_value",
                event_time=NOW,
                service_name="svc",
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

    def test_invalid_schema_version_rejected_deployment(self):
        with pytest.raises(ValidationError):
            ServingDeploymentEvent(
                event_time=NOW,
                service_name="svc",
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
                schema_version="v2",
            )

    def test_invalid_schema_version_rejected_metrics(self):
        ctx = _make_context()
        with pytest.raises(ValidationError):
            build_metrics_window(ctx, **_metrics_kwargs())
            # build via direct construction to override schema_version
            ServingMetricsWindow(
                schema_version="v2",
                window_start=NOW,
                window_end=LATER,
                service_name="svc",
                service_version="v1",
                environment="prod",
                deployment_id="d1",
                model_name="m",
                model_version="v1",
                bundle_id="b1",
                input_dataset_name="ds",
                input_dataset_version="v1",
                endpoint_name="/predict",
                request_count=0,
                success_count=0,
                failure_count=0,
                rejected_count=0,
                timeout_count=0,
                latency_p50_ms=0.0,
                latency_p95_ms=0.0,
                latency_p99_ms=0.0,
                validation_error_count=0,
                feature_lookup_error_count=0,
                model_load_error_count=0,
                inference_runtime_error_count=0,
                dependency_error_count=0,
                internal_error_count=0,
                input_schema_failure_count=0,
                missing_required_field_count=0,
                invalid_type_count=0,
                domain_violation_count=0,
                prediction_count=0,
                prediction_null_count=0,
                prediction_non_finite_count=0,
                prediction_out_of_range_count=0,
                fallback_prediction_count=0,
                heartbeat_emitted_at=LATER,
            )

    def test_completed_before_started_rejected_context(self):
        with pytest.raises(ValidationError, match="completed_at"):
            _make_context(
                started_at=NOW,
                completed_at=NOW - timedelta(hours=1),
            )

    def test_completed_before_started_rejected_event(self):
        with pytest.raises(ValidationError, match="completed_at"):
            ServingDeploymentEvent(
                event_time=NOW,
                service_name="svc",
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
                completed_at=NOW - timedelta(hours=1),
                activation_reason="startup",
                traffic_status="serving",
            )


# ---------------------------------------------------------------------------
# Path generation
# ---------------------------------------------------------------------------


class TestPathGeneration:
    def test_deployment_event_path_deterministic(self):
        base = Path("/artifacts")
        t = datetime(2026, 4, 7, 14, 35, 22, tzinfo=UTC)
        result = deployment_event_path(base, t)
        assert result == base / "artifacts/serving/events/2026-04-07/deployment_events.jsonl"

    def test_deployment_event_path_date_changes(self):
        base = Path("/out")
        t1 = datetime(2026, 1, 1, tzinfo=UTC)
        t2 = datetime(2026, 12, 31, tzinfo=UTC)
        assert deployment_event_path(base, t1) != deployment_event_path(base, t2)

    def test_metrics_window_path_deterministic(self):
        base = Path("/artifacts")
        t = datetime(2026, 4, 7, 14, 35, 22, tzinfo=UTC)
        result = metrics_window_path(base, t)
        assert result == base / "artifacts/serving/metrics/2026-04-07/14/metrics_35.jsonl"

    def test_metrics_window_path_hour_minute(self):
        base = Path("/out")
        t = datetime(2026, 6, 15, 3, 7, 0, tzinfo=UTC)
        result = metrics_window_path(base, t)
        assert "03" in str(result)
        assert "metrics_07.jsonl" in str(result)

    def test_naive_datetime_rejected_deployment_path(self):
        base = Path("/out")
        naive = datetime(2026, 4, 7, 14, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            deployment_event_path(base, naive)

    def test_naive_datetime_rejected_metrics_path(self):
        base = Path("/out")
        naive = datetime(2026, 4, 7, 14, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            metrics_window_path(base, naive)

    def test_non_utc_normalized_deployment_path(self):
        base = Path("/out")
        # UTC+5 at 2026-04-08 02:00 == UTC 2026-04-07 21:00
        utc_plus_5 = timezone(timedelta(hours=5))
        t = datetime(2026, 4, 8, 2, 0, 0, tzinfo=utc_plus_5)
        result = deployment_event_path(base, t)
        assert "2026-04-07" in str(result)

    def test_non_utc_normalized_metrics_path(self):
        base = Path("/out")
        # UTC+5 at 2026-04-08 02:30 == UTC 2026-04-07 21:30
        utc_plus_5 = timezone(timedelta(hours=5))
        t = datetime(2026, 4, 8, 2, 30, 0, tzinfo=utc_plus_5)
        result = metrics_window_path(base, t)
        assert "2026-04-07" in str(result)
        assert "/21/" in str(result)
        assert "metrics_30.jsonl" in str(result)


# ---------------------------------------------------------------------------
# JSONL writer
# ---------------------------------------------------------------------------


class TestJsonlWriter:
    def test_append_creates_file_and_dirs(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "dir" / "out.jsonl"
            append_jsonl(path, event)
            assert path.exists()
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["event_type"] == "serving_deployment_activated"

    def test_append_preserves_existing(self):
        ctx = _make_context()
        e1 = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        e2 = build_deployment_event(ctx, "rollback", "draining", event_time=LATER)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, e1)
            append_jsonl(path, e2)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 2
            assert json.loads(lines[0])["activation_reason"] == "startup"
            assert json.loads(lines[1])["activation_reason"] == "rollback"

    def test_append_batch(self):
        ctx = _make_context()
        events = [
            build_deployment_event(ctx, f"reason-{i}", "serving", event_time=NOW)
            for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl_batch(path, events)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == 3

    def test_append_batch_empty_noop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl_batch(path, [])
            assert not path.exists()

    def test_timestamps_are_iso8601_utc(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, event)
            line = path.read_text().strip()
            parsed = json.loads(line)
            assert parsed["event_time"].endswith("+00:00") or parsed["event_time"].endswith("Z")

    def test_each_line_is_valid_json(self):
        ctx = _make_context()
        event = build_deployment_event(ctx, "startup", "serving", event_time=NOW)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.jsonl"
            append_jsonl(path, event)
            line = path.read_text().strip()
            parsed = json.loads(line)
            assert "event_type" in parsed


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
        assert event.event_time.tzinfo is not None

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
