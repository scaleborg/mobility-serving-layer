"""
Tests for the in-memory 60-second metrics window aggregator.

Covers: window bucketing, percentile calculation, counter tracking,
schema compliance, window non-overlap, and serving hook integration.
"""

import json
import math
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.observability.aggregator import (
    MetricsAggregator,
    _floor_minute,
    _percentile,
)
from app.schemas.observability import BundleLineageContext, ServingMetricsWindow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context(**overrides) -> BundleLineageContext:
    defaults = dict(
        service_name="mobility-serving-layer",
        service_version="0.1.0",
        environment="production",
        deployment_id="deploy-abc",
        instance_id="pod-xyz",
        model_name="mobility",
        model_version="v1-lgbm-20260404",
        bundle_id="/srv/models/mobility/v1",
        bundle_uri="file:///srv/models/mobility/v1",
        bundle_created_at=datetime(2026, 4, 4, 1, 22, 26, tzinfo=UTC),
        input_dataset_name="rides-v3",
        input_dataset_version="ds-2026-04-01",
        started_at=datetime(2026, 4, 4, 0, 45, 0, tzinfo=UTC),
        completed_at=datetime(2026, 4, 4, 1, 22, 26, tzinfo=UTC),
    )
    defaults.update(overrides)
    return BundleLineageContext(**defaults)


# ---------------------------------------------------------------------------
# Window bucketing
# ---------------------------------------------------------------------------


class TestWindowBucketing:
    def test_floor_minute_truncates_seconds(self):
        dt = datetime(2026, 4, 7, 14, 35, 22, 500_000, tzinfo=UTC)
        assert _floor_minute(dt) == datetime(2026, 4, 7, 14, 35, 0, tzinfo=UTC)

    def test_floor_minute_exact_boundary(self):
        dt = datetime(2026, 4, 7, 14, 35, 0, tzinfo=UTC)
        assert _floor_minute(dt) == dt

    def test_requests_in_same_minute_share_window(self):
        agg = MetricsAggregator()
        t1 = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        t2 = datetime(2026, 4, 7, 14, 35, 50, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t1)
        agg.record_success(7.0, 0.6, now=t2)

        ctx = _make_context()
        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert len(windows) == 1
        assert windows[0].request_count == 2

    def test_window_boundaries_are_minute_aligned(self):
        agg = MetricsAggregator()
        t = datetime(2026, 4, 7, 14, 35, 22, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        ctx = _make_context()
        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert len(windows) == 1
        assert windows[0].window_start == datetime(2026, 4, 7, 14, 35, 0, tzinfo=UTC)
        assert windows[0].window_end == datetime(2026, 4, 7, 14, 36, 0, tzinfo=UTC)

    def test_flush_before_window_end_returns_empty(self):
        agg = MetricsAggregator()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        ctx = _make_context()
        flush_time = datetime(2026, 4, 7, 14, 35, 50, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert windows == []

    def test_no_requests_returns_empty_on_flush(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        assert windows == []


# ---------------------------------------------------------------------------
# Multiple windows do not overlap
# ---------------------------------------------------------------------------


class TestWindowNonOverlap:
    def test_consecutive_windows_do_not_overlap(self):
        agg = MetricsAggregator()
        ctx = _make_context()

        # Window 1: minute 35
        agg.record_success(5.0, 0.5, now=datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC))
        w1 = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))

        # Window 2: minute 36
        agg.record_success(8.0, 0.7, now=datetime(2026, 4, 7, 14, 36, 30, tzinfo=UTC))
        w2 = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 37, 1, tzinfo=UTC))

        assert len(w1) == 1
        assert len(w2) == 1
        assert w1[0].window_end == w2[0].window_start
        assert w1[0].window_end <= w2[0].window_start

    def test_flush_clears_state(self):
        agg = MetricsAggregator()
        ctx = _make_context()

        agg.record_success(5.0, 0.5, now=datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC))
        agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))

        # Second flush with no new requests should be empty
        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 37, 1, tzinfo=UTC))
        assert windows == []


# ---------------------------------------------------------------------------
# Percentile calculation
# ---------------------------------------------------------------------------


class TestPercentileCalculation:
    def test_single_value(self):
        assert _percentile([10.0], 50) == 10.0
        assert _percentile([10.0], 95) == 10.0
        assert _percentile([10.0], 99) == 10.0

    def test_empty_returns_zero(self):
        assert _percentile([], 50) == 0.0
        assert _percentile([], 99) == 0.0

    def test_deterministic_values(self):
        # 100 values: 1.0, 2.0, ..., 100.0
        values = [float(i) for i in range(1, 101)]
        assert _percentile(values, 50) == 50.0
        assert _percentile(values, 95) == 95.0
        assert _percentile(values, 99) == 99.0

    def test_p50_on_two_values(self):
        assert _percentile([1.0, 2.0], 50) == 1.0

    def test_percentile_from_aggregator(self):
        """Verify aggregator computes percentiles from recorded latencies."""
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 0, tzinfo=UTC)
        for i in range(1, 101):
            agg.record_success(float(i), 0.5, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert len(windows) == 1
        w = windows[0]
        assert w.latency_p50_ms == 50.0
        assert w.latency_p95_ms == 95.0
        assert w.latency_p99_ms == 99.0


# ---------------------------------------------------------------------------
# Success / failure / rejection counters
# ---------------------------------------------------------------------------


class TestCounterTracking:
    def test_success_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)
        agg.record_success(6.0, 0.6, now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.request_count == 2
        assert w.success_count == 2
        assert w.failure_count == 0
        assert w.rejected_count == 0
        assert w.prediction_count == 2

    def test_failure_internal_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(3.0, error_type="internal", now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.request_count == 1
        assert w.failure_count == 1
        assert w.internal_error_count == 1
        assert w.inference_runtime_error_count == 0
        assert w.feature_lookup_error_count == 0
        assert w.model_load_error_count == 0

    def test_failure_feature_lookup_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(3.0, error_type="feature_lookup", now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.failure_count == 1
        assert w.feature_lookup_error_count == 1
        assert w.internal_error_count == 0

    def test_failure_model_load_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(3.0, error_type="model_load", now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.failure_count == 1
        assert w.model_load_error_count == 1
        assert w.internal_error_count == 0

    def test_failure_inference_runtime_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(4.0, error_type="inference_runtime", now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.failure_count == 1
        assert w.inference_runtime_error_count == 1
        assert w.internal_error_count == 0

    def test_failure_default_is_internal(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(3.0, now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.internal_error_count == 1

    def test_mixed_failure_types(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_failure(3.0, error_type="feature_lookup", now=t)
        agg.record_failure(4.0, error_type="inference_runtime", now=t)
        agg.record_failure(5.0, error_type="internal", now=t)
        agg.record_failure(6.0, error_type="model_load", now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.request_count == 4
        assert w.failure_count == 4
        assert w.feature_lookup_error_count == 1
        assert w.inference_runtime_error_count == 1
        assert w.internal_error_count == 1
        assert w.model_load_error_count == 1
        assert w.success_count == 0

    def test_rejection_increments(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_rejection(2.0, now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.request_count == 1
        assert w.rejected_count == 1
        assert w.validation_error_count == 1
        assert w.success_count == 0
        assert w.failure_count == 0

    def test_mixed_outcomes(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)
        agg.record_failure(3.0, now=t)
        agg.record_rejection(2.0, now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.request_count == 3
        assert w.success_count == 1
        assert w.failure_count == 1
        assert w.rejected_count == 1

    def test_nan_prediction_counted(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, float("nan"), now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.prediction_null_count == 1
        assert w.prediction_non_finite_count == 1

    def test_inf_prediction_counted(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, float("inf"), now=t)

        windows = agg.flush_window(ctx, now=datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC))
        w = windows[0]
        assert w.prediction_non_finite_count == 1
        assert w.prediction_null_count == 0


# ---------------------------------------------------------------------------
# Emitted record matches schema
# ---------------------------------------------------------------------------


class TestSchemaCompliance:
    def test_flushed_window_is_valid_serving_metrics_window(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert len(windows) == 1
        w = windows[0]
        assert isinstance(w, ServingMetricsWindow)
        assert w.schema_version == "v1"
        assert w.endpoint_name == "/predict"

    def test_flushed_window_carries_lineage(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        w = windows[0]
        assert w.model_name == "mobility"
        assert w.model_version == "v1-lgbm-20260404"
        assert w.bundle_id == "/srv/models/mobility/v1"
        assert w.input_dataset_name == "rides-v3"
        assert w.input_dataset_version == "ds-2026-04-01"

    def test_flushed_window_serializes_to_valid_json(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        record = json.loads(windows[0].model_dump_json())

        assert record["schema_version"] == "v1"
        assert record["endpoint_name"] == "/predict"
        assert record["request_count"] == 1
        assert record["success_count"] == 1
        assert record["window_start"].endswith("+00:00") or record["window_start"].endswith("Z")

    def test_zero_traffic_fields_present(self):
        """Contract-required fields with no runtime trigger are present as zero."""
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        w = windows[0]
        assert w.timeout_count == 0
        assert w.feature_lookup_error_count == 0
        assert w.model_load_error_count == 0
        assert w.dependency_error_count == 0
        assert w.fallback_prediction_count == 0


# ---------------------------------------------------------------------------
# Serving hook integration (unit level)
# ---------------------------------------------------------------------------


class TestFlushCurrent:
    """Tests for the unconditional shutdown flush."""

    def test_flush_current_emits_without_crossing_minute_boundary(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)
        agg.record_success(8.0, 0.7, now=t)

        # Flush within the same minute — flush_window would return []
        same_minute = datetime(2026, 4, 7, 14, 35, 55, tzinfo=UTC)
        assert agg.flush_window(ctx, now=same_minute) == []

        # flush_current should still emit
        windows = agg.flush_current(ctx, now=same_minute)
        assert len(windows) == 1
        assert windows[0].request_count == 2
        assert windows[0].success_count == 2

    def test_flush_current_clears_bucket(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)

        agg.flush_current(ctx, now=t)
        # Second flush should be empty
        assert agg.flush_current(ctx, now=t) == []

    def test_flush_current_empty_aggregator(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        assert agg.flush_current(ctx) == []

    def test_flush_current_writes_valid_jsonl(self):
        agg = MetricsAggregator()
        ctx = _make_context()
        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)
        agg.record_failure(3.0, error_type="feature_lookup", now=t)
        agg.record_rejection(2.0, now=t)

        windows = agg.flush_current(ctx, now=t)
        assert len(windows) == 1
        record = json.loads(windows[0].model_dump_json())
        assert record["request_count"] == 3
        assert record["success_count"] == 1
        assert record["failure_count"] == 1
        assert record["rejected_count"] == 1


# ---------------------------------------------------------------------------
# Serving hook integration (unit level)
# ---------------------------------------------------------------------------


class TestServingHookIntegration:
    def test_flush_writes_jsonl_artifact(self):
        """Simulate the predict handler flush path end-to-end."""
        from app.observability.paths import metrics_window_path
        from app.observability.writer import append_jsonl

        agg = MetricsAggregator()
        ctx = _make_context()

        t = datetime(2026, 4, 7, 14, 35, 10, tzinfo=UTC)
        agg.record_success(5.0, 0.5, now=t)
        agg.record_success(8.0, 0.7, now=t)
        agg.record_failure(3.0, now=t)

        flush_time = datetime(2026, 4, 7, 14, 36, 1, tzinfo=UTC)
        windows = agg.flush_window(ctx, now=flush_time)
        assert len(windows) == 1

        with tempfile.TemporaryDirectory() as tmp:
            path = metrics_window_path(Path(tmp), windows[0].window_start)
            append_jsonl(path, windows[0])

            assert path.exists()
            record = json.loads(path.read_text().strip())
            assert record["request_count"] == 3
            assert record["success_count"] == 2
            assert record["failure_count"] == 1
            assert record["endpoint_name"] == "/predict"
