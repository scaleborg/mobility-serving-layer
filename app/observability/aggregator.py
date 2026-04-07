"""
In-memory 60-second metrics window aggregator.

Groups serving request metrics into fixed UTC minute-aligned windows.
Thread-safe — FastAPI sync handlers run in a threadpool.

Callers record outcomes via record_success / record_failure / record_rejection,
then call flush_window() to collect any completed windows. The aggregator does
not perform I/O; flushed windows are returned as ServingMetricsWindow models
for the caller to persist.
"""

import math
import threading
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from app.observability.emission import build_metrics_window

if TYPE_CHECKING:
    from app.schemas.observability import BundleLineageContext, ServingMetricsWindow

_WINDOW_SECONDS = 60


def _floor_minute(dt: datetime) -> datetime:
    """Truncate a UTC datetime to the start of its containing minute."""
    return dt.replace(second=0, microsecond=0)


def _percentile(sorted_values: list[float], p: float) -> float:
    """Nearest-rank percentile on a pre-sorted list.

    Returns 0.0 for empty lists.
    """
    if not sorted_values:
        return 0.0
    k = max(0, min(len(sorted_values) - 1, int(math.ceil(p / 100.0 * len(sorted_values))) - 1))
    return sorted_values[k]


class _WindowBucket:
    """Accumulator for a single 60-second window."""

    __slots__ = (
        "window_start",
        "window_end",
        "request_count",
        "success_count",
        "failure_count",
        "rejected_count",
        "timeout_count",
        "latencies",
        "validation_error_count",
        "feature_lookup_error_count",
        "model_load_error_count",
        "inference_runtime_error_count",
        "dependency_error_count",
        "internal_error_count",
        "input_schema_failure_count",
        "missing_required_field_count",
        "invalid_type_count",
        "domain_violation_count",
        "prediction_count",
        "prediction_null_count",
        "prediction_non_finite_count",
        "prediction_out_of_range_count",
        "fallback_prediction_count",
    )

    def __init__(self, window_start: datetime) -> None:
        self.window_start = window_start
        self.window_end = window_start + timedelta(seconds=_WINDOW_SECONDS)
        self.request_count = 0
        self.success_count = 0
        self.failure_count = 0
        self.rejected_count = 0
        self.timeout_count = 0
        self.latencies: list[float] = []
        self.validation_error_count = 0
        self.feature_lookup_error_count = 0
        self.model_load_error_count = 0
        self.inference_runtime_error_count = 0
        self.dependency_error_count = 0
        self.internal_error_count = 0
        self.input_schema_failure_count = 0
        self.missing_required_field_count = 0
        self.invalid_type_count = 0
        self.domain_violation_count = 0
        self.prediction_count = 0
        self.prediction_null_count = 0
        self.prediction_non_finite_count = 0
        self.prediction_out_of_range_count = 0
        self.fallback_prediction_count = 0


class MetricsAggregator:
    """Minute-aligned serving metrics aggregator.

    Thread-safe. Accumulates request-level observations into 60-second
    windows and produces ServingMetricsWindow records on flush.
    """

    def __init__(self, endpoint_name: str = "/predict") -> None:
        self._endpoint_name = endpoint_name
        self._lock = threading.Lock()
        self._bucket: _WindowBucket | None = None

    def _get_bucket(self, now: datetime) -> _WindowBucket:
        """Return the current bucket, rotating if the window has passed."""
        ws = _floor_minute(now)
        if self._bucket is None or ws >= self._bucket.window_end:
            self._bucket = _WindowBucket(ws)
        return self._bucket

    def record_success(
        self,
        latency_ms: float,
        prediction: float,
        now: datetime | None = None,
    ) -> None:
        """Record a successful prediction request."""
        now = now or datetime.now(UTC)
        with self._lock:
            b = self._get_bucket(now)
            b.request_count += 1
            b.success_count += 1
            b.latencies.append(latency_ms)
            b.prediction_count += 1
            if prediction is None or (isinstance(prediction, float) and math.isnan(prediction)):
                b.prediction_null_count += 1
            if isinstance(prediction, float) and not math.isfinite(prediction):
                b.prediction_non_finite_count += 1

    def record_failure(
        self,
        latency_ms: float,
        error_type: str = "internal",
        now: datetime | None = None,
    ) -> None:
        """Record a failed request.

        error_type selects the specific error counter:
          "feature_lookup"    → feature_lookup_error_count
          "model_load"        → model_load_error_count
          "inference_runtime" → inference_runtime_error_count
          "internal"          → internal_error_count (default)
        """
        now = now or datetime.now(UTC)
        with self._lock:
            b = self._get_bucket(now)
            b.request_count += 1
            b.failure_count += 1
            b.latencies.append(latency_ms)
            if error_type == "feature_lookup":
                b.feature_lookup_error_count += 1
            elif error_type == "model_load":
                b.model_load_error_count += 1
            elif error_type == "inference_runtime":
                b.inference_runtime_error_count += 1
            else:
                b.internal_error_count += 1

    def record_rejection(
        self,
        latency_ms: float,
        now: datetime | None = None,
    ) -> None:
        """Record a rejected request (validation / parity failure)."""
        now = now or datetime.now(UTC)
        with self._lock:
            b = self._get_bucket(now)
            b.request_count += 1
            b.rejected_count += 1
            b.latencies.append(latency_ms)
            b.validation_error_count += 1

    def flush_window(
        self,
        context: "BundleLineageContext",
        now: datetime | None = None,
    ) -> list["ServingMetricsWindow"]:
        """Flush any completed window.

        Returns a list of 0 or 1 ServingMetricsWindow records. The window
        is considered complete when *now* has crossed into the next minute.
        An empty or zero-request window is not emitted.
        """
        now = now or datetime.now(UTC)
        with self._lock:
            if self._bucket is None:
                return []
            if now < self._bucket.window_end:
                return []
            if self._bucket.request_count == 0:
                self._bucket = None
                return []

            b = self._bucket
            self._bucket = None

        sorted_lat = sorted(b.latencies)
        window = build_metrics_window(
            context=context,
            window_start=b.window_start,
            window_end=b.window_end,
            endpoint_name=self._endpoint_name,
            request_count=b.request_count,
            success_count=b.success_count,
            failure_count=b.failure_count,
            rejected_count=b.rejected_count,
            timeout_count=b.timeout_count,
            latency_p50_ms=_percentile(sorted_lat, 50),
            latency_p95_ms=_percentile(sorted_lat, 95),
            latency_p99_ms=_percentile(sorted_lat, 99),
            validation_error_count=b.validation_error_count,
            feature_lookup_error_count=b.feature_lookup_error_count,
            model_load_error_count=b.model_load_error_count,
            inference_runtime_error_count=b.inference_runtime_error_count,
            dependency_error_count=b.dependency_error_count,
            internal_error_count=b.internal_error_count,
            input_schema_failure_count=b.input_schema_failure_count,
            missing_required_field_count=b.missing_required_field_count,
            invalid_type_count=b.invalid_type_count,
            domain_violation_count=b.domain_violation_count,
            prediction_count=b.prediction_count,
            prediction_null_count=b.prediction_null_count,
            prediction_non_finite_count=b.prediction_non_finite_count,
            prediction_out_of_range_count=b.prediction_out_of_range_count,
            fallback_prediction_count=b.fallback_prediction_count,
            heartbeat_emitted_at=now,
        )
        return [window]
