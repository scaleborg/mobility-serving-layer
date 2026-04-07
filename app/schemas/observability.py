"""
Typed schema models for P5 serving observability.

Covers:
- ServingDeploymentEvent (deployment activation records)
- ServingMetricsWindow (aggregated serving metrics per time window)
- BundleLineageContext (active bundle metadata for event construction)
"""

from datetime import UTC, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


def _require_utc(v: datetime, field_name: str) -> datetime:
    """Reject naive datetimes; normalize aware datetimes to UTC."""
    if v.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware (got naive datetime)")
    return v.astimezone(UTC)


def _non_empty(v: str, field_name: str) -> str:
    if not v or not v.strip():
        raise ValueError(f"{field_name} must not be empty")
    return v


class BundleLineageContext(BaseModel):
    """Active serving bundle metadata loaded into P5.

    Represents the lineage information from the P4 inference bundle
    plus P5 deployment context. Used to construct deployment events
    and tag metrics windows.
    """

    service_name: str
    service_version: str
    environment: str
    deployment_id: str
    instance_id: Optional[str] = None
    model_name: str
    model_version: str
    bundle_id: str
    bundle_uri: str
    bundle_created_at: datetime
    input_dataset_name: str
    input_dataset_version: str
    started_at: datetime
    completed_at: datetime

    @model_validator(mode="after")
    def _validate_non_empty_strings(self) -> "BundleLineageContext":
        for name in (
            "service_name",
            "service_version",
            "environment",
            "deployment_id",
            "model_name",
            "model_version",
            "bundle_id",
            "bundle_uri",
            "input_dataset_name",
            "input_dataset_version",
        ):
            _non_empty(getattr(self, name), name)
        return self

    @model_validator(mode="after")
    def _validate_utc_timestamps(self) -> "BundleLineageContext":
        for name in ("bundle_created_at", "started_at", "completed_at"):
            setattr(self, name, _require_utc(getattr(self, name), name))
        return self

    @model_validator(mode="after")
    def _validate_chronology(self) -> "BundleLineageContext":
        if self.completed_at < self.started_at:
            raise ValueError(
                f"completed_at ({self.completed_at}) must be >= started_at ({self.started_at})"
            )
        return self

    @classmethod
    def placeholder(
        cls,
        service_name: str = "mobility-serving-layer",
        service_version: str = "0.1.0",
        environment: str = "development",
    ) -> "BundleLineageContext":
        """Factory for local development / testing.

        Returns a context with synthetic values. Replace with real bundle
        metadata loading when wiring to production startup.
        """
        now = datetime.now(UTC)
        return cls(
            service_name=service_name,
            service_version=service_version,
            environment=environment,
            deployment_id="placeholder-deployment-id",
            model_name="placeholder-model",
            model_version="placeholder-v0",
            bundle_id="placeholder-bundle-id",
            bundle_uri="file:///placeholder/bundle",
            bundle_created_at=now,
            input_dataset_name="placeholder-dataset",
            input_dataset_version="placeholder-ds-v0",
            started_at=now,
            completed_at=now,
        )


class ServingDeploymentEvent(BaseModel):
    """A single deployment activation event emitted by P5."""

    event_type: Literal["serving_deployment_activated"] = "serving_deployment_activated"
    event_time: datetime
    service_name: str
    service_version: str
    environment: str
    deployment_id: str
    instance_id: Optional[str] = None
    model_name: str
    model_version: str
    bundle_id: str
    bundle_uri: str
    bundle_created_at: datetime
    input_dataset_name: str
    input_dataset_version: str
    started_at: datetime
    completed_at: datetime
    activation_reason: str
    traffic_status: str
    schema_version: Literal["v1"] = "v1"

    @model_validator(mode="after")
    def _validate_non_empty_strings(self) -> "ServingDeploymentEvent":
        for name in (
            "service_name",
            "service_version",
            "environment",
            "deployment_id",
            "model_name",
            "model_version",
            "bundle_id",
            "bundle_uri",
            "input_dataset_name",
            "input_dataset_version",
            "activation_reason",
            "traffic_status",
        ):
            _non_empty(getattr(self, name), name)
        return self

    @model_validator(mode="after")
    def _validate_utc_timestamps(self) -> "ServingDeploymentEvent":
        for name in ("event_time", "bundle_created_at", "started_at", "completed_at"):
            setattr(self, name, _require_utc(getattr(self, name), name))
        return self

    @model_validator(mode="after")
    def _validate_chronology(self) -> "ServingDeploymentEvent":
        if self.completed_at < self.started_at:
            raise ValueError(
                f"completed_at ({self.completed_at}) must be >= started_at ({self.started_at})"
            )
        return self


class ServingMetricsWindow(BaseModel):
    """Aggregated serving metrics for a fixed time window."""

    schema_version: Literal["v1"] = "v1"
    window_start: datetime
    window_end: datetime
    service_name: str
    service_version: str
    environment: str
    deployment_id: str
    model_name: str
    model_version: str
    bundle_id: str
    input_dataset_name: str
    input_dataset_version: str

    # Request metrics
    endpoint_name: str
    request_count: int = Field(ge=0)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    timeout_count: int = Field(ge=0)

    # Latency metrics
    latency_p50_ms: float = Field(ge=0.0)
    latency_p95_ms: float = Field(ge=0.0)
    latency_p99_ms: float = Field(ge=0.0)
    feature_lookup_p95_ms: Optional[float] = Field(default=None, ge=0.0)
    model_exec_p95_ms: Optional[float] = Field(default=None, ge=0.0)

    # Error breakdown
    validation_error_count: int = Field(ge=0)
    feature_lookup_error_count: int = Field(ge=0)
    model_load_error_count: int = Field(ge=0)
    inference_runtime_error_count: int = Field(ge=0)
    dependency_error_count: int = Field(ge=0)
    internal_error_count: int = Field(ge=0)

    # Input quality
    input_schema_failure_count: int = Field(ge=0)
    missing_required_field_count: int = Field(ge=0)
    invalid_type_count: int = Field(ge=0)
    domain_violation_count: int = Field(ge=0)

    # Prediction quality
    prediction_count: int = Field(ge=0)
    prediction_null_count: int = Field(ge=0)
    prediction_non_finite_count: int = Field(ge=0)
    prediction_out_of_range_count: int = Field(ge=0)
    fallback_prediction_count: int = Field(ge=0)

    # Feature quality
    missing_feature_count: Optional[int] = Field(default=None, ge=0)
    default_imputed_feature_count: Optional[int] = Field(default=None, ge=0)
    stale_feature_count: Optional[int] = Field(default=None, ge=0)
    feature_vector_build_failure_count: Optional[int] = Field(default=None, ge=0)

    # Heartbeat
    heartbeat_emitted_at: datetime

    @model_validator(mode="after")
    def _validate_utc_timestamps(self) -> "ServingMetricsWindow":
        for name in ("window_start", "window_end", "heartbeat_emitted_at"):
            setattr(self, name, _require_utc(getattr(self, name), name))
        return self

    @model_validator(mode="after")
    def _validate_window_order(self) -> "ServingMetricsWindow":
        if self.window_end <= self.window_start:
            raise ValueError(
                f"window_end ({self.window_end}) must be after window_start ({self.window_start})"
            )
        return self

    @model_validator(mode="after")
    def _validate_non_empty_strings(self) -> "ServingMetricsWindow":
        for name in (
            "service_name",
            "service_version",
            "environment",
            "deployment_id",
            "model_name",
            "model_version",
            "bundle_id",
            "input_dataset_name",
            "input_dataset_version",
            "endpoint_name",
        ):
            _non_empty(getattr(self, name), name)
        return self
