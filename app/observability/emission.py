"""
Emission helpers for constructing observability records.

Pure/helper functions that build typed observability objects from bundle
context and runtime parameters. No I/O — callers are responsible for
persisting the resulting records via the writer module.
"""

from datetime import UTC, datetime

from app.schemas.observability import (
    BundleLineageContext,
    ServingDeploymentEvent,
    ServingMetricsWindow,
)


def build_deployment_event(
    context: BundleLineageContext,
    activation_reason: str,
    traffic_status: str,
    event_time: datetime | None = None,
) -> ServingDeploymentEvent:
    """Construct a ServingDeploymentEvent from bundle context + activation metadata."""
    return ServingDeploymentEvent(
        event_time=event_time or datetime.now(UTC),
        service_name=context.service_name,
        service_version=context.service_version,
        environment=context.environment,
        deployment_id=context.deployment_id,
        instance_id=context.instance_id,
        model_name=context.model_name,
        model_version=context.model_version,
        bundle_id=context.bundle_id,
        bundle_uri=context.bundle_uri,
        bundle_created_at=context.bundle_created_at,
        input_dataset_name=context.input_dataset_name,
        input_dataset_version=context.input_dataset_version,
        started_at=context.started_at,
        completed_at=context.completed_at,
        activation_reason=activation_reason,
        traffic_status=traffic_status,
    )


def build_metrics_window(
    context: BundleLineageContext,
    window_start: datetime,
    window_end: datetime,
    endpoint_name: str,
    request_count: int,
    success_count: int,
    failure_count: int,
    rejected_count: int,
    timeout_count: int,
    latency_p50_ms: float,
    latency_p95_ms: float,
    latency_p99_ms: float,
    validation_error_count: int,
    feature_lookup_error_count: int,
    model_load_error_count: int,
    inference_runtime_error_count: int,
    dependency_error_count: int,
    internal_error_count: int,
    input_schema_failure_count: int,
    missing_required_field_count: int,
    invalid_type_count: int,
    domain_violation_count: int,
    prediction_count: int,
    prediction_null_count: int,
    prediction_non_finite_count: int,
    prediction_out_of_range_count: int,
    fallback_prediction_count: int,
    heartbeat_emitted_at: datetime | None = None,
    feature_lookup_p95_ms: float | None = None,
    model_exec_p95_ms: float | None = None,
    missing_feature_count: int | None = None,
    default_imputed_feature_count: int | None = None,
    stale_feature_count: int | None = None,
    feature_vector_build_failure_count: int | None = None,
) -> ServingMetricsWindow:
    """Construct a ServingMetricsWindow from explicit parameters."""
    return ServingMetricsWindow(
        window_start=window_start,
        window_end=window_end,
        service_name=context.service_name,
        service_version=context.service_version,
        environment=context.environment,
        deployment_id=context.deployment_id,
        model_name=context.model_name,
        model_version=context.model_version,
        bundle_id=context.bundle_id,
        input_dataset_name=context.input_dataset_name,
        input_dataset_version=context.input_dataset_version,
        endpoint_name=endpoint_name,
        request_count=request_count,
        success_count=success_count,
        failure_count=failure_count,
        rejected_count=rejected_count,
        timeout_count=timeout_count,
        latency_p50_ms=latency_p50_ms,
        latency_p95_ms=latency_p95_ms,
        latency_p99_ms=latency_p99_ms,
        feature_lookup_p95_ms=feature_lookup_p95_ms,
        model_exec_p95_ms=model_exec_p95_ms,
        validation_error_count=validation_error_count,
        feature_lookup_error_count=feature_lookup_error_count,
        model_load_error_count=model_load_error_count,
        inference_runtime_error_count=inference_runtime_error_count,
        dependency_error_count=dependency_error_count,
        internal_error_count=internal_error_count,
        input_schema_failure_count=input_schema_failure_count,
        missing_required_field_count=missing_required_field_count,
        invalid_type_count=invalid_type_count,
        domain_violation_count=domain_violation_count,
        prediction_count=prediction_count,
        prediction_null_count=prediction_null_count,
        prediction_non_finite_count=prediction_non_finite_count,
        prediction_out_of_range_count=prediction_out_of_range_count,
        fallback_prediction_count=fallback_prediction_count,
        missing_feature_count=missing_feature_count,
        default_imputed_feature_count=default_imputed_feature_count,
        stale_feature_count=stale_feature_count,
        feature_vector_build_failure_count=feature_vector_build_failure_count,
        heartbeat_emitted_at=heartbeat_emitted_at or datetime.now(UTC),
    )
