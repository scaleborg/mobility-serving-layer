# Serving Observability Contract — P5

**Owner:** P5 Serving Layer
**Consumer:** P6 Observability / Analytics Layer
**Status:** Scaffolding (contract defined, runtime instrumentation pending)

---

## Overview

P5 emits append-only, immutable JSONL artifacts describing serving deployment
events and aggregated serving metrics windows. P6 ingests these artifacts
downstream. This document is the single source of truth for both schemas.

---

## Global Rules

1. **Append-only artifacts** — files are never rewritten or truncated; new
   records are appended.
2. **Immutable once written** — a record that has been flushed to disk must not
   be modified.
3. **UTC ISO-8601 timestamps** — all timestamp fields use UTC in ISO-8601
   format. Both `Z` and `+00:00` suffixes are valid (e.g.
   `2026-04-07T14:35:22Z` or `2026-04-07T14:35:22+00:00`).
4. **Backward-compatible evolution only** — new fields may be added as optional;
   existing required fields must not be removed or have their type changed.

---

## Schema 1 — `serving_deployment_event`

**schema_version:** `"v1"`

Records a single deployment activation: the moment P5 begins serving traffic
with a specific model bundle.

| Field | Type | Required | Constraints | Semantic Meaning |
|---|---|---|---|---|
| `event_type` | string | yes | fixed: `"serving_deployment_activated"` | Discriminator for event routing |
| `event_time` | timestamp | yes | UTC ISO-8601 | When the activation was recorded |
| `service_name` | string | yes | non-empty | Logical service identifier (e.g. `"mobility-serving-layer"`) |
| `service_version` | string | yes | non-empty | Deployed code version / image tag |
| `environment` | string | yes | non-empty | Deployment target (e.g. `"production"`, `"staging"`) |
| `deployment_id` | string | yes | non-empty | Unique identifier for this deployment |
| `instance_id` | string | no | — | Host / pod / instance identifier |
| `model_name` | string | yes | non-empty | Name of the served model |
| `model_version` | string | yes | non-empty | Version of the served model |
| `bundle_id` | string | yes | non-empty | Unique identifier for the inference bundle |
| `bundle_uri` | string | yes | non-empty | URI to the bundle artifact |
| `bundle_created_at` | timestamp | yes | UTC ISO-8601 | When the bundle was produced by P4 |
| `input_dataset_name` | string | yes | non-empty | Name of the training dataset (P4 lineage) |
| `input_dataset_version` | string | yes | non-empty | Version of the training dataset (P4 lineage) |
| `started_at` | timestamp | yes | UTC ISO-8601 | Training / bundle build start time (P4 lineage) |
| `completed_at` | timestamp | yes | UTC ISO-8601 | Training / bundle build completion time (P4 lineage) |
| `activation_reason` | string | yes | non-empty | Why this deployment was activated (e.g. `"startup"`, `"rollback"`) |
| `traffic_status` | string | yes | non-empty | Traffic routing state (e.g. `"serving"`, `"shadow"`, `"draining"`) |
| `schema_version` | string | yes | fixed: `"v1"` | Contract schema version |

---

## Schema 2 — `serving_metrics_window`

**schema_version:** `"v1"`

Aggregated serving metrics for a fixed time window, broken down by endpoint.

| Field | Type | Required | Constraints | Semantic Meaning |
|---|---|---|---|---|
| `schema_version` | string | yes | fixed: `"v1"` | Contract schema version |
| `window_start` | timestamp | yes | UTC ISO-8601 | Inclusive start of aggregation window |
| `window_end` | timestamp | yes | UTC ISO-8601; must be > `window_start` | Exclusive end of aggregation window |
| `service_name` | string | yes | non-empty | Logical service identifier |
| `service_version` | string | yes | non-empty | Deployed code version |
| `environment` | string | yes | non-empty | Deployment target |
| `deployment_id` | string | yes | non-empty | Deployment identifier |
| `model_name` | string | yes | non-empty | Served model name |
| `model_version` | string | yes | non-empty | Served model version |
| `bundle_id` | string | yes | non-empty | Inference bundle identifier |
| `input_dataset_name` | string | yes | non-empty | Training dataset name (P4 lineage) |
| `input_dataset_version` | string | yes | non-empty | Training dataset version (P4 lineage) |
| **Request metrics** | | | | |
| `endpoint_name` | string | yes | non-empty | API endpoint path |
| `request_count` | int64 | yes | >= 0 | Total requests received |
| `success_count` | int64 | yes | >= 0 | Requests that returned a successful prediction |
| `failure_count` | int64 | yes | >= 0 | Requests that failed |
| `rejected_count` | int64 | yes | >= 0 | Requests rejected before inference (e.g. validation) |
| `timeout_count` | int64 | yes | >= 0 | Requests that timed out |
| **Latency metrics** | | | | |
| `latency_p50_ms` | float64 | yes | >= 0.0 | 50th percentile end-to-end latency (ms) |
| `latency_p95_ms` | float64 | yes | >= 0.0 | 95th percentile end-to-end latency (ms) |
| `latency_p99_ms` | float64 | yes | >= 0.0 | 99th percentile end-to-end latency (ms) |
| `feature_lookup_p95_ms` | float64 | no | >= 0.0 | 95th percentile feature lookup latency (ms) |
| `model_exec_p95_ms` | float64 | no | >= 0.0 | 95th percentile model execution latency (ms) |
| **Error breakdown** | | | | |
| `validation_error_count` | int64 | yes | >= 0 | Input validation failures |
| `feature_lookup_error_count` | int64 | yes | >= 0 | Feature store lookup failures |
| `model_load_error_count` | int64 | yes | >= 0 | Model loading errors |
| `inference_runtime_error_count` | int64 | yes | >= 0 | Errors during model inference |
| `dependency_error_count` | int64 | yes | >= 0 | Downstream dependency failures |
| `internal_error_count` | int64 | yes | >= 0 | Uncategorized internal errors |
| **Input quality** | | | | |
| `input_schema_failure_count` | int64 | yes | >= 0 | Requests with schema-invalid input |
| `missing_required_field_count` | int64 | yes | >= 0 | Requests missing required fields |
| `invalid_type_count` | int64 | yes | >= 0 | Requests with type mismatches |
| `domain_violation_count` | int64 | yes | >= 0 | Requests violating domain constraints |
| **Prediction quality** | | | | |
| `prediction_count` | int64 | yes | >= 0 | Total predictions produced |
| `prediction_null_count` | int64 | yes | >= 0 | Null predictions |
| `prediction_non_finite_count` | int64 | yes | >= 0 | NaN / Inf predictions |
| `prediction_out_of_range_count` | int64 | yes | >= 0 | Predictions outside expected range |
| `fallback_prediction_count` | int64 | yes | >= 0 | Predictions using fallback logic |
| **Feature quality** | | | | |
| `missing_feature_count` | int64 | no | >= 0 | Features absent from store |
| `default_imputed_feature_count` | int64 | no | >= 0 | Features filled with defaults |
| `stale_feature_count` | int64 | no | >= 0 | Features older than staleness threshold |
| `feature_vector_build_failure_count` | int64 | no | >= 0 | Feature vector construction failures |
| **Heartbeat** | | | | |
| `heartbeat_emitted_at` | timestamp | yes | UTC ISO-8601 | When this metrics record was emitted |

---

## Artifact Layout

```
artifacts/
  serving/
    events/
      {YYYY-MM-DD}/
        deployment_events.jsonl
    metrics/
      {YYYY-MM-DD}/
        {HH}/
          metrics_{MM}.jsonl
```

- Date/hour/minute segments are derived from the event or window timestamp in UTC.
- One JSON object per line (JSONL).
- Files are append-only; never overwritten.
- **Window alignment:** exact 60-second minute-aligned windows are not enforced
  at the schema level. Enforcement of window duration and alignment is deferred
  to the future runtime aggregator in P5.

---

## Implementation Notes

- This is the **P5-side contract only**. P6 will ingest these artifacts in a
  later phase.
- **Deployment event emission is wired** — one `serving_deployment_event` is
  emitted on startup / bundle activation via the JSONL writer.
- **Metrics window emission remains future work** — the runtime aggregator for
  `serving_metrics_window` is not yet implemented.
- `BundleLineageContext` is sourced from real P4 model metadata where available
  (`model_version`, `trained_at`). Fields not yet emitted by P4
  (`input_dataset_name`, `input_dataset_version`) are required settings in
  non-development environments. In development only, missing lineage fields
  fall back to explicit dev placeholders.
- No Prometheus, Kafka, OpenTelemetry, or vendor monitoring is used at this
  layer.
- No drift monitoring or label/outcome tracking is in scope for this phase.
