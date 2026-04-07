"""
Inference service — orchestrates the prediction pipeline:
  1. Check model is loaded
  2. Resolve observation timestamp
  3. Reconstruct full feature vector from raw upstream data (point-in-time)
  4. Run prediction
  5. Return structured result dict

Feature reconstruction replaces the pre-materialized P3 snapshot lookup.
All 22 features are computed from raw_station_metrics_1min using the same
SQL as P2 training, guaranteeing train/serve parity.
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any

from app.core.state import AppState
from app.features.reconstructor import reconstruct_features
from app.model.predictor import run_prediction
from app.schemas.errors import ModelNotReadyError

logger = logging.getLogger(__name__)


def predict_for_entity(
    state: AppState,
    entity_id: str,
    request_timestamp: str | None = None,
) -> dict[str, Any]:
    """
    End-to-end prediction for a single entity.

    The request_timestamp is used as the observation point for feature
    reconstruction. If omitted, defaults to UTC now.
    """
    if not state.model_loaded:
        raise ModelNotReadyError()

    t0 = time.perf_counter()

    # Resolve observation timestamp
    if request_timestamp:
        obs_ts = datetime.fromisoformat(request_timestamp)
    else:
        obs_ts = datetime.now(timezone.utc).replace(tzinfo=None)

    # Reconstruct full feature vector from raw upstream data
    feature_vector, snapshot_source_ts = reconstruct_features(
        state.upstream_conn,
        entity_id,
        obs_ts,
        state.feature_schema,
    )

    prediction = run_prediction(state.model, feature_vector)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Prediction for '%s' at %s: %.4f (%.2f ms, source: %s)",
        entity_id,
        obs_ts.isoformat(),
        prediction,
        latency_ms,
        snapshot_source_ts,
    )

    result: dict[str, Any] = {
        "entity_id": entity_id,
        "timestamp": obs_ts.isoformat(),
        "prediction": prediction,
        "model_version": state.model_version,
        "latency_ms": round(latency_ms, 3),
    }
    return result
