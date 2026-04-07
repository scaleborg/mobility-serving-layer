"""
Inference service — orchestrates the prediction pipeline:
  1. Check model is loaded
  2. Load raw feature row for entity
  3. Resolve timestamp (request override or snapshot)
  4. Build and validate feature vector (parity check)
  5. Run prediction
  6. Return structured result dict
"""

import logging
import time
from typing import Any

from app.core.state import AppState
from app.features.loader import load_entity_row
from app.features.mapper import extract_timestamp
from app.features.validator import build_feature_vector
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

    If request_timestamp is provided it is preserved in the response;
    otherwise the snapshot timestamp is used.
    """
    if not state.model_loaded:
        raise ModelNotReadyError()

    t0 = time.perf_counter()

    raw_row = load_entity_row(state.db_conn, entity_id, state.entity_key)
    timestamp = request_timestamp if request_timestamp else extract_timestamp(raw_row)
    feature_vector = build_feature_vector(raw_row, state.feature_schema)
    prediction = run_prediction(state.model, feature_vector)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "Prediction for '%s': %.4f (%.2f ms)", entity_id, prediction, latency_ms
    )

    result: dict[str, Any] = {
        "entity_id": entity_id,
        "timestamp": timestamp,
        "prediction": prediction,
        "model_version": state.model_version,
        "latency_ms": round(latency_ms, 3),
    }
    return result
