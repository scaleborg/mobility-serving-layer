import logging
import time

from fastapi import APIRouter

from app.core.config import settings
from app.core.state import app_state
from app.observability.aggregator import MetricsAggregator
from app.observability.paths import metrics_window_path
from app.observability.writer import append_jsonl
from app.schemas.errors import (
    EntityNotFoundError,
    FeatureParityError,
    ModelNotReadyError,
)
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.inference import predict_for_entity

logger = logging.getLogger(__name__)

router = APIRouter()

aggregator = MetricsAggregator(endpoint_name="/predict")


def _flush_and_write() -> None:
    """Flush any completed metrics window and write to JSONL."""
    ctx = app_state.lineage_context
    if ctx is None:
        return
    for window in aggregator.flush_window(ctx):
        try:
            path = metrics_window_path(settings.artifact_base_dir, window.window_start)
            append_jsonl(path, window)
            logger.debug("Metrics window flushed: %s", path)
        except Exception:
            logger.warning("Failed to write metrics window.", exc_info=True)


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    # Flush completed windows before recording, so _get_bucket rotation
    # does not silently discard a filled window.
    _flush_and_write()

    t0 = time.perf_counter()
    try:
        result = predict_for_entity(
            app_state,
            request.entity_id,
            request_timestamp=request.timestamp,
        )
    except FeatureParityError:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        aggregator.record_rejection(latency_ms)
        raise
    except EntityNotFoundError:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        aggregator.record_failure(latency_ms, error_type="feature_lookup")
        raise
    except ModelNotReadyError:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        aggregator.record_failure(latency_ms, error_type="model_load")
        raise
    except Exception:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        aggregator.record_failure(latency_ms, error_type="internal")
        raise

    latency_ms = (time.perf_counter() - t0) * 1000.0
    prediction = result["prediction"]
    aggregator.record_success(latency_ms, prediction)

    if settings.include_features_used:
        result["features_used"] = app_state.expected_features

    return PredictResponse(**result)
