from fastapi import APIRouter

from app.core.config import settings
from app.core.state import app_state
from app.schemas.request import PredictRequest
from app.schemas.response import PredictResponse
from app.services.inference import predict_for_entity

router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    result = predict_for_entity(
        app_state,
        request.entity_id,
        request_timestamp=request.timestamp,
    )

    if settings.include_features_used:
        result["features_used"] = app_state.expected_features

    return PredictResponse(**result)
