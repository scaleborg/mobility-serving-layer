from fastapi import APIRouter

from app.core.state import app_state
from app.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ok = app_state.model_loaded and app_state.feature_store_accessible
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_loaded=app_state.model_loaded,
        feature_store_accessible=app_state.feature_store_accessible,
    )
