from fastapi import APIRouter

from app.core.config import settings
from app.core.state import app_state
from app.schemas.response import MetadataResponse

router = APIRouter()


@router.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    return MetadataResponse(
        model_version=app_state.model_version,
        feature_schema_version=app_state.feature_schema_version,
        expected_features=app_state.expected_features,
        artifact_paths={
            "model": str(settings.model_path),
            "feature_schema": str(settings.feature_schema_path),
            "model_metadata": str(settings.model_metadata_path),
            "feature_snapshot": str(settings.feature_snapshot_path),
        },
    )
