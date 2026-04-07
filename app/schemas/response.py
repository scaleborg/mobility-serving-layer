from pydantic import BaseModel


class PredictResponse(BaseModel):
    entity_id: str
    timestamp: str
    prediction: float
    model_version: str
    latency_ms: float
    features_used: list[str] | None = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    feature_store_accessible: bool


class MetadataResponse(BaseModel):
    model_version: str
    feature_schema_version: str
    expected_features: list[str]
    artifact_paths: dict[str, str]
