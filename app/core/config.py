from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SERVING_", env_file=".env", extra="ignore")

    # Versioned artifact paths (P4 outputs under models/<name>/<version>/)
    model_path: Path = Path("models/mobility/v1/model.pkl")
    feature_schema_path: Path = Path("models/mobility/v1/feature_schema.json")
    model_metadata_path: Path = Path("models/mobility/v1/model_metadata.json")

    # Feature store (P3 output)
    feature_snapshot_path: Path = Path("data/online_features.parquet")

    # Behaviour
    include_features_used: bool = False
    log_level: str = "INFO"

    # Service identity (observability)
    service_name: str = "mobility-serving-layer"
    service_version: str = "0.1.0"
    environment: str = "development"
    deployment_id: str = "local"
    instance_id: str | None = None

    # Observability artifact output
    artifact_base_dir: Path = Path(".")


settings = Settings()
