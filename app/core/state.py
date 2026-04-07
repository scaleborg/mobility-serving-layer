from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import duckdb

if TYPE_CHECKING:
    from app.schemas.feature_schema import FeatureEntry


@dataclass
class AppState:
    """Singleton-like container holding all loaded runtime objects."""

    model: Any = None
    feature_schema: list[FeatureEntry] = field(default_factory=list)
    feature_schema_version: str = "unknown"
    entity_key: str = "station_id"
    target: str = "unknown"
    model_metadata: dict = field(default_factory=dict)
    db_conn: duckdb.DuckDBPyConnection | None = None

    @property
    def model_loaded(self) -> bool:
        return self.model is not None

    @property
    def feature_store_accessible(self) -> bool:
        return self.db_conn is not None

    @property
    def expected_features(self) -> list[str]:
        return [entry.name for entry in self.feature_schema]

    @property
    def model_version(self) -> str:
        return self.model_metadata.get("model_version", "unknown")


# Module-level singleton — populated during startup
app_state = AppState()
