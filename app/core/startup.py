"""
Startup sanity checks — run once before the app accepts traffic.

Checks performed (in order):
1. Required artifact files exist
2. feature_schema.json is structurally valid (envelope + entries)
3. Feature names are unique
4. model_metadata.json is loadable and has required keys
5. Model pickle is loadable
6. Parquet snapshot is accessible via DuckDB
"""

import json
import logging
from pathlib import Path

from app.core.config import Settings
from app.core.state import AppState
from app.features.loader import open_snapshot, probe_snapshot
from app.features.validator import validate_schema
from app.model.loader import load_model
from app.model.metadata import load_model_metadata
from app.observability.context import build_context_from_metadata
from app.observability.emission import build_deployment_event
from app.observability.paths import deployment_event_path
from app.observability.writer import append_jsonl
from app.schemas.errors import StartupError
from app.schemas.feature_schema import FeatureSchema

logger = logging.getLogger(__name__)


def _assert_file(path: Path, label: str) -> None:
    if not path.exists():
        raise StartupError(message=f"Required file not found [{label}]: {path}")


def run_startup(state: AppState, settings: Settings) -> None:
    """Populate *state* in-place. Raises StartupError on any failure."""

    # 1. File existence
    _assert_file(settings.model_path, "model")
    _assert_file(settings.feature_schema_path, "feature_schema")
    _assert_file(settings.model_metadata_path, "model_metadata")
    _assert_file(settings.feature_snapshot_path, "feature_snapshot")
    logger.info("Artifact file checks passed.")

    # 2 & 3. Parse envelope and structurally validate feature_schema.json
    try:
        raw = json.loads(settings.feature_schema_path.read_text())
        schema_envelope = FeatureSchema.model_validate(raw)
    except Exception as exc:
        raise StartupError(message=f"feature_schema.json is invalid: {exc}") from exc

    validate_schema(schema_envelope.features)
    state.feature_schema = schema_envelope.features
    state.feature_schema_version = schema_envelope.version
    state.entity_key = schema_envelope.entity_key
    state.target = schema_envelope.target
    logger.info(
        "Feature schema loaded: version=%s, entity_key=%s, %d features, target=%s",
        schema_envelope.version,
        schema_envelope.entity_key,
        len(schema_envelope.features),
        schema_envelope.target,
    )

    # 4. Model metadata
    try:
        state.model_metadata = load_model_metadata(settings.model_metadata_path)
    except Exception as exc:
        raise StartupError(message=f"model_metadata.json load failed: {exc}") from exc

    # 5. Model
    try:
        state.model = load_model(settings.model_path)
    except Exception as exc:
        raise StartupError(message=f"model.pkl load failed: {exc}") from exc

    # 6. Feature snapshot
    try:
        state.db_conn = open_snapshot(settings.feature_snapshot_path)
        probe_snapshot(state.db_conn)
    except Exception as exc:
        raise StartupError(message=f"Feature snapshot not accessible: {exc}") from exc

    # 7. Observability — emit deployment activation event
    event = None
    try:
        context = build_context_from_metadata(state, settings)
        event = build_deployment_event(
            context,
            activation_reason="startup",
            traffic_status="serving",
        )
    except Exception:
        logger.warning(
            "Failed to build observability context from bundle metadata — "
            "deployment event not emitted. Serving continues.",
            exc_info=True,
        )

    if event is not None:
        try:
            path = deployment_event_path(settings.artifact_base_dir, event.event_time)
            append_jsonl(path, event)
            logger.info(
                "Deployment event emitted: model=%s version=%s path=%s",
                event.model_name,
                event.model_version,
                path,
            )
        except Exception:
            logger.warning(
                "Failed to write deployment event artifact — "
                "serving continues without event emission.",
                exc_info=True,
            )

    logger.info("Startup complete — serving layer ready.")
