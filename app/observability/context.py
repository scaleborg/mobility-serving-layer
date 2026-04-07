"""
Factory for building BundleLineageContext from real P4 model metadata.

Maps the actual loaded AppState + Settings into a typed lineage context.
Dataset lineage fields (input_dataset_name, input_dataset_version,
started_at, completed_at) are read from model_metadata.json, which P4
is expected to populate with dataset lineage.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.state import AppState
from app.schemas.observability import BundleLineageContext

logger = logging.getLogger(__name__)


def _parse_timestamp(metadata: dict, field: str) -> datetime:
    """Extract and parse a required UTC timestamp from model metadata.

    Raises ValueError if missing, unparseable, or naive.
    """
    raw = metadata.get(field)
    if not raw:
        raise ValueError(f"model_metadata is missing required field '{field}'")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware, got: {raw}")
    return dt.astimezone(UTC)


def _require_str(metadata: dict, field: str) -> str:
    """Extract a required non-empty string from model metadata."""
    value = metadata.get(field)
    if not value or not str(value).strip():
        raise ValueError(
            f"Cannot build lineage context: '{field}' is missing or empty "
            f"in model metadata"
        )
    return str(value).strip()


def _parse_trained_at(metadata: dict) -> datetime:
    """Extract and parse trained_at from model metadata.

    Raises ValueError if missing or unparseable.
    """
    return _parse_timestamp(metadata, "trained_at")


def _derive_model_name(model_path: Path) -> str:
    """Derive model name from the artifact directory structure.

    Expected layout: models/<model_name>/<version>/model.pkl
    Falls back to the parent directory name.
    """
    # model_path points to model.pkl; parent is version dir, grandparent is model name
    return model_path.resolve().parent.parent.name


def _derive_bundle_uri(model_path: Path) -> str:
    """Build a file:// URI for the bundle directory."""
    return model_path.resolve().parent.as_uri()


def _derive_bundle_id(model_path: Path) -> str:
    """Derive a deterministic bundle identifier from the artifact path."""
    return str(model_path.resolve().parent)


def build_context_from_metadata(
    state: AppState,
    settings: Settings,
) -> BundleLineageContext:
    """Build a BundleLineageContext from the active P4 inference bundle.

    All lineage fields (model_version, trained_at, input_dataset_name,
    input_dataset_version, started_at, completed_at) are sourced from
    model_metadata.json loaded at startup.

    Raises ValueError if any required metadata field is missing.
    """
    if not state.model_metadata:
        raise ValueError("Cannot build lineage context: model_metadata is empty")

    trained_at = _parse_trained_at(state.model_metadata)
    model_version = state.model_metadata.get("model_version")
    if not model_version:
        raise ValueError("Cannot build lineage context: model_version is missing")

    input_dataset_name = _require_str(state.model_metadata, "input_dataset_name")
    input_dataset_version = _require_str(state.model_metadata, "input_dataset_version")
    started_at = _parse_timestamp(state.model_metadata, "started_at")
    completed_at = _parse_timestamp(state.model_metadata, "completed_at")

    return BundleLineageContext(
        service_name=settings.service_name,
        service_version=settings.service_version,
        environment=settings.environment,
        deployment_id=settings.deployment_id,
        instance_id=settings.instance_id,
        model_name=_derive_model_name(settings.model_path),
        model_version=model_version,
        bundle_id=_derive_bundle_id(settings.model_path),
        bundle_uri=_derive_bundle_uri(settings.model_path),
        bundle_created_at=trained_at,
        input_dataset_name=input_dataset_name,
        input_dataset_version=input_dataset_version,
        started_at=started_at,
        completed_at=completed_at,
    )
