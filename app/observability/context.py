"""
Factory for building BundleLineageContext from real P4 bundle metadata.

Maps the actual loaded AppState + Settings into a typed lineage context.
Fields that P4 does not yet produce are sourced from required settings.
In development environments only, missing lineage fields fall back to
explicit dev placeholders.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import Settings
from app.core.state import AppState
from app.schemas.observability import BundleLineageContext

logger = logging.getLogger(__name__)

_DEV_ENVIRONMENTS = frozenset({"development", "local", "dev"})


def _parse_trained_at(metadata: dict) -> datetime:
    """Extract and parse trained_at from model metadata.

    Raises ValueError if missing or unparseable.
    """
    raw = metadata.get("trained_at")
    if not raw:
        raise ValueError("model_metadata is missing required field 'trained_at'")
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"trained_at must be timezone-aware, got: {raw}")
    return dt.astimezone(UTC)


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


def _resolve_lineage_field(
    value: str,
    field_name: str,
    environment: str,
) -> str:
    """Resolve a lineage field value.

    In development environments, missing values get an explicit dev placeholder.
    In non-development environments, missing values raise ValueError.
    """
    if value and value.strip():
        return value

    if environment in _DEV_ENVIRONMENTS:
        placeholder = f"dev-placeholder-{field_name}"
        logger.info(
            "Lineage field '%s' not configured — using dev placeholder: %s",
            field_name,
            placeholder,
        )
        return placeholder

    raise ValueError(
        f"Required lineage field '{field_name}' is not configured "
        f"for environment '{environment}'. Set it in Settings or .env."
    )


def build_context_from_metadata(
    state: AppState,
    settings: Settings,
) -> BundleLineageContext:
    """Build a BundleLineageContext from the active P4 inference bundle.

    Uses real metadata where available (model_version, trained_at) and
    settings-provided values for fields P4 does not yet emit.

    Raises ValueError if required metadata is missing or if required lineage
    fields are not configured in non-development environments.
    """
    if not state.model_metadata:
        raise ValueError("Cannot build lineage context: model_metadata is empty")

    trained_at = _parse_trained_at(state.model_metadata)
    model_version = state.model_metadata.get("model_version")
    if not model_version:
        raise ValueError("Cannot build lineage context: model_version is missing")

    input_dataset_name = _resolve_lineage_field(
        settings.input_dataset_name, "input_dataset_name", settings.environment,
    )
    input_dataset_version = _resolve_lineage_field(
        settings.input_dataset_version, "input_dataset_version", settings.environment,
    )

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
        started_at=trained_at,
        completed_at=trained_at,
    )
