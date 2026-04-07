"""
Feature parity validator.

Validates that the raw feature row from the snapshot:
1. Contains every feature declared in feature_schema.json
2. Presents features in the declared order
3. Has dtype-compatible values (coerces where safe, raises otherwise)

No silent fallbacks — any mismatch raises FeatureParityError.
"""

import logging
from typing import Any

import numpy as np

from app.schemas.errors import FeatureParityError
from app.schemas.feature_schema import FeatureEntry

logger = logging.getLogger(__name__)

_DTYPE_MAP: dict[str, type] = {
    "float32": np.float32,
    "float64": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": bool,
    "object": object,
    "str": str,
}


def validate_schema(schema: list[FeatureEntry]) -> None:
    """
    Structural checks on the loaded schema (run once at startup).
    Raises FeatureParityError if names are not unique.
    """
    names = [e.name for e in schema]
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in names:
        if name in seen:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        raise FeatureParityError(
            message=f"Duplicate feature names in schema: {duplicates}"
        )


def build_feature_vector(
    raw_row: dict[str, Any],
    schema: list[FeatureEntry],
) -> np.ndarray:
    """
    Given a raw snapshot row and the ordered schema, return a 1-D numpy array
    of feature values in schema order with validated dtypes.

    Raises FeatureParityError on any mismatch.
    """
    missing = [e.name for e in schema if e.name not in raw_row]
    if missing:
        raise FeatureParityError(
            message=f"Features present in schema but missing from snapshot row: {missing}"
        )

    values: list[Any] = []
    for entry in schema:
        raw_value = raw_row[entry.name]
        np_dtype = _DTYPE_MAP.get(entry.dtype)

        if np_dtype is None:
            raise FeatureParityError(
                message=f"Unknown dtype '{entry.dtype}' for feature '{entry.name}'."
            )

        try:
            if np_dtype in (object, str):
                coerced = str(raw_value)
            else:
                coerced = np_dtype(raw_value)
        except (ValueError, TypeError) as exc:
            raise FeatureParityError(
                message=(
                    f"Cannot coerce feature '{entry.name}' value {raw_value!r} "
                    f"to dtype '{entry.dtype}': {exc}"
                )
            ) from exc

        values.append(coerced)

    return np.array(values, dtype=object).reshape(1, -1)
