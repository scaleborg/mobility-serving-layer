"""Tests for feature parity validation logic."""

import pytest

from app.features.validator import build_feature_vector, validate_schema
from app.schemas.errors import FeatureParityError
from app.schemas.feature_schema import FeatureEntry


SCHEMA = [
    FeatureEntry(name="feat_a", dtype="float64"),
    FeatureEntry(name="feat_b", dtype="float64"),
    FeatureEntry(name="feat_c", dtype="int64"),
]


def test_build_feature_vector_correct_order():
    row = {"feat_a": 1.0, "feat_b": 2.0, "feat_c": 3, "extra_col": "ignored"}
    vec = build_feature_vector(row, SCHEMA)
    assert vec.shape == (1, 3)
    assert float(vec[0, 0]) == 1.0
    assert float(vec[0, 1]) == 2.0
    assert int(vec[0, 2]) == 3


def test_build_feature_vector_coerces_int_to_float():
    row = {"feat_a": 1, "feat_b": 2, "feat_c": 3}
    vec = build_feature_vector(row, SCHEMA)
    assert isinstance(vec[0, 0], float)


def test_missing_feature_raises_parity_error():
    row = {"feat_a": 1.0, "feat_c": 3}
    with pytest.raises(FeatureParityError, match="missing from snapshot"):
        build_feature_vector(row, SCHEMA)


def test_unconvertible_dtype_raises_parity_error():
    row = {"feat_a": 1.0, "feat_b": 2.0, "feat_c": "not_a_number"}
    with pytest.raises(FeatureParityError, match="Cannot coerce"):
        build_feature_vector(row, SCHEMA)


def test_duplicate_feature_names_raise_parity_error():
    dup_schema = [
        FeatureEntry(name="feat_a", dtype="float64"),
        FeatureEntry(name="feat_a", dtype="float64"),
    ]
    with pytest.raises(FeatureParityError, match="Duplicate"):
        validate_schema(dup_schema)


def test_schema_envelope_validation():
    """Verify the FeatureSchema envelope model validates correctly."""
    from app.schemas.feature_schema import FeatureSchema

    raw = {
        "version": "v1",
        "entity_key": "station_id",
        "features": [{"name": "x", "dtype": "float64"}],
        "target": "y",
    }
    schema = FeatureSchema.model_validate(raw)
    assert schema.version == "v1"
    assert schema.entity_key == "station_id"
    assert len(schema.features) == 1
    assert schema.target == "y"


def test_schema_envelope_rejects_missing_version():
    from pydantic import ValidationError

    from app.schemas.feature_schema import FeatureSchema

    raw = {
        "entity_key": "station_id",
        "features": [{"name": "x", "dtype": "float64"}],
        "target": "y",
    }
    with pytest.raises(ValidationError):
        FeatureSchema.model_validate(raw)
