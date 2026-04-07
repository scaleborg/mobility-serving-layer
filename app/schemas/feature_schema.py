"""
Mirrors the feature_schema.json contract produced by P4.

Expected file format:
{
  "version": "v1",
  "entity_key": "station_id",
  "features": [
    {"name": "feature_a", "dtype": "float64"},
    {"name": "feature_b", "dtype": "int32"}
  ],
  "target": "demand_next_hour"
}
"""

from pydantic import BaseModel, field_validator


ALLOWED_DTYPES = frozenset(
    ["float32", "float64", "int32", "int64", "bool", "object", "str"]
)


class FeatureEntry(BaseModel):
    name: str
    dtype: str

    @field_validator("dtype")
    @classmethod
    def dtype_must_be_known(cls, v: str) -> str:
        if v not in ALLOWED_DTYPES:
            raise ValueError(f"Unknown dtype '{v}'. Allowed: {sorted(ALLOWED_DTYPES)}")
        return v

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Feature name must not be blank.")
        return v


class FeatureSchema(BaseModel):
    """Top-level envelope for feature_schema.json."""
    version: str
    entity_key: str
    features: list[FeatureEntry]
    target: str
