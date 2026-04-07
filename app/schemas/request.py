from pydantic import BaseModel, field_validator


class PredictRequest(BaseModel):
    entity_id: str
    timestamp: str | None = None

    @field_validator("entity_id")
    @classmethod
    def entity_id_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("entity_id must not be blank.")
        return v.strip()
