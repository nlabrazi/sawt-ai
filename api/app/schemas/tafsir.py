"""Internal tafsir records. Public filtering belongs to the read service."""

from typing import Literal

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

TafsirSource = Literal["ibn_kathir", "as_saadi"]
TafsirStatus = Literal["need_review", "verified"]


class TafsirEntry(BaseModel):
    """One source's French commentary for one ayah, with review metadata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surah_id: int = Field(ge=1, le=114, strict=True)
    ayah: int = Field(ge=1, le=286, strict=True)
    source: TafsirSource
    text_fr: str = Field(min_length=1)
    source_reference: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: TafsirStatus = "need_review"
    reviewed_at: AwareDatetime | None = None

    @field_validator("text_fr", "source_reference", "version")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Tafsir text and source metadata must not be blank.")
        return value

    @model_validator(mode="after")
    def validate_review_metadata(self):
        if self.status == "verified" and self.reviewed_at is None:
            raise ValueError("A verified tafsir requires reviewed_at.")
        if self.status == "need_review" and self.reviewed_at is not None:
            raise ValueError("A tafsir needing review cannot have reviewed_at.")
        return self


class TafsirImportEntry(TafsirEntry):
    """Imports and generated drafts cannot declare themselves verified."""

    status: Literal["need_review"] = "need_review"
    reviewed_at: None = None
