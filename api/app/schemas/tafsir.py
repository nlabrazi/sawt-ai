"""Internal tafsir records. Public filtering belongs to the read service."""

from typing import Any, Literal

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


class VerifiedTafsirEntry(TafsirEntry):
    """Public response contract: an unreviewed entry cannot be serialized here."""

    status: Literal["verified"]
    reviewed_at: AwareDatetime


class TafsirDraftImportEntry(TafsirImportEntry):
    """A French draft with the complete original passage used to prepare it."""

    source_text: str = Field(min_length=1)
    source_surah_id: int = Field(ge=1, le=114, strict=True)
    source_start_ayah: int = Field(ge=1, le=286, strict=True)
    source_end_ayah: int = Field(ge=1, le=286, strict=True)

    @field_validator("source_text")
    @classmethod
    def reject_blank_source_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Le passage source ne doit pas être vide.")
        return value

    @model_validator(mode="after")
    def validate_source_passage(self):
        if (
            self.source_surah_id != self.surah_id
            or not self.source_start_ayah <= self.ayah <= self.source_end_ayah
        ):
            raise ValueError("Le passage source doit couvrir le verset du brouillon.")
        return self


class TafsirFrenchImportBatch(BaseModel):
    """One edition of one source, limited to the small pilot."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: int = Field(default=1, ge=1, le=1, strict=True)
    source: TafsirSource
    source_language: Literal["ar", "fr"]
    source_edition: str = Field(min_length=1)
    version: str = Field(min_length=1)
    reuse_reference: str = Field(min_length=1)
    entries: tuple[TafsirDraftImportEntry, ...] = Field(min_length=1, max_length=13)
    imported_at: AwareDatetime | None = None

    @field_validator("source_edition", "version", "reuse_reference")
    @classmethod
    def reject_blank_provenance(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("L'édition, la version et la référence de réutilisation sont requises.")
        return value

    @model_validator(mode="after")
    def validate_entries_provenance(self):
        passages = {}
        for entry in self.entries:
            if entry.source != self.source or entry.version != self.version:
                raise ValueError("Un lot doit contenir une seule source et une seule version.")
            if self.source_language == "fr" and entry.text_fr != entry.source_text:
                raise ValueError("Un import français doit préserver le texte source à l'identique.")
            key = (entry.source_surah_id, entry.source_start_ayah, entry.source_end_ayah)
            passage = (entry.source_reference, entry.source_text)
            if key in passages and passages[key] != passage:
                raise ValueError("Un même passage source ne peut pas avoir deux contenus ou références.")
            passages[key] = passage
        return self


class TafsirReviewEntry(TafsirEntry):
    """Internal stored row, including the original passage and review revision."""

    provenance: dict[str, Any]
    updated_at: AwareDatetime
