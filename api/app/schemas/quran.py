from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class SurahMetadata(BaseModel):
    id: int = Field(ge=1, le=114)
    name: str = Field(min_length=1)
    transliteration: str = Field(min_length=1)
    total_verses: int = Field(ge=1)


class QuranTranslation(BaseModel):
    """One French translation, preserved as supplied by its provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    surah_id: int = Field(ge=1, le=114, strict=True)
    ayah: int = Field(ge=1, le=286, strict=True)
    text: str = Field(min_length=1)
    source: str = Field(min_length=1)
    translator: str = Field(min_length=1)
    version: str = Field(min_length=1)
    source_url: HttpUrl
    footnotes: str = ""

    @field_validator("text", "source", "translator", "version")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Translation text and source metadata must not be blank.")
        return value
