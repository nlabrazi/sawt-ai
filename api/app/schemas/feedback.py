# ROLE
# ----
# Schémas Pydantic pour recevoir un feedback utilisateur
# sur un résultat de reconnaissance.

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VerseMatchPayload(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sourate_id: int = Field(ge=1, le=114)
    sourate_name: str = Field(min_length=1, max_length=200)
    transliteration: str | None = Field(default=None, max_length=200)
    start_verse: int = Field(ge=1)
    end_verse: int = Field(ge=1)
    text: str = Field(min_length=1, max_length=5000)
    similarity: float = Field(ge=0)

    @field_validator("transliteration", mode="before")
    @classmethod
    def empty_transliteration_to_none(cls, value):
        if value is None:
            return None

        normalized_value = str(value).strip()
        return normalized_value or None

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_verse < self.start_verse:
            raise ValueError("end_verse doit être supérieur ou égal à start_verse.")

        return self


class VerseCorrectionPayload(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    sourate_id: int = Field(ge=1, le=114)
    sourate_name: str = Field(min_length=1, max_length=200)
    transliteration: str = Field(min_length=1, max_length=200)
    start_verse: int = Field(ge=1)
    end_verse: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_range(self):
        if self.end_verse < self.start_verse:
            raise ValueError("end_verse doit être supérieur ou égal à start_verse.")

        return self


class FeedbackPayload(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    is_correct: bool
    transcription_text: str = Field(min_length=1, max_length=5000)
    detected_verse: VerseMatchPayload | None = None
    correction: VerseCorrectionPayload | None = None
    comment: str | None = Field(default=None, max_length=1000)

    @field_validator("comment", mode="before")
    @classmethod
    def empty_comment_to_none(cls, value):
        if value is None:
            return None

        normalized_value = str(value).strip()
        return normalized_value or None
