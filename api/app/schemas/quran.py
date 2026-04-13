from pydantic import BaseModel, Field


class SurahMetadata(BaseModel):
    id: int = Field(ge=1, le=114)
    name: str = Field(min_length=1)
    transliteration: str = Field(min_length=1)
    total_verses: int = Field(ge=1)
