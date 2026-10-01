from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints


class HadithSearchRequest(BaseModel):
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=3, max_length=300)]
    limit: int = Field(default=3, ge=1, le=5, strict=True)


class HadithResult(BaseModel):
    id: str
    title: str
    arabic: str
    translation: str
    explanation: str | None = None
    grade: str | None = None
    attribution: str | None = None
    source_url: str
    provider: Literal["HadeethEnc"] = "HadeethEnc"


class HadithSearchResponse(BaseModel):
    query: str
    results: list[HadithResult]
    search_mode: Literal["keywords", "semantic"] = "semantic"
    search_terms: list[str] = Field(default_factory=list)
