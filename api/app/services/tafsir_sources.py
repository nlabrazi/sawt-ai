"""Identify the two original tafsirs without downloading commentary."""

from dataclasses import dataclass
from typing import Any

from app.schemas.tafsir import TafsirSource

PROVIDER = "quran_foundation"
TERMS_URL = "https://api-docs.quran.com/legal/developer-terms/"
CATALOG_DOCUMENTATION_URL = (
    "https://api-docs.quran.com/docs/content_apis_versioned/4.0.0/tafsirs/"
)


@dataclass(frozen=True)
class TafsirSourceDefinition:
    source: TafsirSource
    display_name: str
    resource_id: int
    slug: str
    language_name: str = "arabic"

    @property
    def source_url(self) -> str:
        return f"https://quran.com/al-fatihah/1/tafsirs/{self.slug}"


TAFSIR_SOURCES = (
    TafsirSourceDefinition("ibn_kathir", "Ibn Kathir", 14, "ar-tafsir-ibn-kathir"),
    TafsirSourceDefinition("as_saadi", "As-Sa‘di", 91, "ar-tafseer-al-saddi"),
)


class TafsirSourceError(Exception):
    pass


def get_tafsir_source(source: TafsirSource) -> TafsirSourceDefinition:
    for definition in TAFSIR_SOURCES:
        if definition.source == source:
            return definition
    raise TafsirSourceError("Source tafsir non prise en charge.")


def resolve_quran_foundation_resource(source: TafsirSource, payload: Any) -> dict[str, Any]:
    """Check catalog identity and original language before a future import.

    The catalog's translated_name describes a label, not the commentary's
    language. Resource 169 is an English abridgement, not our Ibn Kathir source.
    """
    definition = get_tafsir_source(source)
    resources = payload.get("tafsirs") if isinstance(payload, dict) else None
    if not isinstance(resources, list):
        raise TafsirSourceError("Catalogue des tafsirs invalide.")
    matches = [
        resource for resource in resources
        if isinstance(resource, dict)
        and type(resource.get("id")) is int
        and resource["id"] == definition.resource_id
    ]
    if len(matches) != 1:
        raise TafsirSourceError(f"Source {source} absente ou en doublon dans le catalogue.")
    resource = matches[0]
    if (
        resource.get("slug") != definition.slug
        or resource.get("language_name") != definition.language_name
    ):
        raise TafsirSourceError(f"Ouvrage ou langue inattendus pour {source}.")
    return resource
