from copy import deepcopy

import pytest

from app.services.tafsir_sources import (
    TafsirSourceError,
    get_tafsir_source,
    resolve_quran_foundation_resource,
)

# Provider-shaped metadata fixtures; no religious commentary is included.
RESOURCES = [
    {
        "id": 14,
        "name": "Tafsir Ibn Kathir",
        "author_name": "Hafiz Ibn Kathir",
        "slug": "ar-tafsir-ibn-kathir",
        "language_name": "arabic",
        "translated_name": {"name": "Libellé de test français", "language_name": "french"},
    },
    {
        "id": 91,
        "name": "Al-Sa'di",
        "author_name": "Saddi",
        "slug": "ar-tafseer-al-saddi",
        "language_name": "arabic",
    },
    {
        "id": 169,
        "name": "Ibn Kathir (Abridged)",
        "slug": "en-tafisr-ibn-kathir",
        "language_name": "english",
    },
]


def test_sources_resolve_to_separate_original_resources_without_changing_metadata():
    payload = {"tafsirs": deepcopy(RESOURCES)}
    original = deepcopy(payload)

    ibn_kathir = resolve_quran_foundation_resource("ibn_kathir", payload)
    as_saadi = resolve_quran_foundation_resource("as_saadi", payload)

    assert ibn_kathir["id"] == 14
    assert as_saadi["id"] == 91
    assert ibn_kathir["language_name"] == "arabic"
    assert get_tafsir_source("ibn_kathir").source_url.endswith("/ar-tafsir-ibn-kathir")
    assert get_tafsir_source("as_saadi").source_url.endswith("/ar-tafseer-al-saddi")
    assert payload == original


def test_english_abridgement_is_not_a_fallback_for_missing_original():
    with pytest.raises(TafsirSourceError, match="absente"):
        resolve_quran_foundation_resource("ibn_kathir", {"tafsirs": RESOURCES[1:]})


@pytest.mark.parametrize("mismatch", [
    {"slug": "ar-tafseer-al-saddi"},
    {"language_name": "french"},
    {"id": "14"},
    {"id": 14.0},
])
def test_resource_identity_or_original_language_mismatch_is_rejected(mismatch):
    resource = {**RESOURCES[0], **mismatch}

    with pytest.raises(TafsirSourceError):
        resolve_quran_foundation_resource("ibn_kathir", {"tafsirs": [resource]})


@pytest.mark.parametrize("payload", [None, {}, {"tafsirs": {}}, {"tafsirs": []}])
def test_missing_or_invalid_catalog_is_rejected(payload):
    with pytest.raises(TafsirSourceError):
        resolve_quran_foundation_resource("ibn_kathir", payload)


def test_duplicate_source_cannot_be_selected_silently():
    with pytest.raises(TafsirSourceError, match="doublon"):
        resolve_quran_foundation_resource("as_saadi", {"tafsirs": [RESOURCES[1], RESOURCES[1]]})


def test_unsupported_source_is_rejected():
    with pytest.raises(TafsirSourceError, match="non prise en charge"):
        get_tafsir_source("al_tabari")
