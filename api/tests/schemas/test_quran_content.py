from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from app.schemas.quran import QuranTranslation
from app.schemas.tafsir import TafsirEntry, TafsirImportEntry


TRANSLATION = {
    "surah_id": 2,
    "ayah": 255,
    "text": "  Texte fictif de test.\nDeuxième ligne.  ",
    "source": "quranenc",
    "translator": "Traducteur de test",
    "version": "test-1",
    "source_url": "https://example.test/translation/2/255",
    "footnotes": "Note fictive de test.",
}
TAFSIR = {
    "surah_id": 2,
    "ayah": 255,
    "source": "ibn_kathir",
    "text_fr": "Commentaire fictif de test, sans contenu religieux.",
    "source_reference": "Édition fictive de test, page 1",
    "version": "test-1",
}
REVIEWED_AT = datetime(2026, 10, 7, 10, 0, tzinfo=timezone.utc)


def test_translation_preserves_provider_text_and_provenance():
    translation = QuranTranslation.model_validate(TRANSLATION)

    assert translation.model_dump(mode="json") == TRANSLATION


@pytest.mark.parametrize("source", ["ibn_kathir", "as_saadi"])
def test_tafsir_import_starts_unreviewed_for_each_supported_source(source):
    entry = TafsirImportEntry.model_validate({**TAFSIR, "source": source})

    assert (entry.surah_id, entry.ayah, entry.source) == (2, 255, source)
    assert entry.status == "need_review"
    assert entry.reviewed_at is None


@pytest.mark.parametrize("review_metadata", [
    {"status": "verified", "reviewed_at": REVIEWED_AT},
    {"reviewed_at": REVIEWED_AT},
])
def test_tafsir_import_rejects_review_metadata(review_metadata):
    with pytest.raises(ValidationError):
        TafsirImportEntry.model_validate({**TAFSIR, **review_metadata})


@pytest.mark.parametrize("review_metadata", [
    {"status": "verified"},
    {"status": "need_review", "reviewed_at": REVIEWED_AT},
    {"status": "verified", "reviewed_at": "2026-10-07T10:00:00"},
])
def test_stored_tafsir_rejects_inconsistent_or_undated_review(review_metadata):
    with pytest.raises(ValidationError):
        TafsirEntry.model_validate({**TAFSIR, **review_metadata})


def test_stored_verified_tafsir_retains_review_date_and_source():
    entry = TafsirEntry.model_validate({
        **TAFSIR,
        "status": "verified",
        "reviewed_at": REVIEWED_AT,
    })

    assert entry.status == "verified"
    assert entry.reviewed_at == REVIEWED_AT
    assert entry.source == "ibn_kathir"


@pytest.mark.parametrize("overrides", [
    {"source": "other_source"},
    {"source": ["ibn_kathir", "as_saadi"]},
    {"status": "published"},
])
def test_tafsir_rejects_unknown_or_combined_sources_and_unknown_status(overrides):
    with pytest.raises(ValidationError):
        TafsirEntry.model_validate({**TAFSIR, **overrides})


@pytest.mark.parametrize("model,payload,field", [
    (QuranTranslation, TRANSLATION, "text"),
    (QuranTranslation, TRANSLATION, "translator"),
    (QuranTranslation, TRANSLATION, "version"),
    (TafsirImportEntry, TAFSIR, "text_fr"),
    (TafsirImportEntry, TAFSIR, "source_reference"),
])
def test_content_rejects_blank_text_or_provenance(model, payload, field):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, field: " \n "})


@pytest.mark.parametrize("model,payload,overrides", [
    (QuranTranslation, TRANSLATION, {"surah_id": 115}),
    (TafsirImportEntry, TAFSIR, {"ayah": 0}),
    (TafsirImportEntry, TAFSIR, {"ayah": True}),
])
def test_content_rejects_invalid_verse_identifiers(model, payload, overrides):
    with pytest.raises(ValidationError):
        model.model_validate({**payload, **overrides})
