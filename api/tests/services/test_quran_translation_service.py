import copy
import json
from pathlib import Path

import pytest

import app.services.quran_translation_service as service
from scripts import import_quran_translation as importer

API_DIR = Path(__file__).resolve().parents[2]


def provider_metadata():
    return {"translations": [{
        "key": "french_rashid",
        "language_iso_code": "fr",
        "version": "test-1",
        "last_update": 123,
        "title": "Traduction fictive de test",
        "description": "Métadonnées fictives, sans contenu religieux.",
    }]}


def provider_verses():
    return {
        (surah_id, ayah): {"result": {
            "sura": str(surah_id),
            "aya": str(ayah),
            "translation": f"  Texte fictif {surah_id}:{ayah}.\n  ",
            "footnotes": f"Note fictive {surah_id}:{ayah}.",
        }}
        for surah_id, ayah in importer.PILOT_REFERENCES
    }


def snapshot():
    return importer.build_snapshot(provider_metadata(), provider_verses())


@pytest.fixture(autouse=True)
def translation_catalog(monkeypatch, tmp_path):
    # Real verse counts, without loading Whisper or building recognition candidates.
    catalog = json.loads((API_DIR / "assets" / "quran_versets.json").read_text())
    surahs = {surah["id"]: {"total_verses": len(surah["verses"])} for surah in catalog}
    monkeypatch.setattr(service, "get_surah_metadata", surahs.get)
    monkeypatch.setattr(importer, "load_quran_catalog", lambda: None)
    monkeypatch.setenv("QURAN_TRANSLATION_PATH", str(tmp_path / "translations.json"))
    service.clear_quran_translation_cache()
    yield
    service.clear_quran_translation_cache()


def test_bundled_pilot_preserves_original_quranenc_text_notes_and_version():
    data_path = API_DIR / "assets" / "quran_translation_fr.json"
    data = json.loads(data_path.read_text(encoding="utf-8"))
    index = service.load_translation_index(data_path)

    expected = {(1, ayah) for ayah in range(1, 8)} | {(2, ayah) for ayah in range(1, 6)} | {(2, 255)}
    assert set(index) == expected
    assert len(data["source_responses"]) == len(expected)
    metadata = importer.translation_metadata(data["meta"]["metadata_response"]["payload"])
    source_references = set()
    for source in data["source_responses"]:
        result = source["payload"]["result"]
        reference = (int(result["sura"]), int(result["aya"]))
        source_references.add(reference)
        entry = index[reference]
        assert entry.text == result["translation"]
        assert entry.footnotes == result["footnotes"]
        assert str(entry.source_url) == source["url"] == importer.verse_url(*reference)
        assert entry.version == data["meta"]["version"] == metadata["version"]
        assert entry.source == "quranenc"
        assert entry.translator == "Rachid Maach"
    assert source_references == expected


def test_local_reads_match_exact_surah_and_ayah_and_keep_partial_ranges_ordered(tmp_path):
    data = snapshot()
    data["translations"].reverse()
    importer.write_snapshot(tmp_path / "translations.json", data)

    first = service.fetch_quran_translations(1, 1, 1)[0]
    second = service.fetch_quran_translations(2, 1, 1)[0]
    assert (first.surah_id, first.ayah, first.text) == (1, 1, "  Texte fictif 1:1.\n  ")
    assert (second.surah_id, second.ayah, second.text) == (2, 1, "  Texte fictif 2:1.\n  ")
    assert [entry.ayah for entry in service.fetch_quran_translations(2, 4, 7)] == [4, 5]
    assert service.fetch_quran_translations(2, 254, 254) == []
    assert service.fetch_quran_translations(2, 255, 255)[0].text == "  Texte fictif 2:255.\n  "


@pytest.mark.parametrize("corruption", ["duplicate", "invalid_reference", "wrong_version"])
def test_invalid_snapshot_never_replaces_previous_file(tmp_path, corruption):
    data_path = tmp_path / "translations.json"
    importer.write_snapshot(data_path, snapshot())
    original = data_path.read_bytes()
    data = snapshot()
    if corruption == "duplicate":
        data["translations"].append(copy.deepcopy(data["translations"][0]))
    elif corruption == "invalid_reference":
        data["translations"][0]["ayah"] = 8  # Al-Fatiha has only seven verses.
    else:
        data["translations"][0]["version"] = "another-version"

    with pytest.raises(service.QuranTranslationServiceError):
        importer.write_snapshot(data_path, data)
    assert data_path.read_bytes() == original


@pytest.mark.parametrize("reference", [(1, 8, 8), (2, 5, 1)])
def test_local_reads_reject_invalid_verse_references(reference):
    with pytest.raises(service.QuranTranslationServiceError):
        service.fetch_quran_translations(*reference)


def test_failed_local_load_can_recover_when_snapshot_becomes_available(tmp_path):
    with pytest.raises(service.QuranTranslationServiceError, match="lire"):
        service.fetch_quran_translations(1, 1, 1)
    importer.write_snapshot(tmp_path / "translations.json", snapshot())

    assert service.fetch_quran_translations(1, 1, 1)[0].ayah == 1


def mocked_source_fetch(monkeypatch, *, failure=None):
    metadata = provider_metadata()
    verses = provider_verses()
    if failure == "wrong_verse":
        verses[(2, 255)]["result"]["aya"] = "254"
    by_url = {importer.verse_url(*reference): payload for reference, payload in verses.items()}
    metadata_calls = 0

    def fetch(url):
        nonlocal metadata_calls
        if url == importer.METADATA_URL:
            metadata_calls += 1
            result = copy.deepcopy(metadata)
            if failure == "changed_version" and metadata_calls == 2:
                result["translations"][0]["version"] = "test-2"
            return result
        if failure == "unavailable" and url == importer.verse_url(2, 255):
            raise importer.QuranTranslationImportError("Source de test indisponible.")
        return by_url[url]

    monkeypatch.setattr(importer, "fetch_json", fetch)
    return metadata, verses


def test_import_pipeline_preserves_sources_and_writes_readable_local_dataset(monkeypatch, tmp_path):
    metadata, verses = mocked_source_fetch(monkeypatch)
    data_path = tmp_path / "translations.json"

    data = importer.import_pilot(data_path)

    assert data["meta"]["metadata_response"] == {"url": importer.METADATA_URL, "payload": metadata}
    assert len(data["translations"]) == 13
    assert data["source_responses"][-1]["payload"] == verses[(2, 255)]
    entry = service.fetch_quran_translations(2, 255, 255)[0]
    assert entry.text == verses[(2, 255)]["result"]["translation"]
    assert entry.footnotes == verses[(2, 255)]["result"]["footnotes"]
    assert entry.version == "test-1"


@pytest.mark.parametrize("failure", ["wrong_verse", "unavailable", "changed_version"])
def test_failed_import_preserves_previous_snapshot(monkeypatch, tmp_path, failure):
    data_path = tmp_path / "translations.json"
    importer.write_snapshot(data_path, snapshot())
    original = data_path.read_bytes()
    mocked_source_fetch(monkeypatch, failure=failure)

    with pytest.raises(importer.QuranTranslationImportError):
        importer.import_pilot(data_path)
    assert data_path.read_bytes() == original


def test_import_rejects_incomplete_pilot():
    verses = provider_verses()
    del verses[(2, 255)]
    with pytest.raises(importer.QuranTranslationImportError, match="pilote complet"):
        importer.build_snapshot(provider_metadata(), verses)


@pytest.mark.parametrize("identifier", [True, 2.5])
def test_provider_identifiers_cannot_be_coerced_to_a_different_verse(identifier):
    verses = provider_verses()
    verses[(2, 255)]["result"]["sura"] = identifier
    with pytest.raises(importer.QuranTranslationImportError, match="Identifiant"):
        importer.build_snapshot(provider_metadata(), verses)


@pytest.mark.parametrize("field,value", [("language_iso_code", "en"), ("version", ""), ("key", "other")])
def test_import_requires_selected_french_source_and_real_version(field, value):
    metadata = provider_metadata()
    metadata["translations"][0][field] = value
    with pytest.raises(importer.QuranTranslationImportError):
        importer.build_snapshot(metadata, provider_verses())
