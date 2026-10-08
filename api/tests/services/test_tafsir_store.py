import base64
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import pytest

import app.services.tafsir_import_service as importer
import app.services.tafsir_store as store

NOW = "2026-10-07T10:00:00+00:00"
LATER = "2026-10-07T10:01:00+00:00"


def snapshot(source="ibn_kathir"):
    return {
        "schema_version": 1, "source": source, "source_language": "ar",
        "source_edition": "Édition fictive de test", "version": "test-1",
        "reuse_reference": "Conditions fictives du corpus de test", "imported_at": NOW,
        "entries": [{
            "surah_id": 2, "ayah": 255, "source": source, "version": "test-1",
            "text_fr": "  Brouillon fictif, sans contenu religieux.\n  ",
            "source_reference": "Édition fictive, passage 2:255",
            "source_text": "  Passage original fictif.\n  ", "source_surah_id": 2,
            "source_start_ayah": 255, "source_end_ayah": 255,
            "status": "need_review", "reviewed_at": None,
        }],
    }


def row(source="ibn_kathir", status="need_review"):
    entry = snapshot(source)["entries"][0]
    return {field: value for field, value in entry.items() if field in store.TafsirEntry.model_fields} | {
        "status": status, "reviewed_at": LATER if status == "verified" else None,
        "provenance": {"source_text": "Passage fictif de test"}, "updated_at": NOW,
    }


def public_row(source="ibn_kathir", status="verified"):
    return {field: value for field, value in row(source, status).items() if field in store.TafsirEntry.model_fields}


def fake_responses(monkeypatch, responses):
    requests = []
    results = iter(responses)

    def open_request(request, timeout):
        requests.append(request)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return BytesIO(json.dumps(result).encode("utf-8"))

    monkeypatch.setattr(store, "urlopen", open_request)
    return requests


def query(request):
    return parse_qs(urlsplit(request.full_url).query)


def legacy_key(role):
    payload = base64.urlsafe_b64encode(json.dumps({"role": role}).encode()).decode().rstrip("=")
    return f"fakeheader.{payload}.fakesignature"


@pytest.fixture(autouse=True)
def configured_store(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://project.example.test")
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_secret_fake_test_key")
    catalog = json.loads((Path(__file__).resolve().parents[2] / "assets/quran_versets.json").read_text())
    surahs = {surah["id"]: {"total_verses": len(surah["verses"])} for surah in catalog}
    monkeypatch.setattr(store, "get_surah_metadata", surahs.get)
    monkeypatch.setattr(importer, "get_surah_metadata", surahs.get)


def test_bulk_inserts_preserve_provenance_and_keep_both_sources_separate(monkeypatch):
    requests = fake_responses(monkeypatch, [None, None])

    for source in ("ibn_kathir", "as_saadi"):
        data = snapshot(source)
        assert store.insert_tafsir_import(data) == 1
        request = requests[-1]
        stored = json.loads(request.data)[0]
        assert (stored["surah_id"], stored["ayah"], stored["source"]) == (2, 255, source)
        assert stored["status"] == "need_review" and stored["reviewed_at"] is None
        assert stored["text_fr"] == data["entries"][0]["text_fr"]
        assert stored["provenance"]["source_text"] == data["entries"][0]["source_text"]
        assert stored["provenance"]["source_edition"] == data["source_edition"]
        assert stored["provenance"]["reuse_reference"] == data["reuse_reference"]
        imported_at = datetime.fromisoformat(stored["provenance"]["imported_at"].replace("Z", "+00:00"))
        assert imported_at == datetime.fromisoformat(data["imported_at"])
        assert request.method == "POST"
        assert request.get_header("Prefer") == "return=minimal"
        assert not query(request)  # No on_conflict/upsert option.


def test_store_revalidates_imports_before_any_write(monkeypatch):
    requests = fake_responses(monkeypatch, [])
    data = snapshot()
    data["entries"][0].update({"status": "verified", "reviewed_at": NOW})

    with pytest.raises(importer.TafsirImportError):
        store.insert_tafsir_import(data)
    assert requests == []

    data = snapshot()
    data.pop("imported_at")
    with pytest.raises(store.TafsirStoreError, match="horodaté"):
        store.insert_tafsir_import(data)
    assert requests == []


def test_duplicate_import_reports_conflict_without_retry_or_merge(monkeypatch):
    requests = fake_responses(monkeypatch, [HTTPError("https://example.test", 409, "Conflict", {}, BytesIO(b"private details"))])

    with pytest.raises(store.TafsirStoreConflict, match="aucun texte"):
        store.insert_tafsir_import(snapshot())
    assert len(requests) == 1
    assert requests[0].get_header("Prefer") == "return=minimal"


def test_public_read_filters_pending_rows_and_exposes_both_verified_sources(monkeypatch):
    returned = [public_row(status="need_review"), public_row(), public_row("as_saadi")]
    requests = fake_responses(monkeypatch, [returned])

    entries = store.fetch_verified_tafsirs(2, 254, 255)

    assert [(entry.surah_id, entry.ayah, entry.source) for entry in entries] == [(2, 255, "ibn_kathir"), (2, 255, "as_saadi")]
    assert all(entry.status == "verified" for entry in entries)
    assert all("provenance" not in entry.model_dump() for entry in entries)
    params = query(requests[0])
    assert params["surah_id"] == ["eq.2"]
    assert params["ayah"] == ["gte.254", "lte.255"]
    assert params["status"] == ["eq.verified"]
    assert "provenance" not in params["select"][0]
    assert "updated_at" not in params["select"][0]


@pytest.mark.parametrize("corruption", [{"surah_id": 1}, {"ayah": 256}, {"reviewed_at": None}])
def test_public_read_fails_closed_on_wrong_reference_or_invalid_verification(monkeypatch, corruption):
    fake_responses(monkeypatch, [[{**public_row(), **corruption}]])

    with pytest.raises(store.TafsirStoreError, match="incohérente"):
        store.fetch_verified_tafsirs(2, 255, 255)


def test_review_listing_uses_requested_filters_and_preserves_original_passage(monkeypatch):
    requests = fake_responses(monkeypatch, [[row("as_saadi")]])

    entries = store.list_tafsirs_for_review(surah_id=2, source="as_saadi", status="need_review", limit=10, offset=10)

    assert entries[0].provenance == {"source_text": "Passage fictif de test"}
    params = query(requests[0])
    assert params["surah_id"] == ["eq.2"]
    assert params["source"] == ["eq.as_saadi"]
    assert params["status"] == ["eq.need_review"]
    assert (params["limit"], params["offset"]) == (["10"], ["10"])


def test_manual_validation_targets_the_exact_source_and_reviewed_revision(monkeypatch):
    verified = {**row(status="verified"), "updated_at": LATER}
    requests = fake_responses(monkeypatch, [[verified]])

    entry = store.verify_tafsir(2, 255, "ibn_kathir", expected_updated_at=datetime.fromisoformat(NOW))

    assert entry.status == "verified" and entry.reviewed_at is not None
    assert entry.text_fr == row()["text_fr"]
    request = requests[0]
    assert request.method == "PATCH" and request.get_header("Prefer") == "return=representation"
    assert json.loads(request.data) == {"status": "verified"}  # The database assigns the date.
    params = query(request)
    assert params["surah_id"] == ["eq.2"] and params["ayah"] == ["eq.255"]
    assert params["source"] == ["eq.ibn_kathir"] and params["status"] == ["eq.need_review"]
    assert params["updated_at"] == [f"eq.{NOW}"]


def test_text_edit_requests_a_new_review_and_preserves_whitespace(monkeypatch):
    text = "  Correction fictive pour ce test.\n  "
    requests = fake_responses(monkeypatch, [[{**row(), "text_fr": text, "updated_at": LATER}]])

    entry = store.update_tafsir_text(2, 255, "ibn_kathir", text, expected_updated_at=datetime.fromisoformat(NOW))

    assert entry.text_fr == text and entry.status == "need_review" and entry.reviewed_at is None
    assert json.loads(requests[0].data) == {"text_fr": text, "status": "need_review"}
    assert query(requests[0])["updated_at"] == [f"eq.{NOW}"]


@pytest.mark.parametrize("action", ["verify", "edit"])
def test_stale_review_or_missing_entry_cannot_be_saved_or_validated(monkeypatch, action):
    fake_responses(monkeypatch, [[]])

    with pytest.raises(store.TafsirStoreConflict, match="recharger"):
        if action == "verify":
            store.verify_tafsir(2, 255, "ibn_kathir", expected_updated_at=datetime.fromisoformat(NOW))
        else:
            store.update_tafsir_text(2, 255, "ibn_kathir", "Correction fictive.", expected_updated_at=datetime.fromisoformat(NOW))


def test_validation_rejects_a_result_for_the_other_source(monkeypatch):
    fake_responses(monkeypatch, [[row("as_saadi", "verified")]])

    with pytest.raises(store.TafsirStoreError, match="incohérent"):
        store.verify_tafsir(2, 255, "ibn_kathir", expected_updated_at=datetime.fromisoformat(NOW))


@pytest.mark.parametrize("key,authorization", [("sb_secret_fake_test_key", None), (legacy_key("service_role"), True)])
def test_authentication_headers_match_the_supabase_key_type(monkeypatch, key, authorization):
    monkeypatch.setenv("SUPABASE_API_KEY", key)
    requests = fake_responses(monkeypatch, [[]])
    store.fetch_verified_tafsirs(2, 255, 255)

    assert requests[0].get_header("Apikey") == key
    assert requests[0].get_header("Authorization") == (f"Bearer {key}" if authorization else None)


@pytest.mark.parametrize("key", [legacy_key("anon"), legacy_key("authenticated"), "invalid.!.token"])
def test_client_or_malformed_keys_are_rejected_before_requests(monkeypatch, key):
    monkeypatch.setenv("SUPABASE_API_KEY", key)
    requests = fake_responses(monkeypatch, [])

    with pytest.raises(store.TafsirStoreConfigError):
        store.fetch_verified_tafsirs(2, 255, 255)
    assert requests == []


@pytest.mark.parametrize("failure", [HTTPError("https://example.test", 503, "Unavailable", {}, BytesIO(b"private text")), URLError("offline")])
def test_storage_outage_is_reported_without_returning_content(monkeypatch, failure):
    fake_responses(monkeypatch, [failure])

    with pytest.raises(store.TafsirStoreError):
        store.fetch_verified_tafsirs(2, 255, 255)
