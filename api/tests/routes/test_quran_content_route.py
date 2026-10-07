import asyncio
from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from app.main import app
import app.core.model_loader as loader
import app.routes.quran_content as content
import app.services.quran_translation_service as translations
import app.services.tafsir_store as store
from app.schemas.tafsir import TafsirEntry

API_DIR = Path(__file__).resolve().parents[2]
PARAMS = {"surah_id": 2, "start_verse": 255, "end_verse": 255}
NOW = "2026-10-07T10:00:00Z"


def tafsir_row(source="ibn_kathir", status="verified", ayah=255):
    return {
        "surah_id": 2, "ayah": ayah, "source": source,
        "text_fr": f"Commentaire fictif de test ({source}), sans contenu religieux.",
        "source_reference": "Édition fictive de test", "version": "test-1",
        "status": status, "reviewed_at": NOW if status == "verified" else None,
    }


def request(params=None):
    async def run():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            return await client.get("/quran/content", params=params or PARAMS)
    return asyncio.run(run())


def fake_storage(monkeypatch, replies):
    requests = []
    results = iter(replies)

    def open_request(request, timeout):
        requests.append(request)
        result = next(results)
        if isinstance(result, Exception):
            raise result
        return BytesIO(json.dumps(result).encode("utf-8"))
    monkeypatch.setattr(store, "urlopen", open_request)
    return requests


@pytest.fixture(autouse=True)
def configured_content(monkeypatch):
    catalog = json.loads((API_DIR / "assets/quran_versets.json").read_text())
    monkeypatch.setattr(loader, "quran_versets", catalog)
    monkeypatch.setenv("QURAN_TRANSLATION_PATH", str(API_DIR / "assets/quran_translation_fr.json"))
    monkeypatch.setenv("SUPABASE_URL", "https://project.example.test")
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_secret_fictitious_test_key")
    monkeypatch.delenv("TAFSIR_REVIEW_PASSWORD", raising=False)
    # Execute the real services with local files and simulated REST, without threads or startup models.
    async def direct_call(operation, *args, **kwargs):
        return operation(*args, **kwargs)
    monkeypatch.setattr(content, "run_in_threadpool", direct_call)
    translations.clear_quran_translation_cache()
    yield
    translations.clear_quran_translation_cache()


def test_public_http_response_excludes_drafts_and_keeps_verified_sources_and_translation(monkeypatch):
    draft = {**tafsir_row(status="need_review", ayah=254),
             "text_fr": "PRIVATE_DRAFT_SENTINEL", "provenance": {"source_text": "PRIVATE_ORIGINAL_SENTINEL"}}
    verified = [tafsir_row(), tafsir_row("as_saadi")]
    requests = fake_storage(monkeypatch, [[draft, *verified]])

    response = request({"surah_id": 2, "start_verse": 254, "end_verse": 255})

    assert response.status_code == 200
    body = response.json()
    assert (body["surah_id"], body["start_verse"], body["end_verse"]) == (2, 254, 255)
    assert body["tafsir_status"] == "available"
    assert body["ayahs"][0] == {"ayah": 254, "translation": None, "tafsirs": []}
    assert body["ayahs"][1]["tafsirs"] == verified
    snapshot = json.loads((API_DIR / "assets/quran_translation_fr.json").read_text())
    expected = next(entry for entry in snapshot["translations"] if (entry["surah_id"], entry["ayah"]) == (2, 255))
    assert body["ayahs"][1]["translation"] == expected
    for private in ("PRIVATE_DRAFT_SENTINEL", "PRIVATE_ORIGINAL_SENTINEL", "need_review", "provenance", "updated_at", "sb_secret_"):
        assert private not in response.text
    params = parse_qs(urlsplit(requests[0].full_url).query)
    assert params["status"] == ["eq.verified"]
    assert params["surah_id"] == ["eq.2"] and params["ayah"] == ["gte.254", "lte.255"]
    assert "provenance" not in params["select"][0] and "updated_at" not in params["select"][0]
    assert response.headers["Cache-Control"] == "no-store"


@pytest.mark.parametrize("surah,ayah", [(1, 1), (2, 1), (2, 254)])
def test_translation_matches_exact_surah_and_ayah_including_valid_missing_pilot_content(monkeypatch, surah, ayah):
    fake_storage(monkeypatch, [[]])
    response = request({"surah_id": surah, "start_verse": ayah, "end_verse": ayah})
    assert response.status_code == 200
    snapshot = json.loads((API_DIR / "assets/quran_translation_fr.json").read_text())
    expected = next((entry for entry in snapshot["translations"] if (entry["surah_id"], entry["ayah"]) == (surah, ayah)), None)
    assert response.json()["ayahs"] == [{"ayah": ayah, "translation": expected, "tafsirs": []}]


@pytest.mark.parametrize("corruption", [{"surah_id": 1}, {"ayah": 256}, {"source": "other"}, {"reviewed_at": None}])
def test_incoherent_verified_storage_rows_cannot_be_exposed_publicly(monkeypatch, corruption):
    fake_storage(monkeypatch, [[{**tafsir_row(), **corruption}]])
    response = request()
    assert response.status_code == 200
    body = response.json()
    assert body["tafsir_status"] == "unavailable"
    assert body["ayahs"][0]["tafsirs"] == []
    assert body["ayahs"][0]["translation"]["ayah"] == 255
    assert "Commentaire fictif" not in response.text


def test_public_schema_also_blocks_an_unreviewed_entry_if_the_read_service_regresses(monkeypatch, capsys):
    draft = TafsirEntry.model_validate({**tafsir_row(status="need_review"), "text_fr": "PRIVATE_DRAFT_SENTINEL"})
    monkeypatch.setattr(content, "fetch_verified_tafsirs", lambda *args: [draft])
    response = request()
    assert response.status_code == 200
    assert response.json()["ayahs"][0]["tafsirs"] == []
    assert response.json()["tafsir_status"] == "unavailable"
    assert "PRIVATE_DRAFT_SENTINEL" not in response.text + capsys.readouterr().out


@pytest.mark.parametrize("failure", [
    HTTPError("https://example.test", 404, "Not Found", {}, BytesIO(b"PRIVATE_STORAGE_DETAILS")),
    URLError("PRIVATE_STORAGE_DETAILS"),
    store.TafsirStoreConfigError("PRIVATE_STORAGE_DETAILS"),
])
def test_storage_unavailability_preserves_translation_without_exposing_details(monkeypatch, failure, capsys):
    fake_storage(monkeypatch, [failure])
    response = request()
    assert response.status_code == 200
    body = response.json()
    assert body["tafsir_status"] == "unavailable" and body["ayahs"][0]["tafsirs"] == []
    assert body["ayahs"][0]["translation"]["translator"] == "Rachid Maach"
    assert "PRIVATE_STORAGE_DETAILS" not in response.text + capsys.readouterr().out
    assert response.headers["Cache-Control"] == "no-store"


def test_review_status_changes_take_effect_on_the_next_http_read_without_caching(monkeypatch):
    pending = {**tafsir_row(status="need_review"), "text_fr": "PRIVATE_CORRECTION_SENTINEL"}
    requests = fake_storage(monkeypatch, [[pending], [tafsir_row()], [pending]])
    responses = [request() for _ in range(3)]
    assert all(response.status_code == 200 for response in responses)
    assert [len(response.json()["ayahs"][0]["tafsirs"]) for response in responses] == [0, 1, 0]
    assert all("PRIVATE_CORRECTION_SENTINEL" not in response.text for response in responses)
    assert all(response.headers["Cache-Control"] == "no-store" for response in responses)
    assert len(requests) == 3


@pytest.mark.parametrize("params,status", [
    ({"surah_id": 0, "start_verse": 1, "end_verse": 1}, 422),
    ({"surah_id": 2, "start_verse": 0, "end_verse": 1}, 422),
    ({"surah_id": 1, "start_verse": 8, "end_verse": 8}, 400),
    ({"surah_id": 2, "start_verse": 256, "end_verse": 255}, 400),
])
def test_invalid_ranges_are_rejected_before_content_access(monkeypatch, params, status):
    read = Mock(side_effect=AssertionError("Invalid ranges must not load content"))
    monkeypatch.setattr(content, "fetch_quran_translations", read)
    monkeypatch.setattr(content, "fetch_verified_tafsirs", read)
    assert request(params).status_code == status
    read.assert_not_called()


def test_missing_local_translation_returns_a_safe_error_before_any_tafsir_request(monkeypatch, tmp_path):
    monkeypatch.setenv("QURAN_TRANSLATION_PATH", str(tmp_path / "PRIVATE_SNAPSHOT_PATH.json"))
    requests = fake_storage(monkeypatch, [])
    response = request()
    assert response.status_code == 503
    assert response.json() == {"detail": "La traduction française est temporairement indisponible."}
    assert "PRIVATE_SNAPSHOT_PATH" not in response.text
    assert response.headers["Cache-Control"] == "no-store"
    assert requests == []


def test_openapi_documents_only_verified_tafsirs_in_public_content():
    schema = app.openapi()
    contract = schema["paths"]["/quran/content"]["get"]
    assert "security" not in contract
    assert contract["responses"]["200"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/QuranContentResponse"}
    public = schema["components"]["schemas"]["VerifiedTafsirEntry"]
    assert public["properties"]["status"]["const"] == "verified"
    assert "reviewed_at" in public["required"]
    assert "provenance" not in public["properties"] and "updated_at" not in public["properties"]
