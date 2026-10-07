import asyncio
from datetime import datetime
from unittest.mock import Mock

import httpx
import pytest

from app.main import app
import app.routes.tafsir_review as review

NOW = "2026-10-07T10:00:00Z"
PASSWORD = "dedicated-fictitious-review-password"
HEADERS = {"Authorization": f"Bearer {PASSWORD}"}


def row(source="ibn_kathir", status="need_review"):
    return {
        "surah_id": 2, "ayah": 255, "source": source,
        "text_fr": "Brouillon fictif de test, sans contenu religieux.",
        "source_reference": "Édition fictive, 2:255", "version": "test-1",
        "status": status, "reviewed_at": NOW if status == "verified" else None,
        "provenance": {"source_text": "Passage fictif de test"}, "updated_at": NOW,
    }


def request(method, path="", **kwargs):
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            return await client.request(method, f"/internal/tafsir{path}", **kwargs)
    return asyncio.run(run())


@pytest.fixture(autouse=True)
def setup_review(monkeypatch):
    monkeypatch.setenv("TAFSIR_REVIEW_PASSWORD", PASSWORD)
    monkeypatch.setattr(review, "get_surah_metadata", lambda surah: {"total_verses": 7 if surah == 1 else 286})

    async def direct_call(operation, *args, **kwargs):
        return operation(*args, **kwargs)
    monkeypatch.setattr(review, "run_in_threadpool", direct_call)


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic wrong"}])
def test_all_review_operations_reject_unauthenticated_requests_before_storage(monkeypatch, headers):
    store = Mock(side_effect=AssertionError("Storage must not be reached"))
    for name in ("list_tafsirs_for_review", "update_tafsir_text", "verify_tafsir"):
        monkeypatch.setattr(review.tafsir_store, name, store)
    for method, path, body in (
        ("GET", "", None), ("GET", "/access", None),
        ("PATCH", "/2/255/ibn_kathir", {"text_fr": "Correction fictive", "expected_updated_at": NOW}),
        ("POST", "/2/255/ibn_kathir/verify", {"expected_updated_at": NOW}),
    ):
        response = request(method, path, headers=headers, json=body)
        assert response.status_code == 401
        assert response.headers["Cache-Control"] == "no-store"
        assert "Authorization" in response.headers["Vary"]
        assert response.headers["X-Robots-Tag"] == "noindex, nofollow"
        assert "Brouillon" not in response.text
    store.assert_not_called()


def test_missing_password_disables_review_and_access_check_does_not_use_supabase(monkeypatch):
    store = Mock(side_effect=AssertionError("No Supabase access required"))
    monkeypatch.setattr(review.tafsir_store, "list_tafsirs_for_review", store)
    assert request("GET", "/access", headers=HEADERS).status_code == 204
    monkeypatch.delenv("TAFSIR_REVIEW_PASSWORD")
    assert request("GET", headers=HEADERS).status_code == 503
    store.assert_not_called()


def test_authenticated_listing_preserves_filters_and_original_passage(monkeypatch):
    store = Mock(return_value=[row("as_saadi")])
    monkeypatch.setattr(review.tafsir_store, "list_tafsirs_for_review", store)
    response = request("GET", headers=HEADERS, params={"surah_id": 2, "source": "as_saadi", "status": "need_review", "limit": 10, "offset": 10})
    assert response.status_code == 200
    store.assert_called_once_with(surah_id=2, source="as_saadi", status="need_review", limit=10, offset=10)
    assert [entry["source"] for entry in response.json()] == ["as_saadi"]
    assert response.json()[0]["provenance"]["source_text"] == "Passage fictif de test"
    assert response.headers["Cache-Control"] == "no-store"


def test_correction_and_manual_verification_pass_exact_reference_and_review_revision(monkeypatch):
    save = Mock(return_value=row("as_saadi"))
    verify = Mock(return_value=row("as_saadi", "verified"))
    monkeypatch.setattr(review.tafsir_store, "update_tafsir_text", save)
    monkeypatch.setattr(review.tafsir_store, "verify_tafsir", verify)
    response = request("PATCH", "/2/255/as_saadi", headers=HEADERS,
                       json={"text_fr": "  Correction fictive.\n", "expected_updated_at": NOW})
    assert response.status_code == 200 and response.json()["status"] == "need_review"
    revision = datetime.fromisoformat(NOW.replace("Z", "+00:00"))
    save.assert_called_once_with(2, 255, "as_saadi", "  Correction fictive.\n", expected_updated_at=revision)
    response = request("POST", "/2/255/as_saadi/verify", headers=HEADERS, json={"expected_updated_at": NOW})
    assert response.status_code == 200 and response.json()["status"] == "verified"
    assert response.json()["reviewed_at"] is not None
    verify.assert_called_once_with(2, 255, "as_saadi", expected_updated_at=revision)


@pytest.mark.parametrize("method,path,body", [
    ("POST", "/2/255/ibn_kathir/verify", {}),
    ("POST", "/2/255/ibn_kathir/verify", {"expected_updated_at": "2026-10-07T10:00:00"}),
    ("PATCH", "/2/255/ibn_kathir", {"text_fr": "   ", "expected_updated_at": NOW}),
    ("PATCH", "/2/255/ibn_kathir", {"text_fr": "Test", "expected_updated_at": NOW, "status": "verified"}),
    ("POST", "/1/8/ibn_kathir/verify", {"expected_updated_at": NOW}),
    ("POST", "/2/255/other/verify", {"expected_updated_at": NOW}),
    ("GET", "?status=other", None),
])
def test_invalid_reviews_cannot_reach_storage(monkeypatch, method, path, body):
    store = Mock(side_effect=AssertionError("Invalid input must not reach storage"))
    for name in ("list_tafsirs_for_review", "update_tafsir_text", "verify_tafsir"):
        monkeypatch.setattr(review.tafsir_store, name, store)
    assert request(method, path, headers=HEADERS, json=body).status_code == 422
    store.assert_not_called()


@pytest.mark.parametrize("error,status", [
    (review.tafsir_store.TafsirStoreConflict, 409),
    (review.tafsir_store.TafsirStoreConfigError, 503),
    (review.tafsir_store.TafsirStoreError, 502),
])
def test_storage_errors_are_safe_and_conflicts_require_a_new_review(monkeypatch, error, status):
    monkeypatch.setattr(review.tafsir_store, "verify_tafsir", Mock(side_effect=error("private upstream details")))
    response = request("POST", "/2/255/ibn_kathir/verify", headers=HEADERS, json={"expected_updated_at": NOW})
    assert response.status_code == status
    assert "private upstream details" not in response.text
    assert response.headers["Cache-Control"] == "no-store"


def test_private_review_routes_are_not_in_the_public_openapi_schema():
    assert not any(path.startswith("/internal/tafsir") for path in app.openapi()["paths"])
