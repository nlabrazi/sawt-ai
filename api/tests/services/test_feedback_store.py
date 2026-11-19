import io
import json
from urllib.error import HTTPError, URLError

import pytest

import app.services.feedback_store as feedback_store
from app.services.feedback_store import (
    FeedbackStoreConfigError,
    FeedbackStoreError,
    _build_supabase_headers,
    _get_feedback_table,
    _get_supabase_api_key,
    _get_supabase_url,
    save_feedback,
)


def clear_supabase_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_FEEDBACK_TABLE", raising=False)


def test_get_supabase_api_key_uses_primary_env_var(monkeypatch):
    clear_supabase_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_secret_server_key")

    assert _get_supabase_api_key() == "sb_secret_server_key"


def test_get_supabase_api_key_falls_back_to_legacy_env_var(monkeypatch):
    clear_supabase_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "legacy_service_role_key")

    assert _get_supabase_api_key() == "legacy_service_role_key"


def test_get_supabase_api_key_rejects_publishable_keys(monkeypatch):
    clear_supabase_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_publishable_client_key")

    with pytest.raises(FeedbackStoreConfigError, match="server-side key"):
        _get_supabase_api_key()


def test_get_supabase_api_key_requires_a_value(monkeypatch):
    clear_supabase_env(monkeypatch)

    with pytest.raises(FeedbackStoreConfigError, match="SUPABASE_API_KEY"):
        _get_supabase_api_key()


def test_get_supabase_url_requires_project_url(monkeypatch):
    clear_supabase_env(monkeypatch)
    monkeypatch.setenv(
        "SUPABASE_URL",
        "postgresql://postgres:postgres@db.example.supabase.co:5432/postgres",
    )

    with pytest.raises(FeedbackStoreConfigError, match="Project URL"):
        _get_supabase_url()


def test_build_supabase_headers_uses_server_key_for_apikey_and_authorization():
    headers = _build_supabase_headers("sb_secret_server_key")

    assert headers == {
        "apikey": "sb_secret_server_key",
        "Authorization": "Bearer sb_secret_server_key",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def test_get_supabase_url_requires_value_and_strips_trailing_slash(monkeypatch):
    clear_supabase_env(monkeypatch)
    with pytest.raises(FeedbackStoreConfigError, match="SUPABASE_URL is not configured"):
        _get_supabase_url()

    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co/")
    assert _get_supabase_url() == "https://project.supabase.co"


def test_get_feedback_table_uses_default_and_env_override(monkeypatch):
    clear_supabase_env(monkeypatch)
    assert _get_feedback_table() == "feedbacks"

    monkeypatch.setenv("SUPABASE_FEEDBACK_TABLE", "   ")
    assert _get_feedback_table() == "feedbacks"

    monkeypatch.setenv("SUPABASE_FEEDBACK_TABLE", "custom_feedbacks")
    assert _get_feedback_table() == "custom_feedbacks"


def test_save_feedback_posts_json_payload_and_handles_http_and_network_errors(monkeypatch):
    clear_supabase_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co/")
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_secret_server_key")
    monkeypatch.setenv("SUPABASE_FEEDBACK_TABLE", "user_feedbacks")

    captured = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["body"] = json.loads(request.data.decode("utf-8"))
        captured["apikey"] = request.get_header("Apikey")
        captured["timeout"] = timeout
        return DummyResponse()

    monkeypatch.setattr(feedback_store, "urlopen", fake_urlopen)

    save_feedback({"rating": 5, "comment": "Great"})

    assert captured == {
        "url": "https://project.supabase.co/rest/v1/user_feedbacks",
        "method": "POST",
        "body": {"rating": 5, "comment": "Great"},
        "apikey": "sb_secret_server_key",
        "timeout": feedback_store.DEFAULT_SUPABASE_TIMEOUT_SECONDS,
    }

    def fail_http(request, timeout):
        raise HTTPError(
            request.full_url,
            500,
            "Internal Server Error",
            hdrs={},
            fp=io.BytesIO(b"db error"),
        )

    monkeypatch.setattr(feedback_store, "urlopen", fail_http)
    with pytest.raises(FeedbackStoreError, match="Supabase feedback insert failed"):
        save_feedback({"rating": 1})

    def fail_network(_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr(feedback_store, "urlopen", fail_network)
    with pytest.raises(FeedbackStoreError, match="unreachable"):
        save_feedback({"rating": 1})
