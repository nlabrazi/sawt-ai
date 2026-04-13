import pytest

from app.services.feedback_store import (
    FeedbackStoreConfigError,
    _build_supabase_headers,
    _get_supabase_api_key,
    _get_supabase_url,
)


def clear_supabase_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_API_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)


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
