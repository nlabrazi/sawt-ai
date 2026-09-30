# ROLE
# ----
# Stockage du feedback utilisateur dans une table Supabase via l'API REST.

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_SUPABASE_TIMEOUT_SECONDS = 15
PUBLISHABLE_KEY_PREFIX = "sb_publishable_"

_http_client: httpx.Client | None = None


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(timeout=DEFAULT_SUPABASE_TIMEOUT_SECONDS)
    return _http_client


def _close_http_client() -> None:
    global _http_client
    if _http_client is not None and not _http_client.is_closed:
        _http_client.close()
    _http_client = None


class FeedbackStoreError(Exception):
    pass


class FeedbackStoreConfigError(FeedbackStoreError):
    pass


def _get_supabase_url() -> str:
    value = os.getenv("SUPABASE_URL", "").strip().rstrip("/")

    if not value:
        raise FeedbackStoreConfigError("SUPABASE_URL is not configured.")

    if value.startswith("postgres://") or value.startswith("postgresql://"):
        raise FeedbackStoreConfigError(
            "SUPABASE_URL must be the Supabase Project URL, not the Postgres connection string."
        )

    return value


def _get_legacy_supabase_api_key() -> str:
    return (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_SECRET_KEY")
        or ""
    ).strip()


def _get_supabase_api_key() -> str:
    value = (
        os.getenv("SUPABASE_API_KEY")
        or _get_legacy_supabase_api_key()
    ).strip()

    if not value:
        raise FeedbackStoreConfigError("SUPABASE_API_KEY is not configured.")

    if value.startswith(PUBLISHABLE_KEY_PREFIX):
        raise FeedbackStoreConfigError(
            "SUPABASE_API_KEY must be a server-side key, not a publishable key."
        )

    return value


def _build_supabase_headers(api_key: str) -> dict[str, str]:
    return {
        "apikey": api_key,
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _get_feedback_table() -> str:
    return os.getenv("SUPABASE_FEEDBACK_TABLE", "feedbacks").strip() or "feedbacks"


def save_feedback(payload: dict) -> None:
    supabase_url = _get_supabase_url()
    supabase_api_key = _get_supabase_api_key()
    table_name = _get_feedback_table()

    endpoint = f"{supabase_url}/rest/v1/{table_name}"
    client = _get_http_client()

    try:
        response = client.post(
            endpoint,
            json=payload,
            headers=_build_supabase_headers(supabase_api_key),
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.exception(
            "Supabase feedback insert failed with status %s and body %s",
            exc.response.status_code,
            exc.response.text,
        )
        raise FeedbackStoreError("Supabase feedback insert failed.") from exc
    except httpx.RequestError as exc:
        logger.exception("Supabase feedback endpoint is unreachable")
        raise FeedbackStoreError("Supabase feedback endpoint is unreachable.") from exc
