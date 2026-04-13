# ROLE
# ----
# Stockage du feedback utilisateur dans une table Supabase via l'API REST.

from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_SUPABASE_TIMEOUT_SECONDS = 15
PUBLISHABLE_KEY_PREFIX = "sb_publishable_"


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
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers=_build_supabase_headers(supabase_api_key),
    )

    try:
        with urlopen(request, timeout=DEFAULT_SUPABASE_TIMEOUT_SECONDS):
            return
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.exception(
            "Supabase feedback insert failed with status %s and body %s",
            exc.code,
            error_body,
        )
        raise FeedbackStoreError("Supabase feedback insert failed.") from exc
    except URLError as exc:
        logger.exception("Supabase feedback endpoint is unreachable")
        raise FeedbackStoreError("Supabase feedback endpoint is unreachable.") from exc
