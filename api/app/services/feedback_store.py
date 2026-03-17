# ROLE
# ----
# Stockage V2 du feedback utilisateur dans une table Supabase via l'API REST.

from __future__ import annotations

import json
import logging
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

DEFAULT_SUPABASE_TIMEOUT_SECONDS = 15


class FeedbackStoreError(Exception):
    pass


class FeedbackStoreConfigError(FeedbackStoreError):
    pass


def _get_supabase_url() -> str:
    value = os.getenv("SUPABASE_URL", "").strip().rstrip("/")

    if not value:
        raise FeedbackStoreConfigError("SUPABASE_URL is not configured.")

    return value


def _get_supabase_secret() -> str:
    value = (
        os.getenv("SUPABASE_SECRET_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or ""
    ).strip()

    if not value:
        raise FeedbackStoreConfigError(
            "SUPABASE_SECRET_KEY or SUPABASE_SERVICE_ROLE_KEY is not configured."
        )

    return value


def _get_feedback_table() -> str:
    return os.getenv("SUPABASE_FEEDBACK_TABLE", "feedbacks").strip() or "feedbacks"


def save_feedback(payload: dict) -> None:
    supabase_url = _get_supabase_url()
    supabase_secret = _get_supabase_secret()
    table_name = _get_feedback_table()

    endpoint = f"{supabase_url}/rest/v1/{table_name}"
    body = json.dumps(payload).encode("utf-8")

    request = Request(
        endpoint,
        data=body,
        method="POST",
        headers={
            "apikey": supabase_secret,
            "Authorization": f"Bearer {supabase_secret}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )

    try:
        with urlopen(request, timeout=DEFAULT_SUPABASE_TIMEOUT_SECONDS):
            return
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        logger.exception("Supabase feedback insert failed with status %s", exc.code)
        raise FeedbackStoreError(
            f"Supabase feedback insert failed ({exc.code}): {error_body}"
        ) from exc
    except URLError as exc:
        logger.exception("Supabase feedback endpoint is unreachable")
        raise FeedbackStoreError("Supabase feedback endpoint is unreachable.") from exc
