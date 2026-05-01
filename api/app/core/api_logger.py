import json
import os
import sys
from collections.abc import Mapping
from typing import Any

from starlette.requests import Request

APP_NAME = os.getenv("APP_NAME", "sawt-ai")


def _request_context(event: Request | None) -> dict[str, Any]:
    if event is None:
        return {}

    return {
        "route": event.url.path,
        "method": event.method,
    }


def _error_to_string(error: BaseException | str | None) -> str | None:
    if error is None:
        return None

    error_message = str(error)
    if error_message:
        return error_message

    return error.__class__.__name__ if isinstance(error, BaseException) else error_message


def log_api_event(
    *,
    event: Request | None = None,
    level: str = "info",
    message: str,
    route: str | None = None,
    status_code: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "app": APP_NAME,
        "level": level,
        "message": message,
    }

    payload.update(_request_context(event))

    if route is not None:
        payload["route"] = route

    if status_code is not None:
        payload["statusCode"] = status_code

    if extra:
        payload.update(extra)

    stream = sys.stderr if level in {"error", "critical"} else sys.stdout
    print(json.dumps(payload, ensure_ascii=False, default=str), file=stream, flush=True)


def log_api_error(
    *,
    error: BaseException | str | None,
    event: Request | None = None,
    level: str = "error",
    message: str,
    route: str | None = None,
    status_code: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    payload_extra = dict(extra or {})
    payload_extra["error"] = _error_to_string(error)

    if isinstance(error, BaseException):
        payload_extra["errorType"] = error.__class__.__name__

    log_api_event(
        event=event,
        level=level,
        message=message,
        route=route,
        status_code=status_code,
        extra=payload_extra,
    )
