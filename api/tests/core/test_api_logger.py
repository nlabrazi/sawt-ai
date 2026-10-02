import json

from app.core.api_logger import log_api_error, log_api_event


def test_log_api_event_writes_json_to_stdout(capsys):
    log_api_event(
        level="info",
        message="Recognize request received",
        route="/recognize",
        extra={"requestId": "request-1"},
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert captured.err == ""
    assert payload == {
        "app": "sawt-ai",
        "level": "info",
        "message": "Recognize request received",
        "route": "/recognize",
        "requestId": "request-1",
    }


def test_log_api_error_writes_json_to_stderr(capsys):
    log_api_error(
        error=RuntimeError("boom"),
        message="Analysis failed",
        route="/recognize",
        status_code=503,
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.err)

    assert captured.out == ""
    assert payload == {
        "app": "sawt-ai",
        "level": "error",
        "message": "Analysis failed",
        "route": "/recognize",
        "statusCode": 503,
        "error": "boom",
        "errorType": "RuntimeError",
    }


def test_log_api_error_includes_nested_causes(capsys):
    root = FileNotFoundError("hadith_index_meta.json missing")
    index = RuntimeError("Invalid index")
    index.__cause__ = root
    error = RuntimeError("Search unavailable")
    error.__cause__ = index

    log_api_error(error=error, message="Search failed", status_code=503)

    payload = json.loads(capsys.readouterr().err)
    assert payload["error"] == "Search unavailable"
    assert payload["errorCauses"] == [
        {"error": "Invalid index", "errorType": "RuntimeError"},
        {"error": "hadith_index_meta.json missing", "errorType": "FileNotFoundError"},
    ]


def test_log_api_error_stops_at_a_cyclic_cause(capsys):
    error = RuntimeError("Search unavailable")
    cause = ValueError("Invalid index")
    error.__cause__ = cause
    cause.__cause__ = error

    log_api_error(error=error, message="Search failed")

    assert json.loads(capsys.readouterr().err)["errorCauses"] == [
        {"error": "Invalid index", "errorType": "ValueError"},
    ]
