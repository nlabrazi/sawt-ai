import asyncio

import app.main as main
from app.core.upload_policy import build_upload_policy

from app.main import app, build_cors_options, parse_allowed_origins


def test_parse_allowed_origins_normalizes_and_deduplicates_values():
    origins = parse_allowed_origins(
        " https://sawt-ai.netlify.app/ , http://localhost:3000,https://sawt-ai.netlify.app "
    )

    assert origins == [
        "https://sawt-ai.netlify.app",
        "http://localhost:3000",
    ]


def test_build_cors_options_uses_explicit_origin_allowlist_without_regex():
    options = build_cors_options("https://sawt-ai.netlify.app,http://localhost:3000")

    assert options["allow_origins"] == [
        "https://sawt-ai.netlify.app",
        "http://localhost:3000",
    ]
    assert options["allow_credentials"] is True
    assert options["allow_methods"] == ["*"]
    assert options["allow_headers"] == ["*"]
    assert "allow_origin_regex" not in options


def test_openapi_documents_recognize_response_contract():
    schema = app.openapi()
    recognize_response = schema["paths"]["/recognize"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]

    assert recognize_response == {"$ref": "#/components/schemas/RecognizeResponse"}
    assert "imam_detection_enabled" in schema["components"]["schemas"]["RecognizeResponse"]["required"]


def test_health_reports_imam_detection_service_status(monkeypatch):
    monkeypatch.setattr(
        main,
        "get_imam_service_health",
        lambda: {
            "available": False,
            "status": "unavailable",
            "message": "La reconnaissance de l’imam est temporairement indisponible.",
        },
    )

    assert main.health() == {
        "status": "ok",
        "services": {
            "imam_detection": {
                "available": False,
                "status": "unavailable",
                "message": "La reconnaissance de l’imam est temporairement indisponible.",
            },
            "upload_policy": build_upload_policy(),
        },
    }


def test_lifespan_warms_tajwid_cache_without_blocking_startup(monkeypatch):
    calls = []
    warnings = []

    monkeypatch.setattr(main, "load_all_models", lambda: calls.append("models"))
    monkeypatch.setattr(main, "preflight_imam_resources", lambda: calls.append("imam"))

    def fail_warmup():
        calls.append("tajwid")
        raise main.TajwidServiceError("tajwid unavailable")

    monkeypatch.setattr(main, "warm_tajwid_cache", fail_warmup)
    monkeypatch.setattr(main.logger, "warning", lambda message, **kwargs: warnings.append((message, kwargs)))

    async def run_lifespan():
        async with main.lifespan(app):
            calls.append("yield")

    asyncio.run(run_lifespan())

    assert calls == ["models", "imam", "tajwid", "yield"]
    assert warnings == [(
        "Tajwid warmup failed during startup; the API will retry on demand.",
        {"exc_info": True},
    )]
