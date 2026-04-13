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
