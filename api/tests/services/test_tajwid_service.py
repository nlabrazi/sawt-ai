import json
from urllib.error import URLError

import pytest

import app.services.tajwid_service as tajwid_service
from app.services.tajwid_service import TajwidServiceError, fetch_tajwid_text


def build_tajwid_payload():
    return {
        "data": {
            "surahs": [
                {
                    "number": 112,
                    "ayahs": [
                        {"numberInSurah": 1, "text": "[rule[قل]]"},
                        {"numberInSurah": 2, "text": "[rule[هو]]"},
                        {"numberInSurah": 3, "text": "[rule[الله]]"},
                        {"numberInSurah": 4, "text": "[rule[أحد]]"},
                    ],
                },
            ],
        },
    }


class FakeUrlOpenResponse:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def resolve_request_url(request_or_url):
    return getattr(request_or_url, "full_url", request_or_url)


def resolve_request_headers(request_or_url):
    return dict(getattr(request_or_url, "headers", {}))


@pytest.fixture(autouse=True)
def reset_tajwid_cache(monkeypatch, tmp_path):
    tajwid_service.clear_tajwid_cache()
    monkeypatch.delenv("TAJWID_DATA_PATH", raising=False)
    monkeypatch.setattr(
        tajwid_service,
        "DEFAULT_TAJWID_DATA_PATH",
        tmp_path / "quran_tajwid.json",
    )
    yield
    tajwid_service.clear_tajwid_cache()


def test_fetch_tajwid_text_downloads_payload_once_and_reuses_cache(monkeypatch):
    calls = []

    def fake_urlopen(request_or_url, timeout):
        calls.append((resolve_request_url(request_or_url), timeout))
        return FakeUrlOpenResponse(build_tajwid_payload())

    monkeypatch.setattr(tajwid_service, "urlopen", fake_urlopen)

    first_response = fetch_tajwid_text(112, 1, 2)
    second_response = fetch_tajwid_text(112, 3, 4)

    assert first_response["text"] == "[rule[قل]] [rule[هو]]"
    assert first_response["ayahs"] == [
        {"number": 1, "tajwid_text": "[rule[قل]]"},
        {"number": 2, "tajwid_text": "[rule[هو]]"},
    ]
    assert second_response["text"] == "[rule[الله]] [rule[أحد]]"
    assert second_response["ayahs"] == [
        {"number": 3, "tajwid_text": "[rule[الله]]"},
        {"number": 4, "tajwid_text": "[rule[أحد]]"},
    ]
    assert calls == [(tajwid_service.TAJWID_API_BASE, tajwid_service.TAJWID_TIMEOUT_SECONDS)]


def test_fetch_tajwid_text_returns_a_structured_single_ayah(monkeypatch):
    monkeypatch.setattr(
        tajwid_service,
        "urlopen",
        lambda _request_or_url, timeout: FakeUrlOpenResponse(build_tajwid_payload()),
    )

    response = fetch_tajwid_text(112, 2, 2)

    assert response == {
        "surah_id": 112,
        "start_verse": 2,
        "end_verse": 2,
        "text": "[rule[هو]]",
        "ayahs": [
            {"number": 2, "tajwid_text": "[rule[هو]]"},
        ],
    }


def test_fetch_tajwid_text_prefers_local_snapshot_when_available(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "tajwid_snapshot.json"
    snapshot_path.write_text(json.dumps(build_tajwid_payload()), encoding="utf-8")

    def fail_urlopen(*_args, **_kwargs):
        raise AssertionError("urlopen should not be called")

    monkeypatch.setenv("TAJWID_DATA_PATH", str(snapshot_path))
    monkeypatch.setattr(tajwid_service, "urlopen", fail_urlopen)

    response = fetch_tajwid_text(112, 2, 3)

    assert response["text"] == "[rule[هو]] [rule[الله]]"
    assert response["ayahs"] == [
        {"number": 2, "tajwid_text": "[rule[هو]]"},
        {"number": 3, "tajwid_text": "[rule[الله]]"},
    ]


def test_fetch_tajwid_text_falls_back_to_backup_url_when_local_snapshot_is_missing(monkeypatch, tmp_path):
    missing_snapshot_path = tmp_path / "missing_tajwid_snapshot.json"
    calls = []

    def fake_urlopen(request_or_url, timeout):
        calls.append((resolve_request_url(request_or_url), resolve_request_headers(request_or_url), timeout))
        return FakeUrlOpenResponse(build_tajwid_payload())

    monkeypatch.setenv("TAJWID_DATA_PATH", str(missing_snapshot_path))
    monkeypatch.setenv("TAJWID_BACKUP_URL", "https://backup.example/tajwid.json")
    monkeypatch.setattr(tajwid_service, "urlopen", fake_urlopen)

    response = fetch_tajwid_text(112, 1, 2)

    assert response["text"] == "[rule[قل]] [rule[هو]]"
    assert calls == [("https://backup.example/tajwid.json", {}, tajwid_service.TAJWID_TIMEOUT_SECONDS)]


def test_fetch_tajwid_text_falls_back_to_api_when_backup_is_unreachable(monkeypatch):
    calls = []

    def fake_urlopen(request_or_url, timeout):
        url = resolve_request_url(request_or_url)
        calls.append((url, timeout))

        if url == "https://backup.example/tajwid.json":
            raise URLError("backup unavailable")

        if url == tajwid_service.TAJWID_API_BASE:
            return FakeUrlOpenResponse(build_tajwid_payload())

        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setenv("TAJWID_BACKUP_URL", "https://backup.example/tajwid.json")
    monkeypatch.setattr(tajwid_service, "urlopen", fake_urlopen)

    response = fetch_tajwid_text(112, 3, 4)

    assert response["text"] == "[rule[الله]] [rule[أحد]]"
    assert calls == [
        ("https://backup.example/tajwid.json", tajwid_service.TAJWID_TIMEOUT_SECONDS),
        (tajwid_service.TAJWID_API_BASE, tajwid_service.TAJWID_TIMEOUT_SECONDS),
    ]


def test_fetch_tajwid_text_rejects_ranges_with_missing_verses(monkeypatch):
    payload = build_tajwid_payload()
    payload["data"]["surahs"][0]["ayahs"] = [
        {"numberInSurah": 1, "text": "[rule[قل]]"},
        {"numberInSurah": 3, "text": "[rule[الله]]"},
    ]

    monkeypatch.setattr(tajwid_service, "urlopen", lambda _request_or_url, timeout: FakeUrlOpenResponse(payload))

    with pytest.raises(TajwidServiceError, match="plage"):
        fetch_tajwid_text(112, 1, 3)


def test_fetch_tajwid_text_raises_when_all_sources_fail(monkeypatch, tmp_path):
    missing_snapshot_path = tmp_path / "missing_tajwid_snapshot.json"

    def fake_urlopen(request_or_url, timeout):
        raise URLError(f"Unavailable: {resolve_request_url(request_or_url)}")

    monkeypatch.setenv("TAJWID_DATA_PATH", str(missing_snapshot_path))
    monkeypatch.setenv("TAJWID_BACKUP_URL", "https://backup.example/tajwid.json")
    monkeypatch.setattr(tajwid_service, "urlopen", fake_urlopen)

    with pytest.raises(TajwidServiceError, match="Impossible de joindre l'API tajwid"):
        fetch_tajwid_text(112, 1, 1)
