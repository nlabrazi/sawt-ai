import io
import json
from urllib.error import HTTPError, URLError

import pytest

import app.services.hadeethenc_client as module
from app.services.hadeethenc_client import HadeethEncClient, HadeethEncError, HadeethEncNotFound, to_hadith_result


def source_payload(hadith_id="4709"):
    # Synthetic transport fixture, not source content.
    return {"id": hadith_id, "title": "Titre témoin", "hadeeth": "  Texte source\r\nintact. ", "hadeeth_ar": "نص للاختبار", "translations": ["fr", "ar"], "grade": "Degré source", "attribution": "Attribution source"}


def test_client_preserves_source_and_uses_configured_timeout(monkeypatch):
    payload = source_payload()
    captured = {}
    def open_response(request, timeout):
        captured.update(url=request.full_url, timeout=timeout)
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(module, "urlopen", open_response)
    result = to_hadith_result(HadeethEncClient("https://example.test/api", timeout=2).get_hadith("4709"))
    assert captured == {"url": "https://example.test/api/hadeeths/one/?id=4709&language=fr", "timeout": 2}
    assert result.translation == payload["hadeeth"]
    assert result.arabic == payload["hadeeth_ar"]
    assert result.explanation is None
    assert result.provider == "HadeethEnc"
    assert result.source_url == "https://hadeethenc.com/fr/browse/hadith/4709"


@pytest.mark.parametrize("error", [TimeoutError(), URLError("offline"), HTTPError("https://example.test", 503, "unavailable", {}, None)])
def test_network_errors_are_wrapped(monkeypatch, error):
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(module, "urlopen", fail)
    with pytest.raises(HadeethEncError):
        HadeethEncClient().get_hadith("4709")


def test_missing_hadith(monkeypatch):
    def fail(*args, **kwargs):
        raise HTTPError("https://example.test", 404, "missing", {}, None)
    monkeypatch.setattr(module, "urlopen", fail)
    with pytest.raises(HadeethEncNotFound):
        HadeethEncClient().get_hadith("99999999")


@pytest.mark.parametrize("raw", [b"not json", b"[]", b"{}", json.dumps({**source_payload(), "id": "99"}).encode(), json.dumps({**source_payload(), "hadeeth": " "}).encode(), json.dumps({**source_payload(), "translations": ["en"]}).encode()])
def test_invalid_source_is_not_presented(monkeypatch, raw):
    monkeypatch.setattr(module, "urlopen", lambda *args, **kwargs: io.BytesIO(raw))
    with pytest.raises(HadeethEncError):
        HadeethEncClient().get_hadith("4709")


def test_pagination_visits_roots_and_deduplicates(monkeypatch):
    client = HadeethEncClient()
    monkeypatch.setattr(client, "list_categories", lambda *args, **kwargs: [{"id": "1"}, {"id": "2"}])
    calls = []
    def get(endpoint, **params):
        calls.append((params["category_id"], params["page"]))
        return {"data": [{"id": str(params["page"])}], "meta": {"current_page": str(params["page"]), "last_page": 2}}
    monkeypatch.setattr(client, "_get", get)
    assert list(client.iter_hadith_ids()) == ["1", "2"]
    assert calls == [("1", 1), ("1", 2), ("2", 1), ("2", 2)]
