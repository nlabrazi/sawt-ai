"""Exercise the HTTP contract without loading Quran models or calling the source."""

from threading import get_ident
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.routes.hadith as hadith_route
from app.schemas.hadith import HadithResult, HadithSearchResponse
from app.services.hadith_search_service import HadithSearchError


@pytest.fixture
def service(monkeypatch):
    service = Mock()
    service.search.return_value = HadithSearchResponse(
        query="la colère",
        results=[HadithResult(
            id="4709", title="Ne te mets pas colère !", arabic="لَا تَغْضَبْ",
            translation="Ne te mets pas colère !",
            source_url="https://hadeethenc.com/fr/browse/hadith/4709",
        )],
    )
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", lambda: service)
    return service


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(hadith_route.router)
    with TestClient(app) as client:
        yield client


def test_search_returns_official_fields_and_default_limit(client, service):
    response = client.post("/hadith/search", json={"query": "  la colère  "})
    assert response.status_code == 200
    service.search.assert_called_once_with("la colère", 3)
    payload = response.json()
    assert payload["query"] == "la colère"
    result = payload["results"][0]
    assert result["id"] == "4709"
    assert result["arabic"] == "لَا تَغْضَبْ"
    assert result["translation"] == "Ne te mets pas colère !"
    assert result["provider"] == "HadeethEnc"
    assert result["explanation"] is None
    assert "score" not in result


@pytest.mark.parametrize("limit", [1, 5])
def test_search_accepts_limit_bounds(client, service, limit):
    assert client.post("/hadith/search", json={"query": "la colère", "limit": limit}).status_code == 200
    service.search.assert_called_once_with("la colère", limit)


@pytest.mark.parametrize("body", [
    {}, {"query": ""}, {"query": "  "}, {"query": "ab"},
    {"query": "x" * 301}, {"query": None},
    {"query": "la colère", "limit": 0},
    {"query": "la colère", "limit": 6},
    {"query": "la colère", "limit": "3"},
    {"query": "la colère", "limit": True},
])
def test_search_rejects_invalid_requests_before_calling_service(client, service, body):
    assert client.post("/hadith/search", json=body).status_code == 422
    service.search.assert_not_called()


def test_search_maps_unavailability_to_503(client, service):
    service.search.side_effect = HadithSearchError("Recherche indisponible")
    response = client.post("/hadith/search", json={"query": "la colère"})
    assert response.status_code == 503
    assert response.json() == {"detail": "Recherche indisponible"}


def test_missing_index_logs_technical_cause_but_keeps_public_503_generic(monkeypatch, tmp_path, capsys):
    import json

    from app.core.hadith_config import HadithConfig
    from app.main import app
    from app.services.hadith_search_service import HadithSearchService, UNAVAILABLE_MESSAGE

    config = HadithConfig(
        base_url="https://example.test", language="fr", model_name="test-model",
        strategy="multi_context", index_path=tmp_path / "index.npz",
        meta_path=tmp_path / "meta.json",
    )
    service = HadithSearchService(config=config, client=Mock())
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", lambda: service)
    # Do not enter the client's context: Quran startup resources are unrelated.
    response = TestClient(app).post("/hadith/search", json={"query": "colère"})

    assert response.status_code == 503
    assert response.json() == {"detail": UNAVAILABLE_MESSAGE}
    log = json.loads(capsys.readouterr().err)
    assert log["route"] == "/hadith/search"
    assert log["errorType"] == "HadithSearchError"
    assert log["errorCauses"][0]["errorType"] == "HadithIndexError"
    assert log["errorCauses"][-1]["errorType"] == "FileNotFoundError"
    assert str(config.meta_path) in log["errorCauses"][-1]["error"]


def test_search_returns_an_empty_list_without_a_technical_error(client, service):
    service.search.return_value = HadithSearchResponse(query="la colère", results=[])
    response = client.post("/hadith/search", json={"query": "la colère"})
    assert response.status_code == 200
    assert response.json() == {"query": "la colère", "results": [], "search_mode": "semantic", "search_terms": []}


@pytest.mark.parametrize("query", ["couronne", "hadith couronne"])
def test_http_keyword_search_rejects_unrelated_semantic_candidates(client, service, query):
    from app.services.hadith_search_service import HadithSearchService

    index = Mock()
    index.source_documents.return_value = {"1": "Un effort couronné de succès", "2": "Conseil sur la colère"}
    source = Mock()
    service.search.side_effect = HadithSearchService(index=index, client=source).search
    response = client.post("/hadith/search", json={"query": query})
    assert response.status_code == 200
    assert response.json() == {"query": query, "results": [], "search_mode": "keywords", "search_terms": ["couronne"]}
    index.rank.assert_not_called()
    source.get_hadith.assert_not_called()


def test_search_offloads_inference_from_the_event_loop(client, service, monkeypatch):
    event_loop_threads = []
    worker_threads = []
    original = hadith_route.run_in_threadpool

    async def record_event_loop(func, *args, **kwargs):
        event_loop_threads.append(get_ident())
        return await original(func, *args, **kwargs)

    def search(query, limit):
        worker_threads.append(get_ident())
        return HadithSearchResponse(query=query, results=[])

    monkeypatch.setattr(hadith_route, "run_in_threadpool", record_event_loop)
    service.search.side_effect = search
    assert client.post("/hadith/search", json={"query": "la colère"}).status_code == 200
    assert len(event_loop_threads) == len(worker_threads) == 1
    assert worker_threads[0] != event_loop_threads[0]
