from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import HTTPException

import app.routes.hadith as hadith_route
from app.schemas.hadith import HadithResult, HadithSearchResponse
from app.services.hadith_search_service import HadithSearchError


def make_result(hadith_id: str = "4709") -> HadithResult:
    return HadithResult(
        id=hadith_id,
        title="Ne te mets pas en colère !",
        arabic="لَا تَغْضَبْ",
        translation="Ne te mets pas en colère !",
        explanation=None,
        grade="Authentique",
        attribution=None,
        source_url=f"https://hadeethenc.com/fr/browse/hadith/{hadith_id}",
    )


def make_response(query: str = "la colère", *, hadith_id: str = "4709") -> HadithSearchResponse:
    return HadithSearchResponse(query=query, results=[make_result(hadith_id)])


def test_search_returns_service_response(monkeypatch):
    service = Mock()
    service.search.return_value = make_response()
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", lambda: service)

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(hadith_route, "run_in_threadpool", fake_run_in_threadpool)

    import asyncio
    request = Mock(query="la colère", limit=3)
    response = asyncio.run(hadith_route.search_hadith(request))

    service.search.assert_called_once_with("la colère", 3)
    assert response.query == "la colère"
    assert response.results[0].id == "4709"


def test_search_maps_service_error_to_503(monkeypatch):
    service = Mock()
    service.search.side_effect = HadithSearchError("indisponible")
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", lambda: service)

    async def fake_run_in_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(hadith_route, "run_in_threadpool", fake_run_in_threadpool)

    import asyncio
    request = Mock(query="la colère", limit=3)
    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(hadith_route.search_hadith(request))

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "indisponible"


def test_search_runs_service_in_threadpool(monkeypatch):
    """The route must offload CPU-bound inference to a thread pool."""
    threadpool_calls: list = []
    service = Mock()
    service.search.return_value = make_response()
    monkeypatch.setattr(hadith_route, "get_hadith_search_service", lambda: service)

    async def recording_run_in_threadpool(func, *args, **kwargs):
        threadpool_calls.append((func, args))
        return func(*args, **kwargs)

    monkeypatch.setattr(hadith_route, "run_in_threadpool", recording_run_in_threadpool)

    import asyncio
    request = Mock(query="test", limit=1)
    asyncio.run(hadith_route.search_hadith(request))

    assert len(threadpool_calls) == 1


@pytest.mark.parametrize("query,limit", [
    ("", 3),
    ("ab", 3),
    ("requête valide", 0),
    ("requête valide", 6),
])
def test_schema_rejects_invalid_requests(query, limit):
    from pydantic import ValidationError
    from app.schemas.hadith import HadithSearchRequest
    with pytest.raises(ValidationError):
        HadithSearchRequest(query=query, limit=limit)


def test_schema_accepts_valid_request():
    from app.schemas.hadith import HadithSearchRequest
    req = HadithSearchRequest(query="Je cherche le hadith sur la colère", limit=3)
    assert req.query == "Je cherche le hadith sur la colère"
    assert req.limit == 3
