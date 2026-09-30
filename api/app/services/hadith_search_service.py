"""Compose local retrieval and live official content, without generating text."""

from threading import Lock

from app.core.hadith_config import HadithConfig
from app.schemas.hadith import HadithSearchResponse
from app.services.hadeethenc_client import HadeethEncClient, HadeethEncError, HadeethEncNotFound, to_hadith_result
from app.services.hadith_index import HadithIndex

UNAVAILABLE_MESSAGE = "La recherche de hadiths est temporairement indisponible."


class HadithSearchError(Exception):
    pass


class HadithSearchService:
    def __init__(self, config=None, *, index=None, client=None):
        self.config = config or HadithConfig.from_env()
        self.index = index if index is not None else HadithIndex(self.config)
        self.client = client if client is not None else HadeethEncClient(self.config.base_url)

    def search(self, query: str, limit: int = 3) -> HadithSearchResponse:
        try:
            ranked = self.index.rank(query, limit)
        except Exception as exc:
            # Includes missing optional embedding dependencies; Quran stays available.
            raise HadithSearchError(UNAVAILABLE_MESSAGE) from exc
        results = []
        for hadith_id, _ in ranked:
            try:
                payload = self.client.get_hadith(hadith_id, self.config.language)
            except HadeethEncNotFound:
                # A source may have been removed since the last index build.
                continue
            except HadeethEncError as exc:
                raise HadithSearchError(UNAVAILABLE_MESSAGE) from exc
            results.append(to_hadith_result(payload, self.config.language))
        return HadithSearchResponse(query=query, results=results)


_service: HadithSearchService | None = None
_service_lock = Lock()


def get_hadith_search_service() -> HadithSearchService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = HadithSearchService()
    return _service
