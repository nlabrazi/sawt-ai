from unittest.mock import Mock

import pytest

from app.services.hadeethenc_client import HadeethEncError, HadeethEncNotFound
from app.services.hadith_index import HadithIndexError
from app.services.hadith_search_service import HadithSearchError, HadithSearchService, UNAVAILABLE_MESSAGE


def payload(hadith_id):
    return {"id": hadith_id, "title": "Titre témoin", "hadeeth": "Texte source\r\nintact", "hadeeth_ar": "نص للاختبار"}


def test_fetches_live_source_in_ranked_order_without_exposing_scores():
    index = Mock()
    index.rank.return_value = [("3", 0.9), ("1", 0.8)]
    client = Mock()
    client.get_hadith.side_effect = lambda hadith_id, language: payload(hadith_id)
    response = HadithSearchService(index=index, client=client).search("requête", 2)
    assert [item.id for item in response.results] == ["3", "1"]
    assert response.results[0].translation == "Texte source\r\nintact"
    assert "score" not in response.results[0].model_dump()
    assert client.get_hadith.call_count == 2
    index.rank.assert_called_once_with("requête", 2)


@pytest.mark.parametrize("error", [HadithIndexError("missing index"), ImportError("sentence_transformers"), RuntimeError("model unavailable")])
def test_index_failure_is_isolated(error):
    index = Mock()
    index.rank.side_effect = error
    client = Mock()
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("requête")
    client.get_hadith.assert_not_called()


def test_source_down_is_not_reported_as_an_empty_search():
    index = Mock()
    index.rank.return_value = [("1", 0.9)]
    client = Mock()
    client.get_hadith.side_effect = HadeethEncError("timeout")
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("requête")


def test_removed_sources_return_no_usable_results():
    index = Mock()
    index.rank.return_value = [("1", 0.9)]
    client = Mock()
    client.get_hadith.side_effect = HadeethEncNotFound()
    assert HadithSearchService(index=index, client=client).search("requête").results == []
