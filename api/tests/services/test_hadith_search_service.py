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
    response = HadithSearchService(index=index, client=client).search("retrouver un texte source", 2)
    assert [item.id for item in response.results] == ["3", "1"]
    assert response.results[0].translation == "Texte source\r\nintact"
    assert "score" not in response.results[0].model_dump()
    assert client.get_hadith.call_count == 2
    index.rank.assert_called_once_with("retrouver un texte source", 2)
    assert response.search_mode == "semantic"
    assert response.search_terms == []


@pytest.mark.parametrize("error", [HadithIndexError("missing index"), ImportError("sentence_transformers"), RuntimeError("model unavailable")])
def test_index_failure_is_isolated(error):
    index = Mock()
    index.rank.side_effect = error
    client = Mock()
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("retrouver un texte source")
    client.get_hadith.assert_not_called()


def test_source_down_is_not_reported_as_an_empty_search():
    index = Mock()
    index.rank.return_value = [("1", 0.9)]
    client = Mock()
    client.get_hadith.side_effect = HadeethEncError("timeout")
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("retrouver un texte source")


def test_removed_sources_return_no_usable_results():
    index = Mock()
    index.rank.return_value = [("1", 0.9)]
    client = Mock()
    client.get_hadith.side_effect = HadeethEncNotFound()
    assert HadithSearchService(index=index, client=client).search("retrouver un texte source").results == []


@pytest.mark.parametrize("query, searched", [
    ("Donnez moi hadith qui parle de ne pas se mettre en colère", "ne pas se mettre en colère"),
    ("Je ne cherche pas un hadith sur la colère", "Je ne cherche pas un hadith sur la colère"),
])
def test_cleans_only_search_input_while_preserving_the_users_query(query, searched):
    index = Mock()
    index.rank.return_value = [("3", 0.9)]
    client = Mock()
    client.get_hadith.return_value = payload("3")
    response = HadithSearchService(index=index, client=client).search(query)
    index.rank.assert_called_once_with(searched, 3)
    assert response.query == query
    assert response.results[0].translation == "Texte source\r\nintact"


@pytest.mark.parametrize("query", ["couronne", "hadith couronne", "les couronnes", "wifi"])
def test_missing_keywords_return_nothing_instead_of_unrelated_neighbours(query):
    index = Mock()
    index.source_documents.return_value = {
        "1": "Le conseil sur la colère", "2": "Un effort couronné de succès", "3": "L’importance du Coran",
    }
    index.rank.return_value = [("1", 0.86), ("2", 0.85), ("3", 0.84)]
    client = Mock()
    response = HadithSearchService(index=index, client=client).search(query)
    assert response.results == []
    assert response.search_mode == "keywords"
    index.rank.assert_not_called()
    client.get_hadith.assert_not_called()


@pytest.mark.parametrize("query", ["colère", "hadith colère", "Je cherche le hadith sur la colère"])
def test_filters_all_source_documents_before_ranking_and_keeps_the_original_query(query):
    index = Mock()
    index.source_documents.return_value = {"1": "Sans relation", "42": "Sur la colère", "43": "Explication : colères"}
    index.rank.return_value = [("43", 0.6), ("42", 0.5)]
    client = Mock()
    client.get_hadith.side_effect = lambda hid, _: {**payload(hid), "explanation": "La colère"}
    response = HadithSearchService(index=index, client=client).search(query, 2)
    index.rank.assert_called_once_with("colère" if query != "Je cherche le hadith sur la colère" else "la colère", 2,
                                       candidate_ids={"42", "43"})
    assert [result.id for result in response.results] == ["43", "42"]
    assert response.query == query
    assert response.search_terms == ["colère"]
    assert response.search_mode == "keywords"
    assert response.results[0].translation == "Texte source\r\nintact"


def test_keyword_result_is_rechecked_against_the_live_source():
    index = Mock()
    index.source_documents.return_value = {"1": "Une couronne", "2": "Une couronne"}
    index.rank.return_value = [("1", 0.9), ("2", 0.8)]
    client = Mock()
    client.get_hadith.side_effect = [payload("1"), {**payload("2"), "hadeeth": "Une couronne"}]
    response = HadithSearchService(index=index, client=client).search("couronne")
    assert [result.id for result in response.results] == ["2"]


def test_missing_source_documents_do_not_fall_back_to_unrelated_semantic_results():
    index = Mock()
    index.source_documents.side_effect = HadithIndexError("Upgrade metadata")
    client = Mock()
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("couronne")
    index.rank.assert_not_called()
    client.get_hadith.assert_not_called()


def test_keyword_source_failure_is_not_an_empty_search():
    index = Mock()
    index.source_documents.return_value = {"1": "Une couronne"}
    index.rank.return_value = [("1", 0.9)]
    client = Mock()
    client.get_hadith.side_effect = HadeethEncError("timeout")
    with pytest.raises(HadithSearchError, match=UNAVAILABLE_MESSAGE):
        HadithSearchService(index=index, client=client).search("couronne")


@pytest.mark.parametrize("query", ["hadith", "...", "les"])
def test_no_keyword_subject_does_not_return_arbitrary_results(query):
    index = Mock()
    index.source_documents.return_value = {"1": "Texte"}
    response = HadithSearchService(index=index, client=Mock()).search(query)
    assert response.results == []
    index.rank.assert_not_called()
