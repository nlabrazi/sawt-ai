import pytest

from app.services.hadith_lexical import matches_keywords, prepare_search_query, source_search_documents


@pytest.mark.parametrize("query, terms", [
    ("couronne", ("couronne",)),
    ("hadith couronne", ("couronne",)),
    ("Les hadiths sur la colère", ("colère",)),
    ("Je cherche le hadith sur la colère", ("colère",)),
    ("COURONNES", ("couronnes",)),
    ("les intentions", ("intentions",)),
    ("l’intention", ("intention",)),
    ("hadith", ()),
    ("...", ()),
])
def test_short_queries_extract_the_subject(query, terms):
    assert prepare_search_query(query).terms == terms


@pytest.mark.parametrize("query", [
    "Ne pas se mettre en colère", "Ne vole pas", "sans colère",
    "Je ne cherche pas un hadith sur la colère",
    "Les actes dépendent de leurs intentions",
])
def test_sentences_keep_semantic_retrieval_and_negation(query):
    assert prepare_search_query(query).terms is None
    assert prepare_search_query(query).text == query


@pytest.mark.parametrize("text, matched", [
    ("Une couronne de lumière.", True),
    ("Des COURONNES !", True),
    ("Couronné de succès", False),
    ("Le couronnement", False),
    ("Une couronnette", False),
    ("Des conseils sur la colère", False),
])
def test_whole_words_and_simple_plurals_preserve_accents(text, matched):
    assert matches_keywords(text, ("couronne",)) is matched


def test_requires_every_keyword_and_normalizes_unicode():
    assert matches_keywords("Une couronne et une lumière", ("couronne", "lumière"))
    assert not matches_keywords("Une couronne", ("couronne", "lumière"))
    assert matches_keywords("cole\u0300re", ("colère",))
    assert matches_keywords("une intention", ("intentions",))
    assert not matches_keywords("Texte", ())


def test_search_evidence_includes_only_fields_the_reader_can_consult():
    payload = {"id": "1", "title": "Titre", "hadeeth": "Texte", "explanation": "Explication",
               "hints": ["couronne"], "categories": ["couronne"], "hadeeth_ar": "نص"}
    documents = source_search_documents([{"payload": payload}])
    assert documents == {"1": "Titre\n\nTexte\n\nExplication"}
    assert not matches_keywords(documents["1"], ("couronne",))
    assert payload["hints"] == ["couronne"]


@pytest.mark.parametrize("records", [
    [{"payload": {"id": "1", "title": "Titre", "hadeeth": ""}}],
    [{"payload": {"id": "1", "title": "Titre", "hadeeth": "Texte", "explanation": 123}}],
    [{"payload": {"id": "../1", "title": "Titre", "hadeeth": "Texte"}}],
    [{"payload": {"id": "1", "title": "Titre", "hadeeth": "Texte"}}] * 2,
])
def test_invalid_source_evidence_is_rejected(records):
    with pytest.raises(ValueError):
        source_search_documents(records)
