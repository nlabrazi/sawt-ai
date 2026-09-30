import re

from app.services.hadith_documents import PASSAGE_PREFIX, build_documents, inspect_document, join_document
from scripts.build_hadith_index import build_search_text


class WordTokenizer:
    def __call__(self, text, *, add_special_tokens=True, return_offsets_mapping=False, truncation=False, max_length=512):
        offsets = [match.span() for match in re.finditer(r"\S+", text)]
        if add_special_tokens:
            offsets = [(0, 0), *offsets, (0, 0)]
        if truncation and len(offsets) > max_length:
            offsets = offsets[:max_length - 1] + [(0, 0)] if add_special_tokens else offsets[:max_length]
        result = {"input_ids": list(range(len(offsets)))}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


def source():
    return {"id": "1", "title": "Titre court", "hadeeth": "Texte source. " * 40, "explanation": "Explication source. " * 40, "hints": ["Enseignement important", "Autre bénéfice"], "categories": ["10"]}


def test_original_document_is_exactly_the_current_builder_text():
    payload = source()
    document = build_documents(payload, {"10": "Catégorie"}, "original")[0]
    assert document.text == build_search_text(payload, {"10": "Catégorie"})
    assert document.text.startswith(PASSAGE_PREFIX)


def test_reordering_preserves_early_semantic_fields_but_exposes_remaining_truncation():
    original = build_documents(source(), {"10": "Catégorie"}, "original")[0]
    reordered = build_documents(source(), {"10": "Catégorie"}, "semantic_first")[0]
    original_audit = inspect_document(original, WordTokenizer(), 32)
    audit = inspect_document(reordered, WordTokenizer(), 32)
    assert all(s["status"] == "lost" for s in original_audit["sections"] if s["section"] in {"hints", "categories"})
    assert all(s["status"] == "retained" for s in audit["sections"] if s["section"] in {"hints", "categories"})
    assert audit["truncated"] is True


def test_multiple_passages_preserve_all_characters_without_truncation():
    payload = source()
    categories = {"10": "Catégorie"}
    docs = build_documents(payload, categories, "multi", WordTokenizer(), 32)
    assert len(docs) > 3
    assert all(len(WordTokenizer()(doc.text)["input_ids"]) <= 32 for doc in docs)
    for kind, order in (("title_categories", ("title", "categories")), ("hints", ("hints",)), ("hadith_explanation", ("hadith", "explanation"))):
        reconstructed = "".join(doc.text[len(PASSAGE_PREFIX):] for doc in docs if doc.kind.split(":")[0] == kind)
        expected = join_document(payload, categories, order, kind).text[len(PASSAGE_PREFIX):]
        assert reconstructed == expected
    assert payload == source()


def test_missing_optional_sections_do_not_create_empty_embeddings():
    payload = {"id": "1", "title": "Titre", "hadeeth": "Texte"}
    docs = build_documents(payload, {}, "multi", WordTokenizer())
    assert [doc.kind for doc in docs] == ["title_categories", "hadith_explanation"]
