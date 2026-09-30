import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.hadith_config import HadithConfig
from app.services import hadith_index as module
from scripts.build_hadith_index import build_search_text


def make_index(tmp_path, monkeypatch):
    config = HadithConfig("https://example.test", "fr", "test-model", tmp_path / "index.npz", tmp_path / "meta.json")
    np.savez_compressed(config.index_path, embeddings=np.array([[2, 0], [0, 4], [1, 1]], dtype=np.float32))
    meta = {"schema_version": 1, "model": "test-model", "language": "fr", "dimension": 2, "index_sha256": hashlib.sha256(config.index_path.read_bytes()).hexdigest(), "items": [{"hadeethenc_id": str(i), "language": "fr"} for i in (1, 2, 3)]}
    config.meta_path.write_text(json.dumps(meta))
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 2
    model.encode.return_value = [[0, 1]]
    loader = Mock(return_value=model)
    monkeypatch.setattr(module, "load_embedding_model", loader)
    return module.HadithIndex(config), model, loader


def test_cosine_ranking_normalizes_and_loads_only_once(tmp_path, monkeypatch):
    index, model, loader = make_index(tmp_path, monkeypatch)
    assert loader.call_count == 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        rankings = list(pool.map(lambda _: index.rank("requête", 3), range(4)))
    assert [i for i, _ in rankings[0]] == ["2", "3", "1"]
    assert rankings[0][0][1] == pytest.approx(1)
    assert loader.call_count == 1
    assert model.encode.call_args.args[0] == ["query: requête"]


def test_missing_index_does_not_attempt_model_load(tmp_path, monkeypatch):
    index, _, loader = make_index(tmp_path, monkeypatch)
    index.config.index_path.unlink()
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    loader.assert_not_called()


def test_model_failure_can_be_retried(tmp_path, monkeypatch):
    index, model, loader = make_index(tmp_path, monkeypatch)
    loader.side_effect = [RuntimeError("offline"), model]
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    assert index.rank("requête", 1)[0][0] == "2"


@pytest.mark.parametrize("field,value", [("model", "other"), ("language", "en"), ("dimension", 3), ("index_sha256", "wrong"), ("items", [])])
def test_incompatible_artifacts_fail_closed(tmp_path, monkeypatch, field, value):
    index, _, loader = make_index(tmp_path, monkeypatch)
    meta = json.loads(index.config.meta_path.read_text())
    meta[field] = value
    index.config.meta_path.write_text(json.dumps(meta))
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    loader.assert_not_called()


def test_search_passage_includes_context_without_mutating_source():
    payload = {"title": "Titre", "hadeeth": "Texte", "explanation": "Explication", "hints": ["Bénéfice"], "categories": ["10"]}
    assert build_search_text(payload, {"10": "Catégorie"}) == "passage: Titre\n\nTexte\n\nExplication\n\nBénéfice\n\nCatégorie"
    assert payload["hadeeth"] == "Texte"


@pytest.mark.parametrize("matrix", [np.array([[0, 0], [0, 1], [1, 1]]), np.array([[float("nan"), 0], [0, 1], [1, 1]]), np.ones((2, 2))])
def test_broken_vectors_are_rejected_before_loading_model(tmp_path, monkeypatch, matrix):
    index, _, loader = make_index(tmp_path, monkeypatch)
    np.savez_compressed(index.config.index_path, embeddings=matrix)
    meta = json.loads(index.config.meta_path.read_text())
    meta["index_sha256"] = hashlib.sha256(index.config.index_path.read_bytes()).hexdigest()
    index.config.meta_path.write_text(json.dumps(meta))
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    loader.assert_not_called()


def test_model_dimension_mismatch_does_not_publish_resources(tmp_path, monkeypatch):
    index, model, _ = make_index(tmp_path, monkeypatch)
    model.get_sentence_embedding_dimension.return_value = 384
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    assert index._resources is None
