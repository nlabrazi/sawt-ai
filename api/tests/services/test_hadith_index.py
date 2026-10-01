import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.hadith_config import HadithConfig
from app.services import hadith_index as module
from app.services.hadith_index import _aggregate_scores
from scripts.build_hadith_index import build_search_text


def make_config(tmp_path, *, model="test-model"):
    return HadithConfig(
        base_url="https://example.test",
        language="fr",
        model_name=model,
        strategy="multi_context",
        index_path=tmp_path / "index.npz",
        meta_path=tmp_path / "meta.json",
    )


def make_index(tmp_path, monkeypatch, *, ids=("1", "2", "3"), matrix=None):
    config = make_config(tmp_path)
    if matrix is None:
        matrix = np.array([[2, 0], [0, 4], [1, 1]], dtype=np.float32)
    np.savez_compressed(config.index_path, embeddings=matrix)
    meta = {
        "schema_version": 1,
        "model": "test-model",
        "language": "fr",
        "dimension": matrix.shape[1],
        "index_sha256": hashlib.sha256(config.index_path.read_bytes()).hexdigest(),
        "items": [{"hadeethenc_id": i, "language": "fr"} for i in ids],
    }
    config.meta_path.write_text(json.dumps(meta))
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = matrix.shape[1]
    model.encode.return_value = [[0, 1]]
    loader = Mock(return_value=model)
    monkeypatch.setattr(module, "load_embedding_model", loader)
    return module.HadithIndex(config), model, loader


# ---------------------------------------------------------------------------
# Single-embedding index (backward compat)
# ---------------------------------------------------------------------------

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


@pytest.mark.parametrize("field,value", [
    ("model", "other"),
    ("language", "en"),
    ("dimension", 3),
    ("index_sha256", "wrong"),
    ("items", []),
])
def test_incompatible_artifacts_fail_closed(tmp_path, monkeypatch, field, value):
    index, _, loader = make_index(tmp_path, monkeypatch)
    meta = json.loads(index.config.meta_path.read_text())
    meta[field] = value
    index.config.meta_path.write_text(json.dumps(meta))
    with pytest.raises(module.HadithIndexError):
        index.rank("requête")
    loader.assert_not_called()


def test_search_passage_includes_context_without_mutating_source():
    payload = {
        "title": "Titre",
        "hadeeth": "Texte",
        "explanation": "Explication",
        "hints": ["Bénéfice"],
        "categories": ["10"],
    }
    assert build_search_text(payload, {"10": "Catégorie"}) == (
        "passage: Titre\n\nTexte\n\nExplication\n\nBénéfice\n\nCatégorie"
    )
    assert payload["hadeeth"] == "Texte"


@pytest.mark.parametrize("matrix", [
    np.array([[0, 0], [0, 1], [1, 1]]),
    np.array([[float("nan"), 0], [0, 1], [1, 1]]),
    np.ones((2, 2)),
])
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


# ---------------------------------------------------------------------------
# Multi-embedding index — duplicate IDs allowed
# ---------------------------------------------------------------------------

def test_multi_embedding_index_allows_duplicate_ids(tmp_path, monkeypatch):
    # Hadith "1" has two passages; "2" has one. Query vector aligns with passage 2 of "1".
    matrix = np.array([[1, 0], [0, 1], [0.5, 0.5]], dtype=np.float32)  # passages: 1a, 1b, 2
    index, model, _ = make_index(tmp_path, monkeypatch, ids=("1", "1", "2"), matrix=matrix)
    model.encode.return_value = [[0, 1]]  # query aligns with passage "1b"
    results = index.rank("requête", 2)
    # "1" wins via its best passage (1b, score≈1), "2" is second
    assert results[0][0] == "1"
    assert results[1][0] == "2"
    # Each hadith ID appears at most once
    assert len({hid for hid, _ in results}) == len(results)


def test_aggregate_scores_deduplicates_by_max():
    ids = ["A", "B", "A", "C"]
    scores = np.array([0.5, 0.9, 0.8, 0.3])
    result = _aggregate_scores(ids, scores, limit=3)
    result_dict = dict(result)
    assert result_dict["A"] == pytest.approx(0.8)  # max(0.5, 0.8)
    assert result_dict["B"] == pytest.approx(0.9)
    assert result_dict["C"] == pytest.approx(0.3)
    assert [hid for hid, _ in result] == ["B", "A", "C"]


def test_aggregate_scores_respects_limit():
    ids = ["A", "B", "C", "D"]
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    result = _aggregate_scores(ids, scores, limit=2)
    assert len(result) == 2
    assert result[0][0] == "A"


def test_aggregate_scores_on_unique_ids_matches_plain_argsort():
    """_aggregate_scores must be a drop-in replacement for argsort on unique IDs."""
    ids = ["10", "20", "30"]
    scores = np.array([0.6, 0.9, 0.75])
    result = _aggregate_scores(ids, scores, limit=3)
    assert [hid for hid, _ in result] == ["20", "30", "10"]
    assert [s for _, s in result] == pytest.approx([0.9, 0.75, 0.6])
