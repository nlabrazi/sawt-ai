import hashlib
import json
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.hadith_config import HadithConfig
from app.services.hadeethenc_client import HadeethEncError
from app.services.hadith_index import load_search_documents
from scripts import build_hadith_index as builder


def make_config(tmp_path, strategy="original"):
    return HadithConfig(
        "https://example.test", "fr", "model", strategy,
        tmp_path / "index", tmp_path / "meta",
    )


def minimal_payload(hadith_id):
    return {
        "id": hadith_id,
        "title": f"Titre {hadith_id}",
        "hadeeth": f"Texte {hadith_id}",
        "hadeeth_ar": "نص",
        "explanation": "",
        "hints": [],
        "categories": [],
        "translations": ["fr"],
        "grade": None,
        "attribution": None,
    }


def test_download_resumes_and_refresh_replaces_source(tmp_path):
    config = make_config(tmp_path)
    client = Mock()
    client.list_categories.return_value = [{"id": "1", "title": "Catégorie"}]
    client.iter_hadith_ids.return_value = ["2", "1", "3"]
    client.get_hadith.side_effect = lambda hid, language: {"id": hid, "hadeeth": "Source " + hid}
    records, categories = builder.fetch_corpus(client, config, tmp_path / "cache", workers=1)
    assert [r["payload"]["id"] for r in records] == ["1", "2", "3"]
    assert categories == {"1": "Catégorie"}
    assert client.get_hadith.call_count == 3
    builder.fetch_corpus(client, config, tmp_path / "cache", workers=1)
    assert client.get_hadith.call_count == 3
    builder.fetch_corpus(client, config, tmp_path / "cache", workers=1, refresh=True)
    assert client.get_hadith.call_count == 6
    assert json.loads((tmp_path / "cache" / "1.json").read_text())["payload"]["hadeeth"] == "Source 1"


def test_incomplete_download_fails_and_preserves_previous_index(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    config.index_path.write_bytes(b"previous index")
    client = Mock()
    client.list_categories.return_value = [{"id": "1", "title": "Catégorie"}]
    client.iter_hadith_ids.return_value = ["1"]
    client.get_hadith.side_effect = HadeethEncError("offline")
    monkeypatch.setattr(builder.time, "sleep", lambda _: None)
    with pytest.raises(HadeethEncError):
        builder.fetch_corpus(client, config, tmp_path / "cache", workers=1)
    assert config.index_path.read_bytes() == b"previous index"
    assert client.get_hadith.call_count == 3


# ---------------------------------------------------------------------------
# _build_passages
# ---------------------------------------------------------------------------

def _make_fake_model(dim=4):
    """Return a mock sentence-transformer model with a minimal tokenizer.

    The tokenizer does not need to be accurate: build_documents() only uses it
    to estimate token counts for splitting. A simple char-count proxy is enough
    to trigger the multi-passage path without importing heavy dependencies.
    """
    class _FakeTokenizer:
        def __call__(self, text, **kwargs):
            # Approximate 1 token per 4 chars so realistic texts split correctly.
            n = max(1, len(text) // 4)
            return {
                "input_ids": list(range(n)),
                "offset_mapping": [(i * 4, (i + 1) * 4) for i in range(n)],
            }

    model = Mock()
    model.tokenizer = _FakeTokenizer()
    model.encode.side_effect = lambda texts, **kw: np.random.rand(len(texts), dim).astype(np.float32)
    return model


def test_original_strategy_produces_one_passage_per_hadith(tmp_path):
    config = make_config(tmp_path, strategy="original")
    records = [{"payload": minimal_payload("1")}, {"payload": minimal_payload("2")}]
    categories: dict = {}
    model = _make_fake_model()
    passages = builder._build_passages(records, categories, config.strategy, model)
    hadith_ids = [hid for hid, _ in passages]
    assert hadith_ids == ["1", "2"]


def test_multi_context_strategy_produces_multiple_passages_per_hadith(tmp_path):
    config = make_config(tmp_path, strategy="multi_context")
    payload = minimal_payload("1")
    payload["hints"] = ["Bénéfice 1", "Bénéfice 2"]
    payload["explanation"] = "Explication longue"
    records = [{"payload": payload}]
    categories = {"10": "Catégorie"}
    model = _make_fake_model()
    passages = builder._build_passages(records, categories, config.strategy, model)
    hadith_ids = [hid for hid, _ in passages]
    # All passages belong to hadith "1" and there are more than one.
    assert all(hid == "1" for hid in hadith_ids)
    assert len(passages) > 1


def test_multi_context_passages_start_with_passage_prefix(tmp_path):
    config = make_config(tmp_path, strategy="multi_context")
    records = [{"payload": minimal_payload("42")}]
    model = _make_fake_model()
    passages = builder._build_passages(records, {}, config.strategy, model)
    assert all(text.startswith("passage:") for _, text in passages)


# ---------------------------------------------------------------------------
# save_index — meta includes strategy
# ---------------------------------------------------------------------------

def test_save_index_stores_strategy_in_meta(tmp_path, monkeypatch):
    config = make_config(tmp_path, strategy="multi_context")
    records = [{"payload": minimal_payload("1"), "fetched_at": "2026-01-01T00:00:00+00:00"}]

    dim = 4
    fake_model = Mock()
    fake_model.tokenizer = Mock()
    # Simulate encode returning one normalised vector per passage.
    fake_model.encode.side_effect = lambda texts, **kw: (
        np.ones((len(texts), dim), dtype=np.float32) / np.sqrt(dim)
    )
    fake_model.__getitem__ = lambda self, idx: Mock(auto_model=Mock(config=Mock(_commit_hash="abc")))

    monkeypatch.setattr(builder, "_build_passages", lambda *a, **kw: [("1", "passage: texte")])

    builder.save_index(config, records, {}, fake_model, batch_size=1)

    meta = json.loads(config.meta_path.read_text())
    assert meta["strategy"] == "multi_context"
    assert meta["items"][0]["hadeethenc_id"] == "1"
    assert meta["schema_version"] == 2
    assert load_search_documents(config) == {"1": "Titre 1\n\nTexte 1"}


def legacy_index(tmp_path):
    config = make_config(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    records = [{"language": "fr", "base_url": config.base_url, "fetched_at": "2026-01-01",
                "payload": minimal_payload(hid)} for hid in ("1", "2")]
    for record in records:
        (cache / f"{record['payload']['id']}.json").write_text(json.dumps(record))
    with config.index_path.open("wb") as stream:
        np.savez_compressed(stream, embeddings=np.ones((3, 2)))
    meta = {"schema_version": 1, "model": "model", "language": "fr", "base_url": config.base_url, "dimension": 2,
            "index_sha256": hashlib.sha256(config.index_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            "items": [{"hadeethenc_id": hid, "language": "fr"} for hid in ("1", "1", "2")]}
    config.meta_path.write_text(json.dumps(meta))
    return config, cache


def test_upgrade_reuses_exact_snapshot_without_encoding_and_preserves_backup(tmp_path, monkeypatch):
    config, cache = legacy_index(tmp_path)
    previous_index = config.index_path.read_bytes()
    previous_meta = config.meta_path.read_bytes()
    loader = Mock(side_effect=AssertionError("No model needed for metadata upgrade"))
    monkeypatch.setattr(builder, "load_embedding_model", loader)
    builder.upgrade_search_documents(config, cache)
    assert config.index_path.read_bytes() == previous_index
    assert config.meta_path.with_suffix(".before-search-documents.json").read_bytes() == previous_meta
    assert load_search_documents(config) == {"1": "Titre 1\n\nTexte 1", "2": "Titre 2\n\nTexte 2"}
    builder.upgrade_search_documents(config, cache)
    assert config.meta_path.with_suffix(".before-search-documents.json").read_bytes() == previous_meta
    loader.assert_not_called()


@pytest.mark.parametrize("change", ["missing", "content", "language", "base_url", "id"])
def test_upgrade_rejects_incomplete_or_changed_cache_without_replacing_index(tmp_path, change):
    config, cache = legacy_index(tmp_path)
    previous_meta = config.meta_path.read_bytes()
    previous_index = config.index_path.read_bytes()
    path = cache / "1.json"
    if change == "missing":
        path.unlink()
    else:
        record = json.loads(path.read_text())
        if change == "content":
            record["payload"]["hadeeth"] = "Texte modifié"
        elif change == "id":
            record["payload"]["id"] = "3"
        else:
            record[change] = "different"
        path.write_text(json.dumps(record))
    with pytest.raises((ValueError, FileNotFoundError)):
        builder.upgrade_search_documents(config, cache)
    assert config.meta_path.read_bytes() == previous_meta
    assert config.index_path.read_bytes() == previous_index
    assert not config.meta_path.with_suffix(".before-search-documents.json").exists()


@pytest.mark.parametrize("change", ["checksum", "missing_id", "empty_text", "legacy"])
def test_search_documents_reject_corrupted_or_legacy_metadata(tmp_path, change):
    from app.services.hadith_index import HadithIndexError

    config, cache = legacy_index(tmp_path)
    if change != "legacy":
        builder.upgrade_search_documents(config, cache)
        meta = json.loads(config.meta_path.read_text())
        if change == "checksum":
            meta["search_documents"]["1"] = "Contenu altéré"
        elif change == "missing_id":
            del meta["search_documents"]["1"]
        else:
            meta["search_documents"]["1"] = ""
        config.meta_path.write_text(json.dumps(meta))
    with pytest.raises(HadithIndexError):
        load_search_documents(config)
