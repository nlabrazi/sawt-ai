import json
from unittest.mock import Mock

import pytest

from app.core.hadith_config import HadithConfig
from app.services.hadeethenc_client import HadeethEncError
from scripts import build_hadith_index as builder


def test_download_resumes_and_refresh_replaces_source(tmp_path):
    config = HadithConfig("https://example.test", "fr", "model", "original", tmp_path / "index", tmp_path / "meta")
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
    config = HadithConfig("https://example.test", "fr", "model", "original", tmp_path / "index", tmp_path / "meta")
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
