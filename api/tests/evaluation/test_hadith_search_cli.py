import json
from unittest.mock import Mock

import pytest

from app.core.hadith_config import HadithConfig
from app.services import hadeethenc_client, hadith_index
from scripts import hadith_experiment, search_hadith


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("variant, strategy", [("benchmark", "multi_context"), ("benchmark-original", "multi")])
def test_cli_uses_cleaned_subject_and_can_reproduce_raw_search(monkeypatch, capsys, raw, variant, strategy):
    query = "Donnez moi hadith qui parle de ne pas se mettre en colère"
    index = Mock()
    index.rank.return_value = [("4709", 0.8)]
    client = Mock()
    client.get_hadith.return_value = {"id": "4709", "title": "Titre témoin", "hadeeth": "Texte source inchangé", "hadeeth_ar": "نص"}
    factory = Mock(return_value=index)
    monkeypatch.setattr(hadith_experiment, "HadithExperiment", factory)
    monkeypatch.setattr(hadeethenc_client, "HadeethEncClient", lambda url: client)
    monkeypatch.setattr("sys.argv", ["search_hadith.py", "--variant", variant, "--json", *(["--raw-query"] if raw else []), query])
    search_hadith.main()
    searched = query if raw else "ne pas se mettre en colère"
    factory.assert_called_once_with(strategy=strategy)
    index.rank.assert_called_once_with(searched)
    output = json.loads(capsys.readouterr().out)
    assert output["query"] == query
    assert output["search_query"] == searched
    assert output["results"][0]["translation"] == "Texte source inchangé"
    assert output["search_mode"] == "benchmark"


@pytest.mark.parametrize("query", ["couronne", "hadith couronne"])
def test_default_cli_uses_the_same_keyword_filter_as_the_api(monkeypatch, capsys, query):
    index = Mock()
    index.config = HadithConfig.from_env()
    index.source_documents.return_value = {"1": "Un effort couronné de succès"}
    monkeypatch.setattr(hadith_index, "HadithIndex", lambda: index)
    client = Mock()
    monkeypatch.setattr(hadeethenc_client, "HadeethEncClient", lambda _: client)
    monkeypatch.setattr("sys.argv", ["search_hadith.py", "--json", query])
    search_hadith.main()
    output = json.loads(capsys.readouterr().out)
    assert output["search_mode"] == "keywords"
    assert output["search_terms"] == ["couronne"]
    assert output["results"] == []
    index.load.assert_not_called()
    index.rank.assert_not_called()
    client.get_hadith.assert_not_called()


def test_raw_scores_require_an_explicit_benchmark_variant(monkeypatch):
    monkeypatch.setattr("sys.argv", ["search_hadith.py", "--raw-query", "couronne"])
    with pytest.raises(SystemExit) as error:
        search_hadith.main()
    assert error.value.code == 2
