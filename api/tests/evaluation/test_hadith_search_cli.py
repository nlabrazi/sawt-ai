import json
from unittest.mock import Mock

import pytest

from app.services import hadeethenc_client
from scripts import hadith_experiment, search_hadith


@pytest.mark.parametrize("raw", [False, True])
def test_cli_uses_cleaned_subject_and_can_reproduce_raw_search(monkeypatch, capsys, raw):
    query = "Donnez moi hadith qui parle de ne pas se mettre en colère"
    index = Mock()
    index.rank.return_value = [("4709", 0.8)]
    client = Mock()
    client.get_hadith.return_value = {"id": "4709", "title": "Titre témoin", "hadeeth": "Texte source inchangé", "hadeeth_ar": "نص"}
    monkeypatch.setattr(hadith_experiment, "HadithExperiment", lambda: index)
    monkeypatch.setattr(hadeethenc_client, "HadeethEncClient", lambda url: client)
    monkeypatch.setattr("sys.argv", ["search_hadith.py", "--variant", "benchmark", "--json", *(["--raw-query"] if raw else []), query])
    search_hadith.main()
    searched = query if raw else "ne pas se mettre en colère"
    index.rank.assert_called_once_with(searched)
    output = json.loads(capsys.readouterr().out)
    assert output["query"] == query
    assert output["search_query"] == searched
    assert output["results"][0]["translation"] == "Texte source inchangé"
