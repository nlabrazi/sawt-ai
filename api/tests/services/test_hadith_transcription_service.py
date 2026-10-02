from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services.hadith_transcription_service import HadithVoiceQueryError, transcribe_hadith_query
import app.services.transcription_service as transcription


@pytest.fixture
def model(monkeypatch):
    model = Mock()
    monkeypatch.setattr(transcription, "get_whisper_model", lambda: model)
    return model


def test_transcribes_french_with_vad_using_existing_whisper(model):
    model.transcribe.return_value = (iter([
        SimpleNamespace(text=" Trouve-moi les hadiths "),
        SimpleNamespace(text=" qui parlent du mariage. "),
    ]), SimpleNamespace())
    assert transcribe_hadith_query("/tmp/query.wav") == "Trouve-moi les hadiths qui parlent du mariage."
    options = model.transcribe.call_args.kwargs
    assert options["language"] == "fr"
    assert options["vad_filter"] is True


@pytest.mark.parametrize("texts", [[], [" "], ["ab"], ["..."], ["1234"], ["x" * 301]])
def test_rejects_unusable_queries_without_truncating_them(model, texts):
    model.transcribe.return_value = (iter(SimpleNamespace(text=text) for text in texts), SimpleNamespace())
    with pytest.raises(HadithVoiceQueryError):
        transcribe_hadith_query("/tmp/query.wav")
