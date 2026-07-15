import logging
from types import SimpleNamespace

import pytest

import app.services.transcription_service as transcription_service


class FakeWhisperModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio_path, **options):
        self.calls.append((audio_path, options))
        return iter(self.segments), SimpleNamespace()


def test_transcribe_audio_auto_detects_language_and_uses_vad(monkeypatch):
    model = FakeWhisperModel([SimpleNamespace(text=" قل هو الله ")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert result == [{"text": "قل هو الله"}]
    assert model.calls == [
        (
            "/tmp/recitation.wav",
            {
                "beam_size": 5,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": {
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 400,
                },
            },
        ),
    ]


def test_transcribe_audio_preserves_language_and_decode_metrics(monkeypatch):
    model = FakeWhisperModel([
        SimpleNamespace(
            text=" قل هو الله ",
            start=0.5,
            end=2.5,
            avg_logprob=-0.3,
            no_speech_prob=0.1,
            compression_ratio=1.4,
            temperature=0.0,
        ),
        SimpleNamespace(
            text=" احد ",
            start=2.5,
            end=3.5,
            avg_logprob=-0.6,
            no_speech_prob=0.2,
            compression_ratio=1.8,
            temperature=0.2,
        ),
    ])
    model.transcribe = lambda audio_path, **options: (
        iter(model.segments),
        SimpleNamespace(
            language="ar",
            language_probability=0.96,
            all_language_probs=[("ar", 0.96), ("ur", 0.02)],
            duration=4.0,
            duration_after_vad=3.4,
        ),
    )
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert result == [{"text": "قل هو الله"}, {"text": "احد"}]
    assert result.segments is result
    assert result.metadata.language == "ar"
    assert result.metadata.language_probability == 0.96
    assert result.metadata.arabic_probability == 0.96
    assert result.metadata.language_probabilities == (("ar", 0.96), ("ur", 0.02))
    assert result.metadata.duration_seconds == 4.0
    assert result.metadata.duration_after_vad_seconds == 3.4
    assert result.metadata.speech_duration_seconds == 3.0
    assert result.metadata.average_log_probability == pytest.approx(-0.4)
    assert result.metadata.average_no_speech_probability == pytest.approx(0.4 / 3)
    assert result.metadata.max_compression_ratio == 1.8
    assert result.metadata.max_temperature == 0.2
    assert len(result.metadata.segment_metrics) == 2


def test_transcribe_audio_does_not_log_user_transcription(monkeypatch, caplog):
    private_text = "texte privé à ne jamais journaliser"
    model = FakeWhisperModel([SimpleNamespace(text=private_text)])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    with caplog.at_level(logging.INFO, logger=transcription_service.__name__):
        transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert private_text not in caplog.text
    assert "segments=1" in caplog.text


def test_transcribe_audio_discards_empty_segments(monkeypatch):
    model = FakeWhisperModel([
        SimpleNamespace(text="   "),
        SimpleNamespace(text="\n"),
        SimpleNamespace(text="قل اعوذ برب الفلق"),
    ])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert result == [{"text": "قل اعوذ برب الفلق"}]


def test_transcribe_audio_limits_analysis_to_requested_clip(monkeypatch):
    model = FakeWhisperModel([SimpleNamespace(text="قل هو الله")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    transcription_service.transcribe_audio(
        "/tmp/recitation.wav",
        clip_end_seconds=5,
    )

    assert model.calls[0][1]["clip_timestamps"] == [0, 5]
