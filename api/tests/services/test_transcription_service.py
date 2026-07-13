from types import SimpleNamespace

import app.services.transcription_service as transcription_service


class FakeWhisperModel:
    def __init__(self, segments):
        self.segments = segments
        self.calls = []

    def transcribe(self, audio_path, **options):
        self.calls.append((audio_path, options))
        return iter(self.segments), SimpleNamespace()


def test_transcribe_audio_uses_arabic_vad_and_independent_segments(monkeypatch):
    model = FakeWhisperModel([SimpleNamespace(text=" قل هو الله ")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert result == [{"text": "قل هو الله"}]
    assert model.calls == [
        (
            "/tmp/recitation.wav",
            {
                "language": "ar",
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


def test_transcribe_audio_discards_empty_segments(monkeypatch):
    model = FakeWhisperModel([
        SimpleNamespace(text="   "),
        SimpleNamespace(text="\n"),
        SimpleNamespace(text="قل اعوذ برب الفلق"),
    ])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio("/tmp/recitation.wav")

    assert result == [{"text": "قل اعوذ برب الفلق"}]
