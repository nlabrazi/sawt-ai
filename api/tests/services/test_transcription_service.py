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


def test_transcribe_audio_can_preserve_recitation_when_vad_is_disabled(monkeypatch):
    model = FakeWhisperModel([SimpleNamespace(text=" الحمد لله رب العالمين ")])
    monkeypatch.setattr(transcription_service, "get_whisper_model", lambda: model)

    result = transcription_service.transcribe_audio(
        "/tmp/recitation.wav",
        language="ar",
        vad_filter=False,
    )

    assert result == [{"text": "الحمد لله رب العالمين"}]
    assert model.calls[0][1]["vad_filter"] is False
    assert model.calls[0][1]["language"] == "ar"
    assert "vad_parameters" not in model.calls[0][1]


def make_metadata():
    return transcription_service.TranscriptionMetadata(
        language="ar",
        language_probability=1.0,
        arabic_probability=1.0,
        language_probabilities=(("ar", 1.0),),
        duration_seconds=12.0,
        duration_after_vad_seconds=12.0,
        speech_duration_seconds=12.0,
        average_log_probability=-0.3,
        average_no_speech_probability=0.1,
        max_compression_ratio=1.4,
        max_temperature=0.0,
        segment_metrics=(),
    )


def test_language_screen_samples_beginning_middle_and_end():
    windows = transcription_service._distributed_audio_windows(
        list(range(100)),
        window_sample_count=30,
        max_windows=3,
    )

    assert tuple(window[0] for window in windows) == (0, 35, 70)
    assert all(len(window) == 30 for window in windows)


def test_language_screen_preserves_arabic_found_in_one_window(monkeypatch):
    class FakeLanguageModel:
        def __init__(self):
            self.results = iter([
                ("fr", 0.92, [("fr", 0.92), ("ar", 0.02)]),
                ("ar", 0.81, [("ar", 0.81), ("fr", 0.11)]),
                ("fr", 0.88, [("fr", 0.88), ("ar", 0.03)]),
            ])

        def detect_language(self, **_options):
            return next(self.results)

    monkeypatch.setattr(transcription_service, "AUDIO_SAMPLE_RATE", 1)
    monkeypatch.setattr(
        transcription_service,
        "LANGUAGE_SCREENING_WINDOW_SECONDS",
        30,
    )

    language, probability, probabilities = (
        transcription_service._detect_language_across_windows(
            list(range(100)),
            FakeLanguageModel(),
        )
    )

    assert language == "fr"
    assert probability == 0.92
    assert dict(probabilities)["ar"] == 0.81
    assert not transcription_service.is_confidently_non_arabic(
        language,
        probability,
        dict(probabilities)["ar"],
    )


def test_quran_transcription_skips_decode_when_vad_finds_no_speech(monkeypatch):
    screen = transcription_service.AudioLanguageScreen(
        language=None,
        language_probability=None,
        arabic_probability=None,
        language_probabilities=(),
        duration_seconds=6.0,
        speech_duration_seconds=0.0,
    )
    monkeypatch.setattr(transcription_service, "detect_audio_language", lambda _path: screen)
    monkeypatch.setattr(
        transcription_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: pytest.fail("text decoding must be skipped"),
    )

    result = transcription_service.transcribe_quran_audio("/tmp/silence.wav")

    assert result == []
    assert result.metadata.language is None
    assert result.metadata.speech_duration_seconds == 0


def test_quran_transcription_skips_confident_non_arabic_audio(monkeypatch):
    screen = transcription_service.AudioLanguageScreen(
        language="fr",
        language_probability=0.98,
        arabic_probability=0.0,
        language_probabilities=(("fr", 0.98), ("ar", 0.0)),
        duration_seconds=8.0,
        speech_duration_seconds=7.2,
    )
    monkeypatch.setattr(transcription_service, "detect_audio_language", lambda _path: screen)
    monkeypatch.setattr(
        transcription_service,
        "transcribe_audio",
        lambda *_args, **_kwargs: pytest.fail("text decoding must be skipped"),
    )

    result = transcription_service.transcribe_quran_audio("/tmp/french.wav")

    assert result == []
    assert result.metadata.language == "fr"
    assert result.metadata.arabic_probability == 0


def test_quran_transcription_forces_arabic_for_uncertain_audio(monkeypatch):
    screen = transcription_service.AudioLanguageScreen(
        language="fr",
        language_probability=0.61,
        arabic_probability=0.10,
        language_probabilities=(("fr", 0.61), ("ar", 0.10)),
        duration_seconds=48.0,
        speech_duration_seconds=31.0,
    )
    calls = []

    def fake_transcribe(audio_path, **options):
        calls.append((audio_path, options))
        return transcription_service.TranscriptionResult(
            [{"text": "بسم الله الرحمن الرحيم"}],
            make_metadata(),
        )

    monkeypatch.setattr(transcription_service, "detect_audio_language", lambda _path: screen)
    monkeypatch.setattr(transcription_service, "transcribe_audio", fake_transcribe)

    result = transcription_service.transcribe_quran_audio("/tmp/uncertain.wav")

    assert result == [{"text": "بسم الله الرحمن الرحيم"}]
    assert calls == [
        (
            "/tmp/uncertain.wav",
            {"language": "ar", "vad_filter": False},
        )
    ]
    assert result.metadata.language == "fr"
    assert result.metadata.language_probability == 0.61
    assert result.metadata.arabic_probability == 0.10
    assert result.metadata.duration_seconds == 48.0
    assert result.metadata.duration_after_vad_seconds == 31.0
    assert result.metadata.speech_duration_seconds == 31.0


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
