from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.services import audio_metadata_service as service


def mock_decoder(monkeypatch, *, duration=None, frames=(), audio=True):
    stream = SimpleNamespace(duration=duration, time_base=Fraction(1, 48000))
    container = MagicMock()
    container.__enter__.return_value = container
    container.streams.audio = [stream] if audio else []
    container.decode.return_value = iter(frames)
    decoder = SimpleNamespace(open=MagicMock(return_value=container))
    monkeypatch.setattr(service, "_import_runtime_dependencies", lambda: decoder)
    return decoder, container, stream


def test_reads_audio_stream_duration_and_closes_container(monkeypatch):
    decoder, container, _ = mock_decoder(monkeypatch, duration=96000)

    assert service.get_audio_duration_seconds(Path("recitation.m4a")) == 2.0
    decoder.open.assert_called_once_with("recitation.m4a")
    container.decode.assert_not_called()
    container.__exit__.assert_called_once()


@pytest.mark.parametrize("duration", [None, 0, -1])
def test_counts_samples_when_recording_has_no_valid_duration(monkeypatch, duration):
    _, container, stream = mock_decoder(
        monkeypatch,
        duration=duration,
        frames=[
            SimpleNamespace(samples=48000, sample_rate=48000),
            SimpleNamespace(samples=24000, sample_rate=48000),
        ],
    )

    assert service.get_audio_duration_seconds("microphone.webm") == 1.5
    container.decode.assert_called_once_with(stream)
    container.__exit__.assert_called_once()


@pytest.mark.parametrize("audio,frames", [
    (False, []),
    (True, []),
    (True, [SimpleNamespace(samples=48000, sample_rate=0)]),
])
def test_rejects_missing_or_unreadable_audio(monkeypatch, audio, frames):
    _, container, _ = mock_decoder(monkeypatch, audio=audio, frames=frames)

    with pytest.raises(service.AudioMetadataError, match="Impossible de lire la durée"):
        service.get_audio_duration_seconds("invalid.webm")
    container.__exit__.assert_called_once()


def test_preserves_decoder_error_for_diagnosis(monkeypatch):
    decoder, _, _ = mock_decoder(monkeypatch)
    decoder.open.side_effect = ValueError("Invalid container")

    with pytest.raises(service.AudioMetadataError) as failure:
        service.get_audio_duration_seconds("invalid.m4a")
    assert failure.value.__cause__ is decoder.open.side_effect
