import wave
from pathlib import Path

import pytest

import evaluation.audio_benchmark.audio as audio_module
from evaluation.audio_benchmark.audio import generate_tone, write_pcm16_wav
from evaluation.audio_benchmark.secure_io import atomic_output_path, atomic_write_text


def test_atomic_private_write_is_0600_and_replaces_only_after_success(tmp_path):
    output_path = tmp_path / "private.json"

    with atomic_output_path(output_path, private=True) as temp_path:
        assert temp_path.stat().st_mode & 0o777 == 0o600
        temp_path.write_text("complete", encoding="utf-8")
        assert not output_path.exists()

    assert output_path.read_text(encoding="utf-8") == "complete"
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_private_write_keeps_previous_file_and_cleans_temp_on_error(tmp_path):
    output_path = tmp_path / "private.json"
    output_path.write_text("previous", encoding="utf-8")
    output_path.chmod(0o644)

    with pytest.raises(RuntimeError, match="boom"):
        with atomic_output_path(output_path, private=True) as temp_path:
            assert temp_path.stat().st_mode & 0o777 == 0o600
            temp_path.write_text("partial", encoding="utf-8")
            raise RuntimeError("boom")

    assert output_path.read_text(encoding="utf-8") == "previous"
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_private_wav_write_is_atomic_when_encoder_fails(monkeypatch, tmp_path):
    output_path = tmp_path / "private.wav"
    output_path.write_bytes(b"previous")
    output_path.chmod(0o644)
    observed_modes = []

    def fail_open(path, _mode):
        observed_modes.append(Path(path).stat().st_mode & 0o777)
        raise wave.Error("encoder failed")

    monkeypatch.setattr(audio_module.wave, "open", fail_open)

    with pytest.raises(wave.Error, match="encoder failed"):
        write_pcm16_wav(
            output_path,
            generate_tone(0.1, 8_000),
            8_000,
            private=True,
        )

    assert observed_modes == [0o600]
    assert output_path.read_bytes() == b"previous"
    assert output_path.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))


def test_atomic_write_text_uses_requested_private_mode(tmp_path):
    output_path = tmp_path / "report.json"

    atomic_write_text(output_path, "{}\n", private=True)

    assert output_path.stat().st_mode & 0o777 == 0o600
