import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest

import app.services.audio_enhancement_service as audio_enhancement_service


def _generate_synthetic_audio(duration_seconds: float = 1.0, sr: int = 16000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.linspace(0, duration_seconds, int(sr * duration_seconds), endpoint=False)
    voice = 0.5 * np.sin(2 * np.pi * 350 * t).astype(np.float32)
    rumble = 0.4 * np.sin(2 * np.pi * 35 * t).astype(np.float32)
    noise = 0.05 * np.random.normal(0, 1, len(t)).astype(np.float32)
    noisy = voice + rumble + noise
    return noisy, voice, rumble


def test_enhance_audio_array_reduces_low_frequency_noise_and_preserves_voice():
    noisy, voice, rumble = _generate_synthetic_audio(1.0)
    enhanced = audio_enhancement_service.enhance_audio_array(noisy)

    assert enhanced.shape == noisy.shape
    assert enhanced.dtype == np.float32
    assert np.all(enhanced >= -1.0) and np.all(enhanced <= 1.0)

    freqs = np.fft.rfftfreq(len(noisy), d=1.0 / 16000)
    spec_noisy = np.abs(np.fft.rfft(noisy))
    spec_enhanced = np.abs(np.fft.rfft(enhanced))

    # Le grondement basse fréquence (< 50 Hz) doit être fortement atténué
    rumble_before = np.sum(spec_noisy[freqs < 50])
    rumble_after = np.sum(spec_enhanced[freqs < 50])
    assert rumble_after < rumble_before * 0.4

    # La fréquence vocale (350 Hz) doit rester clairement audible
    voice_mask = (freqs >= 330) & (freqs <= 370)
    voice_power_after = np.sum(spec_enhanced[voice_mask])
    assert voice_power_after > 50.0


def test_enhance_audio_array_handles_empty_and_short_signals():
    empty = np.array([], dtype=np.float32)
    assert len(audio_enhancement_service.enhance_audio_array(empty)) == 0

    short = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    result = audio_enhancement_service.enhance_audio_array(short)
    assert np.array_equal(result, short)


def test_enhance_audio_array_respects_disabled_flag(monkeypatch):
    monkeypatch.setattr(audio_enhancement_service, "AUDIO_ENHANCEMENT_ENABLED", False)

    noisy, _, _ = _generate_synthetic_audio(0.5)
    result = audio_enhancement_service.enhance_audio_array(noisy)
    assert np.array_equal(result, noisy)


def test_enhance_audio_file_reads_and_writes_cleaned_wav():
    noisy, _, _ = _generate_synthetic_audio(0.5)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as input_tmp:
        input_path = Path(input_tmp.name)

    try:
        audio_enhancement_service.write_pcm16_wav(input_path, noisy)
        enhanced_path, was_enhanced = audio_enhancement_service.enhance_audio_file(input_path)

        assert was_enhanced is True
        assert enhanced_path.exists()
        assert enhanced_path != input_path

        # Vérifier que le fichier produit est un WAV lisible à 16 kHz
        with wave.open(str(enhanced_path), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() == len(noisy)

        enhanced_path.unlink(missing_ok=True)
    finally:
        input_path.unlink(missing_ok=True)


def test_enhance_audio_file_handles_corrupt_file_gracefully():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as corrupt_tmp:
        corrupt_tmp.write(b"not an audio file")
        corrupt_path = Path(corrupt_tmp.name)

    try:
        path, was_enhanced = audio_enhancement_service.enhance_audio_file(corrupt_path)
        assert was_enhanced is False
        assert path == corrupt_path
    finally:
        corrupt_path.unlink(missing_ok=True)
