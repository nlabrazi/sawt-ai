import math

import pytest

from evaluation.audio_benchmark.audio import (
    AudioGenerationError,
    generate_synthetic_song,
    generate_synthetic_vocalization,
    generate_tone,
    generate_white_noise,
    mix_at_snr,
    read_pcm16_wav,
    write_pcm16_wav,
)


def rms(samples):
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def test_white_noise_is_reproducible_for_a_fixed_seed():
    first = generate_white_noise(0.1, 1_000, seed=42)
    second = generate_white_noise(0.1, 1_000, seed=42)
    other = generate_white_noise(0.1, 1_000, seed=43)

    assert first == second
    assert first != other
    assert rms(first) == pytest.approx(0.08, rel=1e-5)


def test_mix_at_snr_reaches_requested_ratio_without_clipping():
    signal = generate_tone(0.25, 4_000, frequency_hz=250, amplitude=0.1)
    noise = generate_white_noise(0.25, 4_000, seed=7, target_rms=0.05)

    mixed = mix_at_snr(signal, noise, snr_db=10)
    injected_noise = [mixed[index] - signal[index] for index in range(len(signal))]
    measured_snr = 20 * math.log10(rms(signal) / rms(injected_noise))

    assert measured_snr == pytest.approx(10, abs=0.02)


def test_mix_at_snr_rejects_silence_as_signal():
    with pytest.raises(AudioGenerationError, match="signal silencieux"):
        mix_at_snr([0.0, 0.0], [0.1, -0.1], snr_db=10)


def test_pcm16_wav_round_trip_is_mono_and_preserves_duration(tmp_path):
    path = tmp_path / "tone.wav"
    samples = generate_tone(0.2, 8_000, frequency_hz=440)

    write_pcm16_wav(path, samples, 8_000)
    decoded, sample_rate = read_pcm16_wav(path)

    assert sample_rate == 8_000
    assert len(decoded) == len(samples)
    assert decoded[100] == pytest.approx(samples[100], abs=1 / 32_767)


def test_synthetic_song_is_non_silent_and_has_exact_duration():
    samples = generate_synthetic_song(0.5, 8_000)

    assert len(samples) == 4_000
    assert rms(samples) > 0.01


def test_synthetic_vocalization_is_reproducible_without_real_voice():
    first = generate_synthetic_vocalization(0.5, 8_000, seed=42)
    second = generate_synthetic_vocalization(0.5, 8_000, seed=42)
    other = generate_synthetic_vocalization(0.5, 8_000, seed=43)

    assert len(first) == 4_000
    assert first == second
    assert first != other
    assert rms(first) == pytest.approx(0.10, rel=1e-5)
