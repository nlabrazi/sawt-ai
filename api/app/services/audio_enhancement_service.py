# ROLE
# ----
# Isole la voix humaine et atténue les bruits parasites (véhicules, cris, bruits domestiques).
# Fournit une passe de Speech Enhancement en amont de Whisper pour fiabiliser la détection coranique.

from __future__ import annotations

import logging
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

AUDIO_ENHANCEMENT_ENABLED = os.getenv("AUDIO_ENHANCEMENT_ENABLED", "1").lower() in (
    "1",
    "true",
    "yes",
)
AUDIO_SAMPLE_RATE = 16_000
SPECTRAL_OVER_SUBTRACTION = 1.3
SPECTRAL_FLOOR = 0.10
VOICE_LOW_CUT_HZ = 80.0
VOICE_HIGH_CUT_HZ = 7500.0

_DF_MODEL: Any | None = None
_DF_STATE: Any | None = None
_DF_INITIALIZATION_ATTEMPTED = False


def is_audio_enhancement_enabled() -> bool:
    return AUDIO_ENHANCEMENT_ENABLED


def _try_enhance_with_deepfilternet(
    audio: np.ndarray,
    sample_rate: int,
) -> np.ndarray | None:
    global _DF_MODEL, _DF_STATE, _DF_INITIALIZATION_ATTEMPTED
    if _DF_INITIALIZATION_ATTEMPTED and _DF_MODEL is None:
        return None

    try:
        import torch
        from df.enhance import enhance, init_df

        if _DF_MODEL is None:
            _DF_INITIALIZATION_ATTEMPTED = True
            _DF_MODEL, _DF_STATE, _ = init_df()

        audio_tensor = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        enhanced_tensor = enhance(_DF_MODEL, _DF_STATE, audio_tensor)
        return enhanced_tensor.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception:
        _DF_INITIALIZATION_ATTEMPTED = True
        _DF_MODEL = None
        _DF_STATE = None
        return None


def _enhance_with_spectral_gate(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    *,
    over_subtraction: float = SPECTRAL_OVER_SUBTRACTION,
    spectral_floor: float = SPECTRAL_FLOOR,
) -> np.ndarray:
    """Atténue les bruits stationnaires et parasites par soustraction spectrale et filtrage vocal."""
    n_fft = 512
    hop_length = 256
    pad_amount = n_fft // 2
    padded = np.pad(audio, pad_amount, mode="reflect")
    window = np.hanning(n_fft).astype(np.float32)

    num_frames = 1 + (len(padded) - n_fft) // hop_length
    frames = np.lib.stride_tricks.sliding_window_view(
        padded[: (num_frames - 1) * hop_length + n_fft], n_fft
    )[::hop_length]
    stft_spec = np.fft.rfft(frames * window, axis=-1).T

    power = np.abs(stft_spec) ** 2
    frame_energy = np.mean(power, axis=0)

    # Estimation du profil de bruit à partir des 15 % de trames les plus calmes
    cutoff_idx = max(1, int(num_frames * 0.15))
    quiet_frame_indices = np.argsort(frame_energy)[:cutoff_idx]
    noise_power = np.mean(power[:, quiet_frame_indices], axis=1, keepdims=True)

    # Gain de type Wiener avec plancher de sécurité pour éviter le bruit musical
    snr_ratio = (over_subtraction * noise_power) / (power + 1e-10)
    gain = np.clip(1.0 - snr_ratio, spectral_floor, 1.0)

    # Filtrage passe-bande adapté aux formants de la voix humaine (80 Hz -> 7500 Hz)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    bandpass = np.ones(len(freqs), dtype=np.float32)

    low_cut = freqs < VOICE_LOW_CUT_HZ
    if np.any(low_cut):
        bandpass[low_cut] = np.clip((freqs[low_cut] / VOICE_LOW_CUT_HZ) ** 2, 0.05, 1.0)

    high_cut = freqs > VOICE_HIGH_CUT_HZ
    if np.any(high_cut):
        bandpass[high_cut] = np.clip(
            1.0 - ((freqs[high_cut] - VOICE_HIGH_CUT_HZ) / (sample_rate / 2 - VOICE_HIGH_CUT_HZ + 1e-5)),
            0.05,
            1.0,
        )

    gain = gain * bandpass[:, np.newaxis]
    filtered_stft = (stft_spec * gain).T

    time_frames = np.fft.irfft(filtered_stft, n=n_fft, axis=-1) * window
    expected_len = (num_frames - 1) * hop_length + n_fft
    output = np.zeros(expected_len, dtype=np.float32)
    window_sum = np.zeros(expected_len, dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_length
        output[start : start + n_fft] += time_frames[i]
        window_sum[start : start + n_fft] += window**2

    nonzero = window_sum > 1e-6
    output[nonzero] /= window_sum[nonzero]
    cleaned = output[pad_amount : pad_amount + len(audio)]
    return np.clip(cleaned, -1.0, 1.0).astype(np.float32)


def enhance_audio_array(
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
    *,
    over_subtraction: float = SPECTRAL_OVER_SUBTRACTION,
    spectral_floor: float = SPECTRAL_FLOOR,
) -> np.ndarray:
    """Isole la voix et atténue les bruits parasites sur un tableau audio 1D float32."""
    if not is_audio_enhancement_enabled():
        return audio

    if audio.size == 0 or len(audio) < 512:
        return audio

    try:
        df_result = _try_enhance_with_deepfilternet(audio, sample_rate)
        if df_result is not None:
            return df_result

        return _enhance_with_spectral_gate(
            audio,
            sample_rate=sample_rate,
            over_subtraction=over_subtraction,
            spectral_floor=spectral_floor,
        )
    except Exception as exc:
        logger.warning("Échec de l'amélioration audio, signal brut conservé: %s", exc)
        return audio


def load_audio_as_float32(audio_path: Path) -> np.ndarray:
    """Décode un fichier audio quelconque en tableau 1D float32 normalisé."""
    try:
        from faster_whisper.audio import decode_audio

        return np.asarray(decode_audio(str(audio_path)), dtype=np.float32)
    except Exception:
        pass

    try:
        with wave.open(str(audio_path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames = wf.readframes(wf.getnframes())
            dtype = np.int16 if wf.getsampwidth() == 2 else np.uint8
            raw = np.frombuffer(frames, dtype=dtype)
            if channels > 1:
                raw = raw.reshape(-1, channels).mean(axis=1)
            audio = raw.astype(np.float32) / 32768.0

            if sample_rate != AUDIO_SAMPLE_RATE and len(audio) > 0:
                target_len = int(len(audio) * AUDIO_SAMPLE_RATE / sample_rate)
                audio = np.interp(
                    np.linspace(0, len(audio), target_len, endpoint=False),
                    np.arange(len(audio)),
                    audio,
                ).astype(np.float32)
            return audio
    except Exception:
        return np.empty(0, dtype=np.float32)


def write_pcm16_wav(
    output_path: Path,
    audio: np.ndarray,
    sample_rate: int = AUDIO_SAMPLE_RATE,
) -> None:
    """Écrit un tableau float32 au format WAV PCM 16 bits mono."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def enhance_audio_file(
    audio_path: str | Path,
    output_path: str | Path | None = None,
) -> tuple[Path, bool]:
    """Applique l'isolation vocale à un fichier audio et écrit un WAV nettoyé."""
    input_path = Path(audio_path)
    if not is_audio_enhancement_enabled():
        return input_path, False

    try:
        audio = load_audio_as_float32(input_path)
        if audio.size == 0:
            return input_path, False

        cleaned = enhance_audio_array(audio, sample_rate=AUDIO_SAMPLE_RATE)

        if output_path is None:
            import tempfile

            temp_file = tempfile.NamedTemporaryFile(
                suffix=".wav", prefix="enhanced_", delete=False
            )
            dest = Path(temp_file.name)
            temp_file.close()
        else:
            dest = Path(output_path)

        write_pcm16_wav(dest, cleaned, sample_rate=AUDIO_SAMPLE_RATE)
        return dest, True
    except Exception as exc:
        logger.warning(
            "Impossible d'améliorer le fichier audio %s: %s", audio_path, exc
        )
        return input_path, False
