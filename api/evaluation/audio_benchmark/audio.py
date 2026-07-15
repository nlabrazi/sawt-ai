from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
import wave
from array import array
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Sequence

from evaluation.audio_benchmark.secure_io import atomic_output_path

DEFAULT_SAMPLE_RATE = 16_000
PCM_MAX = 32_767


class AudioGenerationError(ValueError):
    """Le corpus audio ne peut pas être construit de façon reproductible."""


class FrenchTtsUnavailableError(AudioGenerationError):
    """Aucun moteur TTS français local et hors ligne n'est disponible."""


def _frame_count(duration_seconds: float, sample_rate: int) -> int:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise AudioGenerationError("La durée audio doit être strictement positive.")

    if sample_rate <= 0:
        raise AudioGenerationError("La fréquence d'échantillonnage doit être positive.")

    return max(1, round(duration_seconds * sample_rate))


def _rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0

    return math.sqrt(math.fsum(sample * sample for sample in samples) / len(samples))


def _scale_to_rms(samples: Sequence[float], target_rms: float) -> array:
    current_rms = _rms(samples)

    if current_rms == 0:
        return array("f", samples)

    scale = target_rms / current_rms
    return array("f", (sample * scale for sample in samples))


def peak_limit(samples: Sequence[float], peak: float = 0.98) -> array:
    if not 0 < peak <= 1:
        raise AudioGenerationError("La limite de crête doit être dans ]0, 1].")

    max_amplitude = max((abs(sample) for sample in samples), default=0.0)
    scale = peak / max_amplitude if max_amplitude > peak else 1.0
    return array("f", (max(-1.0, min(1.0, sample * scale)) for sample in samples))


def generate_silence(duration_seconds: float, sample_rate: int = DEFAULT_SAMPLE_RATE) -> array:
    return array("f", [0.0]) * _frame_count(duration_seconds, sample_rate)


def generate_white_noise(
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    seed: int,
    target_rms: float = 0.08,
) -> array:
    rng = random.Random(seed)
    samples = array(
        "f",
        (rng.uniform(-1.0, 1.0) for _ in range(_frame_count(duration_seconds, sample_rate))),
    )
    return _scale_to_rms(samples, target_rms)


def generate_pink_noise(
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    seed: int,
    target_rms: float = 0.08,
) -> array:
    """Génère un bruit rose déterministe avec le filtre de Paul Kellet."""
    rng = random.Random(seed)
    b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
    samples = array("f")

    for _ in range(_frame_count(duration_seconds, sample_rate)):
        white = rng.uniform(-1.0, 1.0)
        b0 = 0.99886 * b0 + white * 0.0555179
        b1 = 0.99332 * b1 + white * 0.0750759
        b2 = 0.96900 * b2 + white * 0.1538520
        b3 = 0.86650 * b3 + white * 0.3104856
        b4 = 0.55000 * b4 + white * 0.5329522
        b5 = -0.7616 * b5 - white * 0.0168980
        pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
        b6 = white * 0.115926
        samples.append(pink)

    return _scale_to_rms(samples, target_rms)


def generate_background_noise(
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    seed: int,
    target_rms: float = 0.08,
) -> array:
    """Simule un fond domestique non vocal : souffle, ronflement et sons intermittents."""
    frames = _frame_count(duration_seconds, sample_rate)
    pink = generate_pink_noise(
        duration_seconds,
        sample_rate,
        seed=seed,
        target_rms=0.05,
    )
    rng = random.Random(seed + 1)
    phase_a = rng.uniform(0, 2 * math.pi)
    phase_b = rng.uniform(0, 2 * math.pi)
    samples = array("f")

    for index in range(frames):
        time_seconds = index / sample_rate
        hum = 0.014 * math.sin(2 * math.pi * 50 * time_seconds + phase_a)
        hum += 0.007 * math.sin(2 * math.pi * 100 * time_seconds + phase_b)

        cycle = time_seconds % 2.75
        transient_envelope = max(0.0, 1.0 - abs(cycle - 1.25) / 0.18)
        transient = transient_envelope * 0.035 * math.sin(
            2 * math.pi * (620 + 90 * cycle) * time_seconds
        )
        samples.append(pink[index] + hum + transient)

    return _scale_to_rms(samples, target_rms)


def generate_tone(
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    frequency_hz: float = 440.0,
    amplitude: float = 0.18,
) -> array:
    if frequency_hz <= 0 or frequency_hz >= sample_rate / 2:
        raise AudioGenerationError("La fréquence de la tonalité est invalide.")

    frames = _frame_count(duration_seconds, sample_rate)
    return array(
        "f",
        (
            amplitude * math.sin(2 * math.pi * frequency_hz * index / sample_rate)
            for index in range(frames)
        ),
    )


def generate_synthetic_song(
    duration_seconds: float,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    tempo_bpm: float = 108.0,
    amplitude: float = 0.18,
) -> array:
    """Génère une courte mélodie instrumentale, sans voix ni contenu protégé."""
    if tempo_bpm <= 0:
        raise AudioGenerationError("Le tempo doit être strictement positif.")

    notes_hz = (261.63, 329.63, 392.00, 523.25, 392.00, 329.63, 293.66, 349.23)
    frames = _frame_count(duration_seconds, sample_rate)
    note_duration = 60.0 / tempo_bpm
    samples = array("f")

    for index in range(frames):
        time_seconds = index / sample_rate
        note_position = time_seconds / note_duration
        note_index = int(note_position) % len(notes_hz)
        local_position = note_position - math.floor(note_position)
        envelope = min(1.0, local_position / 0.04) * min(1.0, (1.0 - local_position) / 0.12)
        frequency = notes_hz[note_index]
        fundamental = math.sin(2 * math.pi * frequency * time_seconds)
        harmonic = 0.28 * math.sin(2 * math.pi * frequency * 2 * time_seconds)
        samples.append(amplitude * envelope * (fundamental + harmonic))

    return peak_limit(samples)


def mix_at_snr(
    signal: Sequence[float],
    noise: Sequence[float],
    snr_db: float,
) -> array:
    if len(signal) != len(noise):
        raise AudioGenerationError("Le signal et le bruit doivent avoir la même durée.")

    if not math.isfinite(snr_db):
        raise AudioGenerationError("Le SNR doit être un nombre fini.")

    signal_rms = _rms(signal)
    noise_rms = _rms(noise)

    if signal_rms == 0:
        raise AudioGenerationError("Impossible d'appliquer un SNR à un signal silencieux.")

    if noise_rms == 0:
        raise AudioGenerationError("Impossible d'appliquer un SNR avec un bruit silencieux.")

    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise_scale = target_noise_rms / noise_rms
    return peak_limit(
        array("f", (sample + noise[index] * noise_scale for index, sample in enumerate(signal)))
    )


def write_pcm16_wav(
    path: str | Path,
    samples: Sequence[float],
    sample_rate: int,
    *,
    private: bool = False,
) -> None:
    output_path = Path(path)
    pcm = array(
        "h",
        (
            round(max(-1.0, min(1.0, sample)) * PCM_MAX)
            for sample in samples
        ),
    )

    if sys.byteorder != "little":
        pcm.byteswap()

    with atomic_output_path(output_path, private=private) as temp_path:
        with wave.open(str(temp_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm.tobytes())


def read_pcm16_wav(path: str | Path) -> tuple[array, int]:
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
            raise AudioGenerationError("Le WAV doit être en PCM 16 bits.")

        channel_count = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        raw_samples = array("h")
        raw_samples.frombytes(wav_file.readframes(wav_file.getnframes()))

    if sys.byteorder != "little":
        raw_samples.byteswap()

    if channel_count <= 0:
        raise AudioGenerationError("Le WAV ne contient aucun canal.")

    if channel_count == 1:
        return array("f", (sample / PCM_MAX for sample in raw_samples)), sample_rate

    mono = array("f")
    for offset in range(0, len(raw_samples), channel_count):
        frame = raw_samples[offset:offset + channel_count]
        mono.append(sum(frame) / (len(frame) * PCM_MAX))

    return mono, sample_rate


def inspect_pcm16_wav(path: str | Path) -> tuple[int, int, int]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            if wav_file.getcomptype() != "NONE" or wav_file.getsampwidth() != 2:
                raise AudioGenerationError("Le WAV construit doit être en PCM 16 bits.")

            channel_count = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
    except (OSError, wave.Error) as exc:
        raise AudioGenerationError("L'en-tête WAV construit est invalide.") from exc

    if channel_count != 1:
        raise AudioGenerationError("Le WAV construit doit être mono.")
    if sample_rate <= 0 or frame_count <= 0:
        raise AudioGenerationError("Le WAV construit ne contient pas d'audio valide.")

    return sample_rate, channel_count, frame_count


def _decode_with_ffmpeg(source_path: Path, sample_rate: int, ffmpeg_binary: str) -> array:
    if shutil.which(ffmpeg_binary) is None:
        raise AudioGenerationError(
            "ffmpeg est requis pour convertir ce fichier en WAV mono 16 kHz."
        )

    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        normalized_path = Path(temp_file.name)

    try:
        process = subprocess.run(
            [
                ffmpeg_binary,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source_path),
                "-ac",
                "1",
                "-ar",
                str(sample_rate),
                "-c:a",
                "pcm_s16le",
                str(normalized_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if process.returncode != 0:
            raise AudioGenerationError(
                f"ffmpeg n'a pas pu décoder {source_path.name}."
            )

        samples, decoded_sample_rate = read_pcm16_wav(normalized_path)
        if decoded_sample_rate != sample_rate:
            raise AudioGenerationError("ffmpeg a produit une fréquence inattendue.")
        return samples
    finally:
        normalized_path.unlink(missing_ok=True)


def load_local_audio(
    source_path: str | Path,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> array:
    path = Path(source_path)
    if not path.is_file():
        raise AudioGenerationError(f"Source audio locale absente : {path.name}")

    if path.suffix.lower() == ".wav":
        try:
            samples, source_sample_rate = read_pcm16_wav(path)
            if source_sample_rate == sample_rate:
                return samples
        except (AudioGenerationError, wave.Error):
            pass

    return _decode_with_ffmpeg(path, sample_rate, ffmpeg_binary)


def find_french_tts_backend() -> tuple[str, str] | None:
    for executable in ("espeak-ng", "espeak"):
        binary = shutil.which(executable)
        if binary is not None:
            voices = subprocess.run(
                [binary, "--voices=fr"],
                check=False,
                capture_output=True,
                text=True,
            )
            if voices.returncode == 0 and "fr" in voices.stdout.lower():
                return executable, binary

    pico_binary = shutil.which("pico2wave")
    if pico_binary is not None:
        return "pico2wave", pico_binary

    return None


def render_french_tts(
    text: str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> array:
    if not text.strip():
        raise AudioGenerationError("Le texte TTS français ne peut pas être vide.")

    backend = find_french_tts_backend()
    if backend is None:
        raise FrenchTtsUnavailableError(
            "Aucun moteur TTS français local détecté (espeak-ng, espeak ou pico2wave)."
        )

    backend_name, binary = backend
    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        output_path = Path(temp_file.name)

    try:
        if backend_name in {"espeak-ng", "espeak"}:
            command = [binary, "-v", "fr", "-s", "150", "-w", str(output_path), text]
        else:
            command = [binary, "-l", "fr-FR", "-w", str(output_path), text]

        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            raise AudioGenerationError("Le moteur TTS français local a échoué.")

        return load_local_audio(output_path, sample_rate)
    finally:
        output_path.unlink(missing_ok=True)
