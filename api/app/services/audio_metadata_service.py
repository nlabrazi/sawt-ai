from __future__ import annotations

import math
from pathlib import Path


class AudioMetadataError(Exception):
    pass


def _import_runtime_dependencies():
    import torchaudio

    return torchaudio


def _compute_duration_seconds(sample_rate: int, num_frames: int) -> float | None:
    if sample_rate <= 0 or num_frames <= 0:
        return None

    duration_seconds = num_frames / sample_rate

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return None

    return duration_seconds


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    try:
        torchaudio = _import_runtime_dependencies()
        normalized_path = str(audio_path)
        metadata = torchaudio.info(normalized_path)
        duration_seconds = _compute_duration_seconds(
            int(getattr(metadata, "sample_rate", 0) or 0),
            int(getattr(metadata, "num_frames", 0) or 0),
        )

        if duration_seconds is not None:
            return duration_seconds

        waveform, sample_rate = torchaudio.load(normalized_path)
        fallback_num_frames = int(getattr(waveform, "shape", [0])[-1] or 0)
        fallback_duration_seconds = _compute_duration_seconds(sample_rate, fallback_num_frames)

        if fallback_duration_seconds is not None:
            return fallback_duration_seconds
    except Exception as exc:
        raise AudioMetadataError("Impossible de lire la durée du fichier audio.") from exc

    raise AudioMetadataError("Impossible de lire la durée du fichier audio.")
