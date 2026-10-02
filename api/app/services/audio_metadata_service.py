from __future__ import annotations

import math
from pathlib import Path


class AudioMetadataError(Exception):
    pass


def _import_runtime_dependencies():
    import av

    return av


def _compute_duration_seconds(sample_rate: int, num_frames: int) -> float | None:
    if sample_rate <= 0 or num_frames <= 0:
        return None

    duration_seconds = num_frames / sample_rate

    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return None

    return duration_seconds


def get_audio_duration_seconds(audio_path: str | Path) -> float:
    try:
        av = _import_runtime_dependencies()
        # Use the same decoder as Whisper. TorchAudio's available backends
        # depend on the system FFmpeg version and may reject WebM/M4A.
        with av.open(str(audio_path)) as container:
            if not container.streams.audio:
                raise ValueError("No audio stream")
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                duration_seconds = float(stream.duration * stream.time_base)
                if math.isfinite(duration_seconds) and duration_seconds > 0:
                    return duration_seconds

            # Browser recordings may omit duration metadata. Count decoded
            # samples without retaining the entire waveform in memory.
            duration_seconds = 0.0
            for frame in container.decode(stream):
                frame_duration = _compute_duration_seconds(frame.sample_rate, frame.samples)
                if frame_duration is None:
                    raise ValueError("Invalid audio frame duration")
                duration_seconds += frame_duration
            if math.isfinite(duration_seconds) and duration_seconds > 0:
                return duration_seconds
    except Exception as exc:
        raise AudioMetadataError("Impossible de lire la durée du fichier audio.") from exc

    raise AudioMetadataError("Impossible de lire la durée du fichier audio.")
