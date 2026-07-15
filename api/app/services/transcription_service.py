# ROLE
# ----
# Transcrit un fichier audio avec le modèle faster-whisper déjà chargé.

import logging
from dataclasses import dataclass
from typing import Any, Iterable

from app.core.model_loader import get_whisper_model

logger = logging.getLogger(__name__)

WHISPER_BEAM_SIZE = 5
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6
VAD_MIN_SILENCE_DURATION_MS = 500
VAD_SPEECH_PAD_MS = 400


@dataclass(frozen=True, slots=True)
class TranscriptionSegmentMetrics:
    start_seconds: float | None
    end_seconds: float | None
    average_log_probability: float | None
    no_speech_probability: float | None
    compression_ratio: float | None
    temperature: float | None


@dataclass(frozen=True, slots=True)
class TranscriptionMetadata:
    language: str | None
    language_probability: float | None
    arabic_probability: float | None
    language_probabilities: tuple[tuple[str, float], ...]
    duration_seconds: float | None
    duration_after_vad_seconds: float | None
    speech_duration_seconds: float
    average_log_probability: float | None
    average_no_speech_probability: float | None
    max_compression_ratio: float | None
    max_temperature: float | None
    segment_metrics: tuple[TranscriptionSegmentMetrics, ...]


class TranscriptionResult(list[dict[str, str]]):
    """Liste compatible avec l'ancien contrat, enrichie de métriques Whisper."""

    __slots__ = ("metadata",)

    def __init__(
        self,
        segments: Iterable[dict[str, str]],
        metadata: TranscriptionMetadata,
    ) -> None:
        super().__init__(segments)
        self.metadata = metadata

    @property
    def segments(self) -> list[dict[str, str]]:
        return self


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _read_float(source: Any, attribute: str) -> float | None:
    return _optional_float(getattr(source, attribute, None))


def _read_language_probabilities(info: Any) -> tuple[tuple[str, float], ...]:
    probabilities: list[tuple[str, float]] = []

    for item in getattr(info, "all_language_probs", None) or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue

        language, probability = item
        parsed_probability = _optional_float(probability)

        if not isinstance(language, str) or parsed_probability is None:
            continue

        probabilities.append((language, parsed_probability))

    return tuple(probabilities)


def _weighted_average(
    values_and_weights: Iterable[tuple[float | None, float]],
) -> float | None:
    weighted_total = 0.0
    total_weight = 0.0

    for value, weight in values_and_weights:
        if value is None:
            continue

        safe_weight = weight if weight > 0 else 1.0
        weighted_total += value * safe_weight
        total_weight += safe_weight

    return weighted_total / total_weight if total_weight else None


def _maximum(values: Iterable[float | None]) -> float | None:
    present_values = [value for value in values if value is not None]
    return max(present_values) if present_values else None


def _build_metadata(info: Any, metrics: list[TranscriptionSegmentMetrics]):
    language = getattr(info, "language", None)
    language = language if isinstance(language, str) else None
    language_probability = _read_float(info, "language_probability")
    language_probabilities = _read_language_probabilities(info)
    arabic_probability = next(
        (
            probability
            for candidate_language, probability in language_probabilities
            if candidate_language == "ar"
        ),
        language_probability if language == "ar" else None,
    )
    durations = [
        max(0.0, (metric.end_seconds or 0.0) - (metric.start_seconds or 0.0))
        if metric.start_seconds is not None and metric.end_seconds is not None
        else 1.0
        for metric in metrics
    ]

    return TranscriptionMetadata(
        language=language,
        language_probability=language_probability,
        arabic_probability=arabic_probability,
        language_probabilities=language_probabilities,
        duration_seconds=_read_float(info, "duration"),
        duration_after_vad_seconds=_read_float(info, "duration_after_vad"),
        speech_duration_seconds=sum(
            duration
            for metric, duration in zip(metrics, durations)
            if metric.start_seconds is not None and metric.end_seconds is not None
        ),
        average_log_probability=_weighted_average(
            (metric.average_log_probability, duration)
            for metric, duration in zip(metrics, durations)
        ),
        average_no_speech_probability=_weighted_average(
            (metric.no_speech_probability, duration)
            for metric, duration in zip(metrics, durations)
        ),
        max_compression_ratio=_maximum(metric.compression_ratio for metric in metrics),
        max_temperature=_maximum(metric.temperature for metric in metrics),
        segment_metrics=tuple(metrics),
    )


def transcribe_audio(
    audio_path: str,
    clip_end_seconds: float | None = None,
) -> TranscriptionResult:
    model = get_whisper_model()
    clip_options = (
        {"clip_timestamps": [0, clip_end_seconds]}
        if clip_end_seconds is not None
        else {}
    )

    segments, info = model.transcribe(
        audio_path,
        beam_size=WHISPER_BEAM_SIZE,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters={
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        },
        **clip_options,
    )

    result: list[dict[str, str]] = []
    metrics: list[TranscriptionSegmentMetrics] = []

    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        result.append({"text": text})
        metrics.append(
            TranscriptionSegmentMetrics(
                start_seconds=_read_float(segment, "start"),
                end_seconds=_read_float(segment, "end"),
                average_log_probability=_read_float(segment, "avg_logprob"),
                no_speech_probability=_read_float(segment, "no_speech_prob"),
                compression_ratio=_read_float(segment, "compression_ratio"),
                temperature=_read_float(segment, "temperature"),
            )
        )

    transcription = TranscriptionResult(result, _build_metadata(info, metrics))

    logger.info(
        "Transcription complete: segments=%s language=%s language_probability=%s average_log_probability=%s",
        len(transcription),
        transcription.metadata.language,
        transcription.metadata.language_probability,
        transcription.metadata.average_log_probability,
    )

    return transcription
