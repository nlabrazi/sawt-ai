# ROLE
# ----
# Transcrit un fichier audio avec le modèle faster-whisper déjà chargé.

import logging
import math
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable

from app.core.transcription_policy import is_confidently_non_arabic
from app.core.model_loader import get_whisper_model

logger = logging.getLogger(__name__)

WHISPER_BEAM_SIZE = 5
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6
VAD_MIN_SILENCE_DURATION_MS = 500
VAD_SPEECH_PAD_MS = 400
WHISPER_VAD_FILTER = True
QURAN_TRANSCRIPTION_VAD_FILTER = False
QURAN_TRANSCRIPTION_LANGUAGE = "ar"
QURAN_TRANSCRIPTION_DITHER_SNR_DB = 40.0
QURAN_TRANSCRIPTION_DITHER_SEED = 20_260_715
LANGUAGE_SCREENING_WINDOWS = 3
LANGUAGE_SCREENING_WINDOW_SECONDS = 30
AUDIO_SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class TranscriptionSegmentMetrics:
    start_seconds: float | None
    end_seconds: float | None
    average_log_probability: float | None
    no_speech_probability: float | None
    compression_ratio: float | None
    temperature: float | None


@dataclass(frozen=True, slots=True)
class AudioLanguageScreen:
    language: str | None
    language_probability: float | None
    arabic_probability: float | None
    language_probabilities: tuple[tuple[str, float], ...]
    duration_seconds: float
    speech_duration_seconds: float

    @property
    def has_speech(self) -> bool:
        return self.speech_duration_seconds > 0


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


def _distributed_audio_windows(
    audio: Any,
    *,
    window_sample_count: int,
    max_windows: int,
) -> tuple[Any, ...]:
    """Échantillonne début, milieu et fin sans dépendre du premier bloc."""
    if window_sample_count <= 0 or max_windows <= 0:
        raise ValueError("Language screening window values must be positive.")

    if len(audio) <= window_sample_count:
        return (audio,)

    window_count = min(max_windows, math.ceil(len(audio) / window_sample_count))

    if window_count == 1:
        return (audio[:window_sample_count],)

    last_start = len(audio) - window_sample_count
    starts = (
        round(index * last_start / (window_count - 1))
        for index in range(window_count)
    )
    return tuple(
        audio[start:start + window_sample_count]
        for start in starts
    )


def _detect_language_across_windows(
    speech_audio: Any,
    model: Any,
) -> tuple[str, float, tuple[tuple[str, float], ...]]:
    """Agrège explicitement plusieurs fenêtres temporelles.

    faster-whisper peut arrêter sa détection dès le premier bloc dépassant son
    seuil. Des appels indépendants rendent donc réellement observables le
    début, le milieu et la fin. La probabilité maximale par langue conserve une
    trace prudente d'une récitation arabe présente dans une seule fenêtre.
    """
    windows = _distributed_audio_windows(
        speech_audio,
        window_sample_count=(
            LANGUAGE_SCREENING_WINDOW_SECONDS * AUDIO_SAMPLE_RATE
        ),
        max_windows=LANGUAGE_SCREENING_WINDOWS,
    )
    detections = [
        model.detect_language(
            audio=window,
            language_detection_segments=1,
        )
        for window in windows
    ]
    votes = Counter(language for language, _probability, _all in detections)
    winning_probabilities = {
        language: max(
            float(probability)
            for detected_language, probability, _all in detections
            if detected_language == language
        )
        for language in votes
    }
    language = sorted(
        votes,
        key=lambda candidate: (
            -votes[candidate],
            -winning_probabilities[candidate],
            candidate,
        ),
    )[0]
    probability_by_language: dict[str, float] = {}

    for _language, _probability, raw_probabilities in detections:
        for candidate_language, probability in raw_probabilities:
            probability_by_language[candidate_language] = max(
                probability_by_language.get(candidate_language, 0.0),
                float(probability),
            )

    language_probabilities = tuple(
        sorted(
            probability_by_language.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    return language, winning_probabilities[language], language_probabilities


def detect_audio_language(audio_path: str) -> AudioLanguageScreen:
    """Détecte parole et langue avant tout décodage de texte.

    Le VAD sert ici uniquement à éviter de classifier le silence et le bruit.
    Le décodage coranique garde ensuite le signal original, car les longues
    modulations d'une récitation peuvent être supprimées par un VAD de parole.
    """
    import numpy as np
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    audio = decode_audio(audio_path)
    duration_seconds = len(audio) / AUDIO_SAMPLE_RATE
    speech_chunks = get_speech_timestamps(
        audio,
        VadOptions(
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS,
        ),
    )
    speech_duration_seconds = sum(
        chunk["end"] - chunk["start"] for chunk in speech_chunks
    ) / AUDIO_SAMPLE_RATE

    if not speech_chunks:
        return AudioLanguageScreen(
            language=None,
            language_probability=None,
            arabic_probability=None,
            language_probabilities=(),
            duration_seconds=duration_seconds,
            speech_duration_seconds=0.0,
        )

    speech_audio = np.concatenate(
        [audio[chunk["start"]:chunk["end"]] for chunk in speech_chunks]
    )
    language, language_probability, language_probabilities = (
        _detect_language_across_windows(
            speech_audio,
            get_whisper_model(),
        )
    )
    arabic_probability = next(
        (
            probability
            for candidate_language, probability in language_probabilities
            if candidate_language == "ar"
        ),
        0.0,
    )

    return AudioLanguageScreen(
        language=language,
        language_probability=float(language_probability),
        arabic_probability=arabic_probability,
        language_probabilities=language_probabilities,
        duration_seconds=duration_seconds,
        speech_duration_seconds=speech_duration_seconds,
    )


def _empty_screened_transcription(
    screen: AudioLanguageScreen,
) -> TranscriptionResult:
    return TranscriptionResult(
        [],
        TranscriptionMetadata(
            language=screen.language,
            language_probability=screen.language_probability,
            arabic_probability=screen.arabic_probability,
            language_probabilities=screen.language_probabilities,
            duration_seconds=screen.duration_seconds,
            duration_after_vad_seconds=screen.speech_duration_seconds,
            speech_duration_seconds=screen.speech_duration_seconds,
            average_log_probability=None,
            average_no_speech_probability=None,
            max_compression_ratio=None,
            max_temperature=None,
            segment_metrics=(),
        ),
    )


def _decode_audio(audio_path: str):
    from faster_whisper.audio import decode_audio

    return decode_audio(audio_path)


def _prepare_dithered_audio(
    audio_path: str,
    *,
    snr_db: float,
):
    """Ajoute un dither déterministe très faible pour stabiliser le décodage.

    Certains enregistrements propres et très modulés sont tronqués par Whisper,
    alors que le même signal avec un bruit imperceptible est décodé en entier.
    Ce prétraitement intervient uniquement après le filtre parole/langue.
    """
    import numpy as np

    if not math.isfinite(snr_db) or snr_db <= 0:
        raise ValueError("The transcription dither SNR must be positive.")

    audio = np.asarray(_decode_audio(audio_path), dtype=np.float32)

    if audio.size == 0:
        return audio

    signal_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))

    if signal_rms == 0:
        return audio

    rng = np.random.default_rng(QURAN_TRANSCRIPTION_DITHER_SEED)
    noise = rng.uniform(-1.0, 1.0, size=audio.shape).astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(np.square(noise, dtype=np.float64))))
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    mixed = audio + noise * (target_noise_rms / noise_rms)
    peak = float(np.max(np.abs(mixed)))

    if peak > 0.98:
        mixed *= 0.98 / peak

    return mixed.astype(np.float32, copy=False)


def transcribe_audio(
    audio_path: str,
    clip_end_seconds: float | None = None,
    *,
    language: str | None = None,
    vad_filter: bool = WHISPER_VAD_FILTER,
    dither_snr_db: float | None = None,
) -> TranscriptionResult:
    model = get_whisper_model()
    audio_source = (
        _prepare_dithered_audio(audio_path, snr_db=dither_snr_db)
        if dither_snr_db is not None
        else audio_path
    )
    clip_options = (
        {"clip_timestamps": [0, clip_end_seconds]}
        if clip_end_seconds is not None
        else {}
    )

    transcription_options: dict[str, Any] = {
        "beam_size": WHISPER_BEAM_SIZE,
        "log_prob_threshold": WHISPER_LOG_PROB_THRESHOLD,
        "no_speech_threshold": WHISPER_NO_SPEECH_THRESHOLD,
        "condition_on_previous_text": False,
        "vad_filter": vad_filter,
        **clip_options,
    }
    if language is not None:
        transcription_options["language"] = language
    if vad_filter:
        transcription_options["vad_parameters"] = {
            "min_silence_duration_ms": VAD_MIN_SILENCE_DURATION_MS,
            "speech_pad_ms": VAD_SPEECH_PAD_MS,
        }

    segments, info = model.transcribe(audio_source, **transcription_options)

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


def transcribe_quran_audio(
    audio_path: str,
) -> TranscriptionResult:
    """Filtre les non-paroles/non-arabes puis décode le signal complet en arabe."""
    screen = detect_audio_language(audio_path)

    if not screen.has_speech or is_confidently_non_arabic(
        screen.language,
        screen.language_probability,
        screen.arabic_probability,
    ):
        return _empty_screened_transcription(screen)

    transcription = transcribe_audio(
        audio_path,
        language=QURAN_TRANSCRIPTION_LANGUAGE,
        vad_filter=QURAN_TRANSCRIPTION_VAD_FILTER,
        dither_snr_db=QURAN_TRANSCRIPTION_DITHER_SNR_DB,
    )
    screened_metadata = replace(
        transcription.metadata,
        language=screen.language,
        language_probability=screen.language_probability,
        arabic_probability=screen.arabic_probability,
        language_probabilities=screen.language_probabilities,
        duration_seconds=screen.duration_seconds,
        duration_after_vad_seconds=screen.speech_duration_seconds,
        speech_duration_seconds=screen.speech_duration_seconds,
    )

    return TranscriptionResult(transcription, screened_metadata)
