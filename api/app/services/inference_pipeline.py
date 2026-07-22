# ROLE
# ----
# Orchestration principale :
# transcription -> détection verset -> détection imam

import logging
from dataclasses import dataclass, replace
from typing import Literal

from app.core.api_logger import log_api_event
from app.core.detection_policy import MAX_AMBIGUOUS_RESCUE_SCORE
from app.core.transcription_policy import (
    HIGH_COMPRESSION_MAX_LOG_PROBABILITY,
    HIGH_COMPRESSION_RATIO,
    HIGH_TEMPERATURE,
    HIGH_TEMPERATURE_MAX_LOG_PROBABILITY,
    LANGUAGE_CONFLICT_RESCUE_MIN_SPEECH_SECONDS,
    LANGUAGE_CONFLICT_MIN_MATCHED_WORD_COUNT,
    LANGUAGE_CONFLICT_MIN_QURAN_SIMILARITY,
    MIN_AVERAGE_LOG_PROBABILITY,
    is_confidently_non_arabic,
)
from app.services.imam_prediction_service import ImamPredictionError, predict_imam
from app.services.transcription_service import (
    TranscriptionMetadata,
    TranscriptionResult,
    transcribe_quran_audio as transcribe_audio,
    transcribe_quran_audio_rescue as transcribe_rescue_audio,
)
from app.services.verse_detection_service import (
    VerseDetectionOutcome,
    detect_verse_with_metadata,
)

logger = logging.getLogger(__name__)

ImamStatus = Literal["disabled", "unknown", "unavailable", "high", "medium", "low"]
AudioRejectionReason = Literal[
    "insufficient_speech",
    "non_arabic_speech",
    "low_transcription_confidence",
]
AudioQualityWarning = Literal[
    "low_transcription_confidence",
    "non_arabic_speech",
]
AUDIO_REJECTION_REASONS: frozenset[AudioRejectionReason] = frozenset(
    {
        "insufficient_speech",
        "non_arabic_speech",
        "low_transcription_confidence",
    }
)
@dataclass(frozen=True, slots=True)
class AudioQualityAssessment:
    accepted: bool
    rejection_reason: AudioRejectionReason | None
    warning_reason: AudioQualityWarning | None = None


def assess_audio_quality(segments) -> AudioQualityAssessment:
    """Écarte uniquement les cas dont les signaux Whisper sont suffisamment nets."""
    metadata: TranscriptionMetadata | None = getattr(segments, "metadata", None)
    has_transcription = any(
        bool(segment.get("text", "").strip())
        for segment in segments
    )

    if metadata is not None and is_confidently_non_arabic(
        metadata.language,
        metadata.language_probability,
        metadata.arabic_probability,
    ):
        if has_transcription:
            return AudioQualityAssessment(
                True,
                None,
                warning_reason="non_arabic_speech",
            )
        return AudioQualityAssessment(False, "non_arabic_speech")

    if not has_transcription:
        return AudioQualityAssessment(False, "insufficient_speech")

    if metadata is None:
        # Compatibilité avec les doubles de tests et les anciens appelants.
        return AudioQualityAssessment(True, None)

    average_log_probability = metadata.average_log_probability
    has_low_average_log_probability = (
        average_log_probability is not None
        and average_log_probability < MIN_AVERAGE_LOG_PROBABILITY
    )
    has_unstable_temperature = (
        average_log_probability is not None
        and metadata.max_temperature is not None
        and metadata.max_temperature >= HIGH_TEMPERATURE
        and average_log_probability < HIGH_TEMPERATURE_MAX_LOG_PROBABILITY
    )
    has_suspicious_compression = (
        average_log_probability is not None
        and metadata.max_compression_ratio is not None
        and metadata.max_compression_ratio > HIGH_COMPRESSION_RATIO
        and average_log_probability < HIGH_COMPRESSION_MAX_LOG_PROBABILITY
    )

    if (
        has_low_average_log_probability
        or has_unstable_temperature
        or has_suspicious_compression
    ):
        # Ces métriques Whisper indiquent un décodage fragile, pas l'absence
        # de preuve coranique. Le matching reste autorisé, mais uniquement un
        # résultat confiant pourra alors exposer un passage.
        return AudioQualityAssessment(
            True,
            None,
            warning_reason="low_transcription_confidence",
        )

    return AudioQualityAssessment(True, None)


def build_audio_rejection_outcome(
    rejection_reason: AudioRejectionReason,
) -> VerseDetectionOutcome:
    return VerseDetectionOutcome(
        verse=None,
        status="insufficient",
        score=None,
        score_margin=None,
        matched_word_count=0,
        rejection_reason=rejection_reason,
    )


def detect_verse_after_audio_quality_check(
    segments,
    *,
    include_ambiguous_verse: bool,
) -> VerseDetectionOutcome:
    quality = assess_audio_quality(segments)

    if not quality.accepted:
        if quality.rejection_reason is None:
            raise RuntimeError("A rejected audio quality assessment requires a reason.")

        logger.info(
            "Verse detection skipped after audio quality check: reason=%s",
            quality.rejection_reason,
        )
        return build_audio_rejection_outcome(quality.rejection_reason)

    detection = detect_verse_with_metadata(
        segments,
        include_ambiguous_verse=(
            include_ambiguous_verse
            and quality.warning_reason != "low_transcription_confidence"
        ),
    )
    if quality.warning_reason == "non_arabic_speech":
        return apply_language_conflict_gate(detection)
    return detection


def apply_language_conflict_gate(
    detection: VerseDetectionOutcome,
) -> VerseDetectionOutcome:
    """Exige une preuve coranique longue lors d'un conflit de langue."""
    has_supported_quran_evidence = (
        detection.verse is not None
        and detection.score is not None
        and detection.score >= LANGUAGE_CONFLICT_MIN_QURAN_SIMILARITY
        and detection.matched_word_count >= LANGUAGE_CONFLICT_MIN_MATCHED_WORD_COUNT
        and detection.status in {"confident", "ambiguous"}
    )
    if has_supported_quran_evidence or detection.verse is None:
        return detection

    return replace(
        detection,
        verse=None,
        status="insufficient",
        rejection_reason="non_arabic_speech",
    )


def detect_verse_progressively(
    audio_path: str,
    audio_duration_seconds: float | None,
    allow_ambiguous_result: bool = True,
) -> tuple[
    TranscriptionResult | list[dict],
    VerseDetectionOutcome,
    float | None,
    int,
]:
    # /recognize receives a complete audio file, including microphone snapshots.
    # Re-clipping that file here could accept a confident first verse and discard
    # the rest of a longer recitation (for example Al-Fatiha 1:1-7).
    primary_segments = transcribe_audio(audio_path)
    primary_detection = detect_verse_after_audio_quality_check(
        primary_segments,
        include_ambiguous_verse=allow_ambiguous_result,
    )

    primary_metadata = getattr(primary_segments, "metadata", None)
    primary_speech_duration = (
        primary_metadata.speech_duration_seconds
        if primary_metadata is not None
        else None
    )
    if not should_run_transcription_rescue(
        primary_detection,
        speech_duration_seconds=primary_speech_duration,
    ):
        return (
            primary_segments,
            primary_detection,
            audio_duration_seconds,
            1,
        )

    rescue_segments = transcribe_rescue_audio(audio_path, primary_metadata)
    rescue_detection = detect_verse_after_audio_quality_check(
        rescue_segments,
        include_ambiguous_verse=allow_ambiguous_result,
    )
    primary_quality = detection_outcome_quality(primary_detection)
    rescue_quality = detection_outcome_quality(rescue_detection)
    selected_segments, selected_detection = (
        (rescue_segments, rescue_detection)
        if rescue_quality > primary_quality
        else (primary_segments, primary_detection)
    )

    return (
        selected_segments,
        selected_detection,
        audio_duration_seconds,
        2,
    )


def detection_outcome_quality(outcome: VerseDetectionOutcome) -> tuple:
    """Compare deux passes sans promouvoir une hypothèse moins sûre."""
    status_rank = {
        "insufficient": 0,
        "probable": 1,
        "ambiguous": 2,
        "confident": 3,
    }
    return (
        status_rank[outcome.status],
        outcome.verse is not None,
        outcome.score if outcome.score is not None else -1.0,
        outcome.score_margin if outcome.score_margin is not None else -1.0,
        outcome.matched_word_count,
    )


def should_run_transcription_rescue(
    outcome: VerseDetectionOutcome,
    *,
    speech_duration_seconds: float | None = None,
) -> bool:
    if outcome.status == "confident":
        return False

    if outcome.rejection_reason == "insufficient_speech":
        return False

    if outcome.rejection_reason == "non_arabic_speech":
        return (
            speech_duration_seconds is not None
            and speech_duration_seconds
            >= LANGUAGE_CONFLICT_RESCUE_MIN_SPEECH_SECONDS
        )

    # Une proposition ambiguë déjà forte reste explicitement présentée comme
    # telle. Une seconde passe coûteuse a peu de chances de la départager.
    if (
        outcome.status == "ambiguous"
        and outcome.verse is not None
        and outcome.score is not None
        and outcome.score >= MAX_AMBIGUOUS_RESCUE_SCORE
    ):
        return False

    return True


def compute_imam_status(
    predictions,
    detect_imam: bool = True,
    unavailable: bool = False,
) -> ImamStatus:
    """
    Détermine le niveau de confiance.
    """
    if not detect_imam:
        return "disabled"

    if unavailable:
        return "unavailable"

    if not predictions:
        return "unknown"

    top_score = predictions[0]["score"]

    if top_score >= 0.85:
        return "high"

    if top_score >= 0.65:
        return "medium"

    return "low"


def build_recognition_decision_signals(
    segments,
    verse_detection: VerseDetectionOutcome,
    *,
    analyzed_duration_seconds: float | None,
    analysis_attempts: int,
) -> dict[str, object]:
    """Construit une trace exploitable sans journaliser le contenu récité."""
    metadata: TranscriptionMetadata | None = getattr(segments, "metadata", None)
    quality = assess_audio_quality(segments)
    verse = verse_detection.verse

    return {
        "segmentCount": len(segments),
        "transcriptionChars": sum(
            len(segment.get("text", "").strip()) for segment in segments
        ),
        "language": metadata.language if metadata else None,
        "languageProbability": metadata.language_probability if metadata else None,
        "arabicProbability": metadata.arabic_probability if metadata else None,
        "averageLogProbability": (
            metadata.average_log_probability if metadata else None
        ),
        "averageNoSpeechProbability": (
            metadata.average_no_speech_probability if metadata else None
        ),
        "maxCompressionRatio": metadata.max_compression_ratio if metadata else None,
        "maxTemperature": metadata.max_temperature if metadata else None,
        "speechDurationSeconds": metadata.speech_duration_seconds if metadata else None,
        "audioQualityWarning": quality.warning_reason,
        "detectionStatus": verse_detection.status,
        "detectionScore": verse_detection.score,
        "scoreMargin": verse_detection.score_margin,
        "matchedWordCount": verse_detection.matched_word_count,
        "rejectionReason": verse_detection.rejection_reason,
        "verseFound": verse is not None,
        "predictedSurahId": verse.get("sourate_id") if verse else None,
        "analyzedDurationSeconds": analyzed_duration_seconds,
        "analysisAttempts": analysis_attempts,
        "candidateCount": len(verse_detection.candidates),
        "topCandidates": list(verse_detection.candidates),
    }


def run_inference_pipeline(
    audio_path: str,
    detect_imam: bool = True,
    audio_duration_seconds: float | None = None,
    allow_ambiguous_result: bool = True,
    request_id: str | None = None,
):
    """
    Pipeline principal
    """

    # 1️⃣ transcription whisper
    segments, verse_detection, analyzed_duration_seconds, analysis_attempts = (
        detect_verse_progressively(
            audio_path,
            audio_duration_seconds,
            allow_ambiguous_result=allow_ambiguous_result,
        )
    )

    transcription_text = " ".join(
        segment.get("text", "").strip()
        for segment in segments
    ).strip()

    # 2️⃣ détection verset
    verse = verse_detection.verse

    # 3️⃣ détection imam
    imam_predictions = []
    imam_unavailable = False

    audio_was_rejected = verse_detection.rejection_reason in AUDIO_REJECTION_REASONS

    if detect_imam and not audio_was_rejected:
        try:
            imam_predictions = predict_imam(audio_path)
        except ImamPredictionError:
            imam_unavailable = True

    # 4️⃣ calcul statut
    imam_status = compute_imam_status(
        imam_predictions,
        detect_imam=detect_imam,
        unavailable=imam_unavailable,
    )

    logger.info(
        "Inference complete: segments=%s transcription_chars=%s verse_found=%s verse_similarity=%s analyzed_duration_seconds=%s analysis_attempts=%s imam_predictions=%s imam_status=%s",
        len(segments),
        len(transcription_text),
        verse is not None,
        verse.get("similarity") if verse else None,
        analyzed_duration_seconds,
        analysis_attempts,
        len(imam_predictions),
        imam_status,
    )

    decision_signals = build_recognition_decision_signals(
        segments,
        verse_detection,
        analyzed_duration_seconds=analyzed_duration_seconds,
        analysis_attempts=analysis_attempts,
    )
    if request_id is not None:
        decision_signals["requestId"] = request_id
    log_api_event(
        message="Recognition decision",
        route="/recognize",
        extra=decision_signals,
    )

    return {
        "transcription_text": transcription_text,
        "verse": verse,
        "detection": {
            **verse_detection.metadata(),
            "analyzed_duration_seconds": analyzed_duration_seconds,
            "analysis_attempts": analysis_attempts,
        },
        "imam_predictions": imam_predictions,
        "imam_status": imam_status,
        "imam_detection_enabled": detect_imam,
        # Le schéma HTTP public ignore ce champ. Le benchmark backend l'utilise
        # pour expliquer les décisions sans conserver le contenu récité.
        "recognition_diagnostics": decision_signals,
    }
