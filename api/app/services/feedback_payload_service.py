from __future__ import annotations

from typing import Any

from app.schemas.feedback import FeedbackPayload
from app.services.quran_catalog_service import get_surah_metadata


class FeedbackPayloadValidationError(Exception):
    pass


def _validate_surah_reference(
    *,
    sourate_id: int,
    start_verse: int,
    end_verse: int,
    sourate_name: str | None,
    transliteration: str | None,
    field_name: str,
) -> dict[str, Any]:
    surah = get_surah_metadata(sourate_id)

    if surah is None:
        raise FeedbackPayloadValidationError(
            f"La référence fournie pour {field_name} contient une sourate inconnue."
        )

    if start_verse > surah["total_verses"] or end_verse > surah["total_verses"]:
        raise FeedbackPayloadValidationError(
            f"La plage de versets fournie pour {field_name} est invalide."
        )

    if sourate_name is not None and sourate_name != surah["name"]:
        raise FeedbackPayloadValidationError(
            f"Le nom de sourate fourni pour {field_name} ne correspond pas à l'identifiant choisi."
        )

    if transliteration is not None and transliteration != surah["transliteration"]:
        raise FeedbackPayloadValidationError(
            f"La translittération fournie pour {field_name} ne correspond pas à l'identifiant choisi."
        )

    return surah


def build_feedback_store_payload(payload: FeedbackPayload) -> dict[str, Any]:
    if payload.is_correct and payload.correction is not None:
        raise FeedbackPayloadValidationError(
            "Une correction ne doit pas être fournie pour un feedback positif."
        )

    if not payload.is_correct and payload.correction is None:
        raise FeedbackPayloadValidationError(
            "Une correction est obligatoire quand le résultat est incorrect."
        )

    normalized_payload = payload.model_dump(mode="json")

    if payload.detected_verse is not None:
        detected_surah = _validate_surah_reference(
            sourate_id=payload.detected_verse.sourate_id,
            start_verse=payload.detected_verse.start_verse,
            end_verse=payload.detected_verse.end_verse,
            sourate_name=payload.detected_verse.sourate_name,
            transliteration=payload.detected_verse.transliteration,
            field_name="le verset détecté",
        )
        normalized_payload["detected_verse"]["sourate_name"] = detected_surah["name"]
        normalized_payload["detected_verse"]["transliteration"] = detected_surah["transliteration"]

    if payload.correction is not None:
        correction_surah = _validate_surah_reference(
            sourate_id=payload.correction.sourate_id,
            start_verse=payload.correction.start_verse,
            end_verse=payload.correction.end_verse,
            sourate_name=payload.correction.sourate_name,
            transliteration=payload.correction.transliteration,
            field_name="la correction",
        )
        normalized_payload["correction"]["sourate_name"] = correction_surah["name"]
        normalized_payload["correction"]["transliteration"] = correction_surah["transliteration"]

    return normalized_payload
