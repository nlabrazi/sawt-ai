"""Validate local French drafts; no generation or public reads happen here."""

from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from app.schemas.tafsir import TafsirFrenchImportBatch
from app.services.quran_catalog_service import get_surah_metadata

PILOT_REFERENCES = frozenset({
    *((1, ayah) for ayah in range(1, 8)),
    *((2, ayah) for ayah in range(1, 6)),
    (2, 255),
})


class TafsirImportError(Exception):
    pass


def validate_tafsir_import_batch(payload: Any) -> TafsirFrenchImportBatch:
    if not isinstance(payload, dict):
        raise TafsirImportError("Le fichier tafsir doit être un objet JSON.")
    try:
        batch = TafsirFrenchImportBatch.model_validate(payload)
    except ValidationError as exc:
        # Keep field locations useful to the operator without logging source text.
        details = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'lot'} : {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise TafsirImportError(f"Fichier tafsir invalide ; {details}") from exc

    seen = set()
    for entry in batch.entries:
        reference = (entry.surah_id, entry.ayah)
        surah = get_surah_metadata(entry.surah_id)
        if surah is None or entry.source_end_ayah > surah["total_verses"]:
            raise TafsirImportError(f"Passage source absent du catalogue : {reference}.")
        if reference not in PILOT_REFERENCES:
            raise TafsirImportError(f"Verset hors du pilote tafsir : {reference}.")
        if reference in seen:
            raise TafsirImportError(f"Tafsir en doublon pour {reference} et {batch.source}.")
        seen.add(reference)
    return batch


def build_tafsir_import_snapshot(payload: Any) -> dict:
    batch = validate_tafsir_import_batch(payload)
    snapshot = batch.model_dump(mode="json")
    if batch.generation is None:
        snapshot.pop("generation")
    snapshot["entries"].sort(key=lambda entry: (entry["surah_id"], entry["ayah"]))
    snapshot["imported_at"] = datetime.now(timezone.utc).isoformat()
    return snapshot
