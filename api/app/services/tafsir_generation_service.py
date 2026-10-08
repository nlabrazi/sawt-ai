"""Manual pilot translation of supplied originals; never used by public reads."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from app.schemas.tafsir import (
    TafsirDraftImportEntry,
    TafsirGenerationBatch,
    TafsirGenerationMetadata,
    TafsirGenerationProgress,
)
from app.services.tafsir_generation_progress import (
    REQUEST_VERSION,
    TafsirProgressError,
    locked_generation_progress,
    save_generation_progress,
)
from app.services.quran_catalog_service import get_surah_metadata
from app.services.tafsir_import_service import PILOT_REFERENCES, build_tafsir_import_snapshot

DEEPL_FREE_URL = "https://api-free.deepl.com"
DEEPL_PRO_URL = "https://api.deepl.com"
MAX_REQUEST_BYTES = 128 * 1024
TIMEOUT_SECONDS = 90


class TafsirGenerationError(Exception):
    pass


def _translation_body(source_text: str) -> bytes:
    # No supplementary context, glossary, tools, or per-ayah rewriting.
    body = json.dumps({
        "text": [source_text], "source_lang": "AR", "target_lang": "FR",
        "preserve_formatting": True,
    }, ensure_ascii=False).encode("utf-8")
    if len(body) > MAX_REQUEST_BYTES:
        raise TafsirGenerationError(
            "Un passage dépasse la limite DeepL de 128 Kio ; ne pas tronquer le texte original."
        )
    return body


def validate_tafsir_generation_batch(payload: Any) -> TafsirGenerationBatch:
    if not isinstance(payload, dict):
        raise TafsirGenerationError("Le lot source doit être un objet JSON.")
    try:
        batch = TafsirGenerationBatch.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(map(str, error['loc'])) or 'lot'} : {error['msg']}"
            for error in exc.errors(include_input=False, include_url=False)
        )
        raise TafsirGenerationError(f"Lot source invalide ; {details}") from exc

    seen_ayahs, seen_passages = set(), set()
    for passage in batch.passages:
        surah = get_surah_metadata(passage.source_surah_id)
        if surah is None or passage.source_end_ayah > surah["total_verses"]:
            raise TafsirGenerationError("Un passage original est absent du catalogue coranique.")
        key = (passage.source_surah_id, passage.source_start_ayah, passage.source_end_ayah)
        if key in seen_passages:
            raise TafsirGenerationError("Un passage original figure deux fois ; regrouper ses versets.")
        seen_passages.add(key)
        for ayah in passage.ayahs:
            reference = (passage.source_surah_id, ayah)
            if reference not in PILOT_REFERENCES:
                raise TafsirGenerationError(f"Verset hors du pilote tafsir : {reference}.")
            if reference in seen_ayahs:
                raise TafsirGenerationError(f"Verset en doublon : {reference}.")
            seen_ayahs.add(reference)
        # Validate every request size before consuming any translation quota.
        _translation_body(passage.source_text)
    return batch


def _connection() -> tuple[str, str]:
    key = os.getenv("DEEPL_API_KEY", "").strip()
    if not key or any(character.isspace() for character in key):
        raise TafsirGenerationError("Configurer DEEPL_API_KEY uniquement côté backend.")
    api_url = os.getenv("DEEPL_API_URL", "").strip() or DEEPL_FREE_URL
    # Restrict the destination so a misconfiguration cannot send the key elsewhere.
    if api_url not in (DEEPL_FREE_URL, DEEPL_PRO_URL):
        raise TafsirGenerationError(
            "DEEPL_API_URL doit être https://api-free.deepl.com ou https://api.deepl.com."
        )
    return api_url, key


def _translate_passage(source_text: str, api_url: str, key: str) -> str:
    request = Request(
        f"{api_url}/v2/translate", method="POST", data=_translation_body(source_text),
        headers={"Authorization": f"DeepL-Auth-Key {key}",
                 "Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
    except HTTPError as exc:
        # Do not expose the provider's response body, source passages, or key.
        if exc.code == 456:
            raise TafsirGenerationError("Quota DeepL épuisé (HTTP 456) ; reprendre après rétablissement du quota.") from exc
        raise TafsirGenerationError(f"Traduction DeepL interrompue (HTTP {exc.code}).") from exc
    except (URLError, OSError) as exc:
        raise TafsirGenerationError("Impossible de joindre DeepL ; aucun nouvel essai automatique.") from exc
    try:
        result = json.loads(data)
        translations = result["translations"]
        if not isinstance(translations, list) or len(translations) != 1:
            raise ValueError()
        translation = translations[0]
        text_fr = translation["text"]
        if not isinstance(text_fr, str) or not text_fr.strip():
            raise ValueError()
        if translation.get("detected_source_language", "AR") != "AR":
            raise ValueError()
        return text_fr
    except (KeyError, TypeError, ValueError, UnicodeError) as exc:
        raise TafsirGenerationError("Réponse DeepL invalide ; aucun brouillon publié.") from exc


def _generate_snapshot(
    batch: TafsirGenerationBatch, progress: TafsirGenerationProgress,
    checkpoint_path: Path,
) -> dict:
    completed = {(entry.surah_id, entry.ayah): entry for entry in progress.entries}
    pending = [passage for passage in batch.passages
               if (passage.source_surah_id, passage.ayahs[0]) not in completed]
    if pending:
        api_url, key = _connection()
    if not checkpoint_path.exists():
        save_generation_progress(checkpoint_path, progress)
    entries = list(progress.entries)
    for passage in batch.passages:
        if (passage.source_surah_id, passage.ayahs[0]) in completed:
            continue
        text_fr = _translate_passage(passage.source_text, api_url, key)
        metadata = TafsirGenerationMetadata(
            provider="deepl", request_version=REQUEST_VERSION, api_url=api_url,
            target_language="fr", generated_at=datetime.now(timezone.utc),
        )
        original = passage.model_dump(exclude={"ayahs"})
        for ayah in passage.ayahs:
            entries.append(TafsirDraftImportEntry.model_validate({
                **original, "surah_id": passage.source_surah_id, "ayah": ayah,
                "source": batch.source, "version": batch.version, "text_fr": text_fr,
                "status": "need_review", "reviewed_at": None, "generation": metadata,
            }))
        progress = TafsirGenerationProgress(
            batch_sha256=progress.batch_sha256, request_version=REQUEST_VERSION,
            entries=tuple(entries),
        )
        # Persist the whole group before starting the next chargeable request.
        save_generation_progress(checkpoint_path, progress)
    latest_generation = max((entry.generation for entry in entries), key=lambda item: item.generated_at)
    return build_tafsir_import_snapshot({
        **batch.model_dump(exclude={"passages"}), "entries": entries,
        "generation": latest_generation,
    })


def generate_tafsir_snapshot(payload: Any, *, checkpoint_path: Path) -> dict:
    batch = validate_tafsir_generation_batch(payload)
    try:
        with locked_generation_progress(checkpoint_path, batch) as progress:
            return _generate_snapshot(batch, progress, checkpoint_path)
    except TafsirProgressError as exc:
        raise TafsirGenerationError(str(exc)) from exc
