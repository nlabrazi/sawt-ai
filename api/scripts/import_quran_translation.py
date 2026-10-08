#!/usr/bin/env python3
"""Import only the French Quran pilot, preserving QuranEnc source responses."""

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.schemas.quran import QuranTranslation
from app.services.quran_translation_service import (
    DEFAULT_TRANSLATION_PATH,
    QuranTranslationServiceError,
    build_translation_index,
    validate_verse_range,
)

API_BASE = "https://quranenc.com/api/v1"
TRANSLATION_KEY = "french_rashid"
TRANSLATOR = "Rachid Maach"
METADATA_URL = f"{API_BASE}/translations/list/fr?localization=fr"
TERMS_URL = "https://quranenc.com/en/home/api"
TIMEOUT_SECONDS = 20
PILOT_REFERENCES = (
    *((1, ayah) for ayah in range(1, 8)),
    *((2, ayah) for ayah in range(1, 6)),
    (2, 255),
)


class QuranTranslationImportError(Exception):
    pass


def verse_url(surah_id: int, ayah: int) -> str:
    return f"{API_BASE}/translation/aya/{TRANSLATION_KEY}/{surah_id}/{ayah}"


def fetch_json(url: str) -> Any:
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Sawt-AI Quran Translation Importer/1.0",
    })
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return json.loads(response.read())
    except (URLError, OSError, ValueError) as exc:
        raise QuranTranslationImportError(
            f"Réponse QuranEnc indisponible ou invalide : {url} ({exc})"
        ) from exc


def translation_metadata(payload: Any) -> dict:
    if not isinstance(payload, dict) or not isinstance(payload.get("translations"), list):
        raise QuranTranslationImportError("Liste des traductions QuranEnc invalide.")
    matches = [
        item for item in payload["translations"]
        if isinstance(item, dict) and item.get("key") == TRANSLATION_KEY
    ]
    if len(matches) != 1:
        raise QuranTranslationImportError("Traduction french_rashid absente ou en doublon.")
    metadata = matches[0]
    if (
        metadata.get("language_iso_code") != "fr"
        or not isinstance(metadata.get("version"), str)
        or not metadata["version"].strip()
        or type(metadata.get("last_update")) is not int
    ):
        raise QuranTranslationImportError("Langue ou version de la traduction invalide.")
    return metadata


def provider_integer(value: Any) -> int:
    if type(value) is int:
        return value
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        return int(value)
    raise QuranTranslationImportError("Identifiant de verset QuranEnc invalide.")


def build_snapshot(metadata_payload: Any, verse_payloads: dict[tuple[int, int], Any]) -> dict:
    metadata = translation_metadata(metadata_payload)
    if set(verse_payloads) != set(PILOT_REFERENCES):
        raise QuranTranslationImportError("Le téléchargement ne correspond pas au pilote complet.")

    translations = []
    source_responses = []
    for surah_id, ayah in PILOT_REFERENCES:
        payload = verse_payloads[(surah_id, ayah)]
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise QuranTranslationImportError(f"Réponse de verset invalide : {surah_id}:{ayah}.")
        reference = (provider_integer(result.get("sura")), provider_integer(result.get("aya")))
        if reference != (surah_id, ayah):
            raise QuranTranslationImportError(f"QuranEnc a retourné un autre verset que {surah_id}:{ayah}.")
        entry = QuranTranslation.model_validate({
            "surah_id": surah_id,
            "ayah": ayah,
            "text": result.get("translation"),
            "source": "quranenc",
            "translator": TRANSLATOR,
            "version": metadata["version"],
            "source_url": verse_url(surah_id, ayah),
            "footnotes": result.get("footnotes"),
        })
        translations.append(entry.model_dump(mode="json"))
        source_responses.append({"url": verse_url(surah_id, ayah), "payload": payload})

    snapshot = {
        "meta": {
            "schema_version": 1,
            "imported_at": datetime.now(timezone.utc).isoformat(),
            "source": "quranenc",
            "translator": TRANSLATOR,
            "translation_key": TRANSLATION_KEY,
            "version": metadata["version"],
            "terms_url": TERMS_URL,
            "metadata_response": {"url": METADATA_URL, "payload": metadata_payload},
        },
        "source_responses": source_responses,
        "translations": translations,
    }
    build_translation_index(snapshot)
    return snapshot


def write_snapshot(output_path: Path, snapshot: dict) -> None:
    # Validate before writing, then replace atomically so failures preserve the old file.
    build_translation_index(snapshot)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(snapshot, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def import_pilot(output_path: Path) -> dict:
    # Load only the existing catalog, without Whisper or any other AI model.
    load_quran_catalog()
    for surah_id, ayah in PILOT_REFERENCES:
        validate_verse_range(surah_id, ayah, ayah)
    metadata_payload = fetch_json(METADATA_URL)
    before = translation_metadata(metadata_payload)
    verse_payloads = {}
    for surah_id, ayah in PILOT_REFERENCES:
        verse_payloads[(surah_id, ayah)] = fetch_json(verse_url(surah_id, ayah))
    after = translation_metadata(fetch_json(METADATA_URL))
    if (before["version"], before["last_update"]) != (after["version"], after["last_update"]):
        raise QuranTranslationImportError("La source a changé pendant l'import ; relancez-le.")
    snapshot = build_snapshot(metadata_payload, verse_payloads)
    write_snapshot(output_path, snapshot)
    return snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_TRANSLATION_PATH)
    args = parser.parse_args()
    try:
        snapshot = import_pilot(args.output)
    except (QuranTranslationImportError, QuranTranslationServiceError, OSError, ValueError) as exc:
        print(f"Import interrompu : {exc}", file=sys.stderr)
        return 1
    print(
        f"{len(snapshot['translations'])} traductions importées ; "
        f"Rachid Maach, version {snapshot['meta']['version']} ; {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
