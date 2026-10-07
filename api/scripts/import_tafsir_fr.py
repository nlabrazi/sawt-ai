#!/usr/bin/env python3
"""Import a local French tafsir pilot into a new, private need_review file."""

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.services.tafsir_import_service import (
    TafsirImportError,
    build_tafsir_import_snapshot,
    validate_tafsir_import_batch,
)

DEFAULT_IMPORT_DIRECTORY = API_DIR / "data" / "tafsir"


def write_new_snapshot(output_path: Path, snapshot: dict) -> None:
    batch = validate_tafsir_import_batch(snapshot)
    if batch.imported_at is None:
        raise TafsirImportError("La date d'import est requise pour écrire le snapshot.")
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
        # Publish a complete file atomically, failing if the destination exists.
        # Unlike replace(), this also protects a reviewed file from a racing import.
        os.link(temporary_path, output_path)
    except FileExistsError as exc:
        raise TafsirImportError(
            "Le fichier de sortie existe déjà ; choisir un nouveau fichier."
        ) from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def import_drafts(input_path: Path, output_path: Path | None = None) -> tuple[Path, dict]:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TafsirImportError("Impossible de lire le fichier JSON d'entrée.") from exc
    # Reuse the Quran catalog without loading Whisper or the recognition pipeline.
    load_quran_catalog()
    snapshot = build_tafsir_import_snapshot(payload)
    if output_path is None:
        timestamp = datetime.fromisoformat(snapshot["imported_at"]).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = DEFAULT_IMPORT_DIRECTORY / snapshot["source"] / f"drafts-{timestamp}.json"
    write_new_snapshot(output_path, snapshot)
    return output_path, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Lot local avec sa provenance.")
    parser.add_argument("--output", type=Path, help="Nouveau fichier ; une sortie existante est refusée.")
    args = parser.parse_args()
    try:
        output_path, snapshot = import_drafts(args.input, args.output)
    except (TafsirImportError, OSError, ValueError) as exc:
        print(f"Import interrompu : {exc}", file=sys.stderr)
        return 1
    print(
        f"{len(snapshot['entries'])} brouillons {snapshot['source']} importés ; "
        f"status=need_review ; {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
