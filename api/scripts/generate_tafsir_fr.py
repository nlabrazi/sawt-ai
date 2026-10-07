#!/usr/bin/env python3
"""Translate a supplied Arabic tafsir pilot into private French need_review drafts."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.services.tafsir_generation_service import (
    TafsirGenerationError,
    generate_tafsir_snapshot,
    validate_tafsir_generation_batch,
)
from app.services.tafsir_import_service import TafsirImportError
from scripts.import_tafsir_fr import DEFAULT_IMPORT_DIRECTORY, write_new_snapshot


def read_source_batch(input_path: Path) -> dict:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TafsirGenerationError("Impossible de lire le lot source JSON.") from exc
    load_quran_catalog()
    return payload


def generate_drafts(input_path: Path, output_path: Path | None = None) -> tuple[Path, dict]:
    if output_path is not None and output_path.exists():
        raise TafsirGenerationError("Le fichier de sortie existe déjà ; aucun appel DeepL effectué.")
    snapshot = generate_tafsir_snapshot(read_source_batch(input_path))
    if output_path is None:
        timestamp = datetime.fromisoformat(snapshot["imported_at"]).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = DEFAULT_IMPORT_DIRECTORY / snapshot["source"] / f"drafts-{timestamp}.json"
    write_new_snapshot(output_path, snapshot)
    return output_path, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Passages originaux et provenance.")
    parser.add_argument("--output", type=Path, help="Nouveau snapshot ; aucune sortie n'est écrasée.")
    parser.add_argument("--dry-run", action="store_true", help="Valider et compter sans appel DeepL ni écriture.")
    args = parser.parse_args()
    try:
        if args.dry_run:
            batch = validate_tafsir_generation_batch(read_source_batch(args.input))
            ayahs = sum(len(passage.ayahs) for passage in batch.passages)
            characters = sum(len(passage.source_text) for passage in batch.passages)
            print(
                f"{batch.source} : {ayahs} versets, {len(batch.passages)} passages, "
                f"{characters} caractères sources ; aucun appel DeepL, aucun fichier créé."
            )
            return 0
        output_path, snapshot = generate_drafts(args.input, args.output)
    except (TafsirGenerationError, TafsirImportError, OSError, ValueError) as exc:
        print(f"Génération interrompue : {exc}", file=sys.stderr)
        return 1
    print(
        f"{len(snapshot['entries'])} brouillons {snapshot['source']} générés ; "
        f"status=need_review ; {output_path} ; import Supabase à effectuer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
