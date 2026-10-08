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
from app.services.tafsir_source_import import read_archive_batch
from app.services.tafsir_sources import PROVIDER, TafsirSourceError
from app.services.tafsir_generation_progress import (
    TafsirProgressError,
    generation_fingerprint,
    load_generation_progress,
)
from scripts.import_tafsir_fr import DEFAULT_IMPORT_DIRECTORY, write_new_snapshot


def read_source_batch(input_path: Path) -> dict:
    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise TafsirGenerationError("Impossible de lire le lot source JSON.") from exc
    load_quran_catalog()
    if isinstance(payload, dict) and payload.get("provider") == PROVIDER:
        try:
            return read_archive_batch(payload)
        except TafsirSourceError as exc:
            raise TafsirGenerationError(str(exc)) from exc
    return payload


def _progress_path(batch, checkpoint_path: Path | None) -> Path:
    return checkpoint_path if checkpoint_path is not None else (
        DEFAULT_IMPORT_DIRECTORY / batch.source / f"progress-{generation_fingerprint(batch)}.json"
    )


def generate_drafts(
    input_path: Path, output_path: Path | None = None, checkpoint_path: Path | None = None,
) -> tuple[Path, dict]:
    if output_path is not None and output_path.exists():
        raise TafsirGenerationError("Le fichier de sortie existe déjà ; aucun appel DeepL effectué.")
    payload = read_source_batch(input_path)
    batch = validate_tafsir_generation_batch(payload)
    checkpoint_path = _progress_path(batch, checkpoint_path)
    if checkpoint_path.resolve() == input_path.resolve() or (
        output_path is not None and checkpoint_path.resolve() == output_path.resolve()
    ):
        raise TafsirGenerationError("La progression doit être distincte du fichier source et du snapshot final.")
    print(f"Progression : {checkpoint_path}")
    snapshot = generate_tafsir_snapshot(payload, checkpoint_path=checkpoint_path)
    if output_path is None:
        timestamp = datetime.fromisoformat(snapshot["imported_at"]).strftime("%Y%m%dT%H%M%S%fZ")
        output_path = DEFAULT_IMPORT_DIRECTORY / snapshot["source"] / f"drafts-{timestamp}.json"
    write_new_snapshot(output_path, snapshot)
    return output_path, snapshot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Passages originaux et provenance.")
    parser.add_argument("--output", type=Path, help="Nouveau snapshot ; aucune sortie n'est écrasée.")
    parser.add_argument("--checkpoint", type=Path, help="Fichier privé de progression ; reprise automatique s'il existe.")
    parser.add_argument("--dry-run", action="store_true", help="Valider et compter sans appel DeepL ni écriture.")
    args = parser.parse_args()
    try:
        if args.dry_run:
            batch = validate_tafsir_generation_batch(read_source_batch(args.input))
            progress = load_generation_progress(_progress_path(batch, args.checkpoint), batch)
            completed = {(entry.surah_id, entry.ayah) for entry in progress.entries}
            pending = [passage for passage in batch.passages
                       if (passage.source_surah_id, passage.ayahs[0]) not in completed]
            ayahs = sum(len(passage.ayahs) for passage in batch.passages)
            characters = sum(len(passage.source_text) for passage in pending)
            print(
                f"{batch.source} : {ayahs} versets, {len(batch.passages)} passages au total ; "
                f"{len(batch.passages) - len(pending)} passages enregistrés, "
                f"{len(pending)} passages restants, {characters} caractères sources restants ; "
                "aucun appel DeepL, aucun fichier créé."
            )
            return 0
        output_path, snapshot = generate_drafts(args.input, args.output, args.checkpoint)
    except KeyboardInterrupt:
        print("Génération interrompue ; les passages sauvegardés seront réutilisés à la reprise.", file=sys.stderr)
        return 130
    except (TafsirGenerationError, TafsirProgressError, TafsirImportError, OSError, ValueError) as exc:
        print(f"Génération interrompue : {exc}", file=sys.stderr)
        return 1
    print(
        f"{len(snapshot['entries'])} brouillons {snapshot['source']} générés ; "
        f"status=need_review ; {output_path} ; import Supabase à effectuer."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
