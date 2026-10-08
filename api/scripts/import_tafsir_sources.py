#!/usr/bin/env python3
"""Fetch a private Arabic tafsir pilot via Quran Foundation Content Sync."""

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.services.quran_foundation_client import QuranFoundationClient
from app.services.tafsir_generation_service import TafsirGenerationError
from app.services.tafsir_source_import import build_source_archive, read_archive_batch
from app.services.tafsir_sources import TafsirSourceError, get_tafsir_source, resolve_quran_foundation_resource
from scripts.import_tafsir_fr import DEFAULT_IMPORT_DIRECTORY


def write_source_archive(output_path, archive):
    read_archive_batch(archive)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_path.parent,
            prefix=f".{output_path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(archive, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_path, output_path)
    except FileExistsError as exc:
        raise TafsirSourceError("Le fichier source existe déjà ; choisir un nouveau fichier.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def import_source(source, output_path):
    if output_path.exists():
        raise TafsirSourceError("Le fichier source existe déjà ; aucun appel Quran Foundation effectué.")
    definition = get_tafsir_source(source)
    load_quran_catalog()
    client = QuranFoundationClient()
    catalog = client.get("/api/v4/resources/tafsirs?language=en")
    resources = catalog.get("tafsirs")
    if isinstance(resources, list) and not any(
        isinstance(row, dict) and type(row.get("id")) is int and row["id"] == definition.resource_id
        for row in resources
    ):
        hint = (
            "Demander l'accès de production dans la Developer Console, puis configurer "
            "les identifiants de production avec QF_ENV=production."
            if client.environment == "prelive"
            else "Vérifier la disponibilité de cet ouvrage et les permissions dans la Developer Console."
        )
        raise TafsirSourceError(
            f"Source {source} (ressource arabe {definition.resource_id}) absente du catalogue "
            f"{client.environment}. {hint}"
        )
    resource = resolve_quran_foundation_resource(source, catalog)
    snapshot, sync_state = client.bootstrap_tafsir(definition.resource_id)
    archive = build_source_archive(source, resource, snapshot, client.environment, sync_state)
    write_source_archive(output_path, archive)
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("ibn_kathir", "as_saadi"), required=True)
    parser.add_argument("--output", type=Path, help="Nouveau fichier privé ; aucune sortie existante n'est écrasée.")
    args = parser.parse_args()
    output = args.output or DEFAULT_IMPORT_DIRECTORY / "inputs" / f"{args.source}-source.json"
    try:
        archive = import_source(args.source, output)
    except (TafsirSourceError, TafsirGenerationError, OSError, ValueError) as exc:
        print(f"Import source interrompu : {exc}", file=sys.stderr)
        return 1
    batch = archive["generation_batch"]
    print(f"{args.source} : {sum(len(p['ayahs']) for p in batch['passages'])} versets, "
          f"{len(batch['passages'])} passages originaux ; {output} ; aucune traduction DeepL effectuée.")
    if archive["missing_references"]:
        print("Versets sans contenu source : " + ", ".join(archive["missing_references"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
