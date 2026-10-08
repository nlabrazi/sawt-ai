#!/usr/bin/env python3
"""Insert a validated tafsir pilot snapshot into Supabase without overwriting rows."""

import argparse
import json
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.services.tafsir_import_service import TafsirImportError
from app.services.tafsir_store import TafsirStoreError, insert_tafsir_import


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Snapshot produit par import_tafsir_fr.py ou generate_tafsir_fr.py.")
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        load_quran_catalog()
        count = insert_tafsir_import(payload)
    except (TafsirImportError, TafsirStoreError, OSError, ValueError) as exc:
        print(f"Stockage interrompu : {exc}", file=sys.stderr)
        return 1
    print(f"{count} tafsirs enregistrés dans Supabase ; status=need_review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
