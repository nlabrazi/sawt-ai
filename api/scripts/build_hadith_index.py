#!/usr/bin/env python3
"""Download the official French API corpus, then build one E5 vector per ID."""

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.core.hadith_config import HadithConfig
from app.services.hadeethenc_client import HadeethEncClient, HadeethEncError
from app.services.hadith_index import load_embedding_model


def build_search_text(payload: dict, categories: dict[str, str]) -> str:
    # Search-only text: the UI always fetches the untouched official source.
    parts = [payload["title"], payload["hadeeth"], payload.get("explanation") or ""]
    parts.extend(hint for hint in payload.get("hints", []) if isinstance(hint, str))
    parts.extend(categories[str(cid)] for cid in payload.get("categories", []) if str(cid) in categories)
    return "passage: " + "\n\n".join(part for part in parts if part.strip())


def fetch_corpus(client, config, cache_dir, *, refresh=False, workers=4):
    cache_dir.mkdir(parents=True, exist_ok=True)
    categories = {str(c["id"]): c["title"] for c in client.list_categories(config.language)}
    ids = sorted(client.iter_hadith_ids(config.language), key=int)
    if not ids:
        raise ValueError("HadeethEnc returned an empty corpus")
    print(f"HadeethEnc: {len(ids)} IDs uniques, {len(categories)} catégories", flush=True)

    def fetch(hadith_id):
        path = cache_dir / f"{hadith_id}.json"
        if path.is_file() and not refresh:
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("language") == config.language and cached.get("base_url") == config.base_url and str(cached["payload"].get("id")) == hadith_id:
                return cached
        for attempt in range(3):
            try:
                payload = client.get_hadith(hadith_id, config.language)
                break
            except HadeethEncError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
        cached = {"language": config.language, "base_url": config.base_url, "fetched_at": datetime.now(timezone.utc).isoformat(), "payload": payload}
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(cached, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
        return cached

    records = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for record in pool.map(fetch, ids):
            records.append(record)
            if len(records) % 100 == 0 or len(records) == len(ids):
                print(f"Fiches: {len(records)}/{len(ids)}", flush=True)
    return records, categories


def save_index(config, records, categories, model, *, batch_size=16):
    import numpy as np

    passages = [build_search_text(record["payload"], categories) for record in records]
    matrix = np.asarray(model.encode(passages, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True), dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records) or not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) <= 0).any():
        raise ValueError("Invalid embeddings; previous index preserved")
    config.index_path.parent.mkdir(parents=True, exist_ok=True)
    config.meta_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_index = config.index_path.with_suffix(".npz.tmp")
    with temporary_index.open("wb") as file:
        np.savez_compressed(file, embeddings=matrix)
    revision = getattr(model[0].auto_model.config, "_commit_hash", None)
    meta = {
        "schema_version": 1, "provider": "HadeethEnc", "base_url": config.base_url,
        "language": config.language, "model": config.model_name, "model_revision": revision,
        "dimension": matrix.shape[1], "built_at": datetime.now(timezone.utc).isoformat(),
        "source_fetched_from": min(r["fetched_at"] for r in records),
        "source_fetched_to": max(r["fetched_at"] for r in records),
        # The documented API exposes no corpus release/version field.
        "corpus_version": None,
        "index_sha256": hashlib.sha256(temporary_index.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
        "items": [{"hadeethenc_id": str(r["payload"]["id"]), "language": config.language} for r in records],
    }
    temporary_meta = config.meta_path.with_suffix(".json.tmp")
    temporary_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # A checksum prevents serving mismatched files if interrupted between replacements.
    temporary_index.replace(config.index_path)
    temporary_meta.replace(config.meta_path)
    print(f"Index: {matrix.shape}, {config.index_path.stat().st_size / 1024**2:.2f} MiB", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=API_DIR / ".cache" / "hadeethenc")
    parser.add_argument("--refresh", action="store_true", help="Refetch all source content instead of resuming cached downloads")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    config = HadithConfig.from_env()
    records, categories = fetch_corpus(HadeethEncClient(config.base_url), config, args.cache_dir / config.language, refresh=args.refresh, workers=args.workers)
    if not args.fetch_only:
        save_index(config, records, categories, load_embedding_model(config.model_name), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
