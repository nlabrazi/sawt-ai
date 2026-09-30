#!/usr/bin/env python3
"""Measure Top-1/Top-3 only against human-reviewed HadeethEnc labels."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

API_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(API_DIR))

from app.services.hadith_index import HadithIndex


def validate_corpus(corpus: dict, *, preview: bool = False) -> list[dict]:
    cases = corpus.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Le corpus doit contenir des requêtes.")
    seen = set()
    for case in cases:
        query = case.get("query", "")
        if not isinstance(query, str) or not 3 <= len(query.strip()) <= 300 or query in seen:
            raise ValueError("Requête invalide ou dupliquée.")
        seen.add(query)
        if preview:
            continue
        ids = case.get("expected_hadeethenc_ids")
        if case.get("review_status") != "human_reviewed" or not case.get("reviewed_by") or not case.get("reviewed_at"):
            raise ValueError("Validation humaine manquante : utilisez --preview pour préparer la relecture, sans calculer d'accuracy.")
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.isascii() or not i.isdigit() for i in ids):
            raise ValueError("IDs attendus non validés.")
    return cases


def evaluate(cases: list[dict], index, *, preview: bool = False) -> dict:
    started = perf_counter()
    index.load()
    load_ms = (perf_counter() - started) * 1000
    # Warm the encoder; latency below includes encoding and ranking, not model loading or HTTP.
    index.rank(cases[0]["query"], 3)
    rows = []
    for case in cases:
        started = perf_counter()
        ids = [hadith_id for hadith_id, _ in index.rank(case["query"], 3)]
        elapsed_ms = (perf_counter() - started) * 1000
        row = {"query": case["query"], "retrieved_ids": ids, "latency_ms": round(elapsed_ms, 2), "source_urls": [f"https://hadeethenc.com/fr/browse/hadith/{i}" for i in ids]}
        if not preview:
            expected = set(case["expected_hadeethenc_ids"])
            row.update(top_1_hit=bool(expected.intersection(ids[:1])), top_3_hit=bool(expected.intersection(ids[:3])))
        rows.append(row)
    report = {"status": "pending_human_review" if preview else "human_reviewed", "queries": len(rows), "model_load_ms": round(load_ms, 2), "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / len(rows), 2), "latency_scope": "warm query embedding + cosine ranking, excluding HadeethEnc HTTP", "results": rows}
    if not preview:
        report["top_1_accuracy"] = sum(r["top_1_hit"] for r in rows) / len(rows)
        report["top_3_accuracy"] = sum(r["top_3_hit"] for r in rows) / len(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=API_DIR / "evaluation" / "hadith_search_corpus.json")
    parser.add_argument("--preview", action="store_true", help="Retrieve candidates for human review; never compute accuracy")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
        cases = validate_corpus(corpus, preview=args.preview)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    report = evaluate(cases, HadithIndex(), preview=args.preview)
    output = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
