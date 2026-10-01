#!/usr/bin/env python3
"""Recherche terminal : phrase française, résultats classés et liens HadeethEnc."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+")
    parser.add_argument("--variant", choices=("initial", "benchmark", "benchmark-original"), default="initial")
    parser.add_argument("--json", action="store_true", help="Afficher les textes complets en JSON (scores pour les variantes benchmark)")
    parser.add_argument("--raw-query", action="store_true", help="Comparer sans retirer les amorces de recherche")
    args = parser.parse_args()
    if args.raw_query and args.variant == "initial":
        parser.error("--raw-query est réservé aux variantes benchmark pour reproduire leurs scores bruts")
    if any(not query.strip() for query in args.queries):
        parser.error("Écrivez une phrase à rechercher.")
    from app.core.hadith_config import HadithConfig
    from app.services.hadeethenc_client import HadeethEncClient, to_hadith_result
    from app.services.hadith_index import HadithIndex
    from app.services.hadith_query import normalize_hadith_query
    from app.services.hadith_search_service import HadithSearchService

    if args.variant in ("benchmark", "benchmark-original"):
        from scripts.hadith_experiment import HadithExperiment
        strategy = "multi_context" if args.variant == "benchmark" else "multi"
        index = HadithExperiment(strategy=strategy)
        config = HadithConfig.from_env()
        model_label = "E5-base, passages avec contexte" if strategy == "multi_context" else "E5-base, découpage original"
    else:
        index = HadithIndex()
        config = index.config
        model_label = config.model_name + " — recherche API"
    print(f"Chargement : {model_label}. Le premier résultat peut prendre quelques secondes…", file=sys.stderr, flush=True)
    if args.variant != "initial":
        index.load()
    client = HadeethEncClient(config.base_url)
    service = HadithSearchService(config, index=index, client=client)
    for query in args.queries:
        search_query = query if args.raw_query else normalize_hadith_query(query)
        started = perf_counter()
        if args.variant == "initial":
            output = service.search(query).model_dump()
            results = output["results"]
        else:
            results = []
            for hadith_id, score in index.rank(search_query):
                result = to_hadith_result(client.get_hadith(hadith_id, config.language), config.language)
                results.append({**result.model_dump(), "debug_cosine": score})
            output = {"query": query, "search_query": search_query, "results": results, "search_mode": "benchmark"}
        output["elapsed_ms"] = round((perf_counter() - started) * 1000)
        if args.json:
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print(f"\nVotre recherche : {query}\n")
            if search_query != query:
                print(f"Recherche utilisée : {search_query}\n")
            label = {"keywords": "Mots-clés dans les textes français", "semantic": "Propositions par sens",
                     "benchmark": "Diagnostic : voisins sémantiques bruts, sans filtre de mots-clés"}[output["search_mode"]]
            print(label + "\n")
            if not results:
                print("Aucun résultat disponible dans la collection indexée.\n")
            for rank, result in enumerate(results, 1):
                print(f"{rank}. {result['title']}\n   ID : {result['id']}\n   Lire le texte : {result['source_url']}\n")
            print("Ces résultats sont des propositions de recherche. Ouvrez les liens pour vérifier le texte.\n")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"Dépendance Python absente : {exc.name}. Depuis api/, utilisez : bash scripts/search_hadith.sh \"votre recherche\"", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"Recherche impossible : {exc}", file=sys.stderr)
        sys.exit(1)
