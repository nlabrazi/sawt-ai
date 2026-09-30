#!/usr/bin/env python3
"""Terminal smoke test: French query -> E5 IDs -> live official source."""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.hadeethenc_client import HadeethEncClient, to_hadith_result
from app.services.hadith_index import HadithIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("queries", nargs="+")
    args = parser.parse_args()
    index = HadithIndex()
    client = HadeethEncClient(index.config.base_url)
    for query in args.queries:
        started = perf_counter()
        results = []
        for hadith_id, score in index.rank(query):
            result = to_hadith_result(client.get_hadith(hadith_id, index.config.language), index.config.language)
            results.append({**result.model_dump(), "debug_cosine": score})
        print(json.dumps({"query": query, "elapsed_ms": round((perf_counter() - started) * 1000), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
