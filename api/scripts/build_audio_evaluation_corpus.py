#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from evaluation.audio_benchmark.corpus import build_audio_corpus, load_built_corpus

DEFAULT_PUBLIC_MANIFEST = API_DIR / "evaluation" / "audio_corpus.public.json"
DEFAULT_OUTPUT_DIR = API_DIR / "evaluation" / "generated" / "audio"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a reproducible, local-only audio evaluation corpus."
    )
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        help="Source manifest. Repeat to merge public and private cases.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    manifest_paths = args.manifest or [DEFAULT_PUBLIC_MANIFEST]
    built_manifest_path = build_audio_corpus(manifest_paths, args.output_dir)
    corpus = load_built_corpus(built_manifest_path)
    print(
        json.dumps(
            {
                "manifest": str(built_manifest_path),
                "generated_cases": len(corpus.cases),
                "skipped": [dict(item) for item in corpus.skipped],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
