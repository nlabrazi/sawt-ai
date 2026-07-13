#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

API_DIR = Path(__file__).resolve().parents[1]

if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

from app.core.model_loader import load_quran_catalog
from app.services.detection_evaluation_service import (
    evaluate_detection_cases,
    load_evaluation_cases,
)

DEFAULT_CORPUS_PATH = API_DIR / "evaluation" / "verse_detection_corpus.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Quran verse text detection.")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    args = parser.parse_args()

    load_quran_catalog()
    cases = load_evaluation_cases(args.corpus)
    report = evaluate_detection_cases(cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
