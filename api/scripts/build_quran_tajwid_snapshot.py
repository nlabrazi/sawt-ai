#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.alquran.cloud/v1/quran/quran-tajweed"
TIMEOUT_SECONDS = 60

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "api" / "assets" / "quran_tajwid.json"

EXPECTED_SURAH_COUNT = 114


class SnapshotBuildError(Exception):
    pass


def fetch_quran_tajwid() -> dict[str, Any]:
    request = Request(
        API_URL,
        headers={
            "User-Agent": "Sawt-AI Tajwid Snapshot Builder/1.1",
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            raw_body = response.read().decode(charset)
            payload = json.loads(raw_body)
    except HTTPError as exc:
        raise SnapshotBuildError(f"HTTP error {exc.code} while calling {API_URL}") from exc
    except URLError as exc:
        raise SnapshotBuildError(f"Network error while calling {API_URL}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise SnapshotBuildError("Invalid JSON returned by API") from exc

    if payload.get("status") != "OK":
        raise SnapshotBuildError(f"API returned non-OK status: {payload.get('status')}")

    data = payload.get("data")
    if not isinstance(data, dict):
        raise SnapshotBuildError("Missing or invalid 'data' field in API response")

    surahs = data.get("surahs")
    if not isinstance(surahs, list):
        raise SnapshotBuildError("Missing or invalid 'data.surahs' field in API response")

    return payload


def ensure_int(value: Any, field_name: str, context: str) -> int:
    if not isinstance(value, int):
        raise SnapshotBuildError(f"Invalid field '{field_name}' in {context}: expected int, got {type(value).__name__}")
    return value


def ensure_str(value: Any, field_name: str, context: str) -> str:
    if not isinstance(value, str):
        raise SnapshotBuildError(f"Invalid field '{field_name}' in {context}: expected str, got {type(value).__name__}")
    return value


def build_compatible_snapshot(api_payload: dict[str, Any]) -> dict[str, Any]:
    data = api_payload["data"]

    edition = data.get("edition")
    if edition is not None and not isinstance(edition, dict):
        raise SnapshotBuildError("Invalid 'data.edition' field in API response")

    surahs = data["surahs"]
    surahs_out: list[dict[str, Any]] = []
    total_ayahs = 0

    for surah_index, surah in enumerate(surahs, start=1):
        if not isinstance(surah, dict):
            raise SnapshotBuildError(f"Invalid surah at index {surah_index}: expected object")

        surah_number = ensure_int(surah.get("number"), "number", f"surah index {surah_index}")
        ayahs = surah.get("ayahs")

        if not isinstance(ayahs, list):
            raise SnapshotBuildError(f"Invalid field 'ayahs' in surah {surah_number}: expected list")

        ayahs_out: list[dict[str, Any]] = []

        for ayah_index, ayah in enumerate(ayahs, start=1):
            if not isinstance(ayah, dict):
                raise SnapshotBuildError(
                    f"Invalid ayah at surah {surah_number}, ayah index {ayah_index}: expected object"
                )

            number_in_surah = ensure_int(
                ayah.get("numberInSurah"),
                "numberInSurah",
                f"surah {surah_number}, ayah index {ayah_index}",
            )
            text = ensure_str(
                ayah.get("text"),
                "text",
                f"surah {surah_number}, ayah {number_in_surah}",
            ).strip()

            if not text:
                raise SnapshotBuildError(f"Empty tajwid text in surah {surah_number}, ayah {number_in_surah}")

            ayahs_out.append(
                {
                    "numberInSurah": number_in_surah,
                    "text": text,
                }
            )
            total_ayahs += 1

        if not ayahs_out:
            raise SnapshotBuildError(f"Surah {surah_number} has no valid ayahs")

        surahs_out.append(
            {
                "number": surah_number,
                "ayahs": ayahs_out,
            }
        )

    validate_snapshot_shape(surahs_out)

    snapshot = {
        "meta": {
            "snapshot_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source": "alquran.cloud",
            "source_endpoint": API_URL,
            "edition_identifier": edition.get("identifier") if isinstance(edition, dict) else None,
            "edition_name": edition.get("name") if isinstance(edition, dict) else None,
            "total_surahs": len(surahs_out),
            "total_ayahs": total_ayahs,
            "format": "sawt-ai-v1-compatible",
        },
        "data": {
            "surahs": surahs_out,
        },
    }

    return snapshot


def validate_snapshot_shape(surahs_out: list[dict[str, Any]]) -> None:
    if len(surahs_out) != EXPECTED_SURAH_COUNT:
        raise SnapshotBuildError(
            f"Unexpected surah count: expected {EXPECTED_SURAH_COUNT}, got {len(surahs_out)}"
        )

    expected_surah_numbers = list(range(1, EXPECTED_SURAH_COUNT + 1))
    actual_surah_numbers = [surah["number"] for surah in surahs_out]

    if actual_surah_numbers != expected_surah_numbers:
        raise SnapshotBuildError(
            "Surah numbers are not sequential from 1 to 114"
        )

    for surah in surahs_out:
        surah_number = surah["number"]
        ayahs = surah["ayahs"]

        if not ayahs:
            raise SnapshotBuildError(f"Surah {surah_number} has no ayahs")

        expected_ayah_number = 1
        for ayah in ayahs:
            actual_ayah_number = ayah["numberInSurah"]
            if actual_ayah_number != expected_ayah_number:
                raise SnapshotBuildError(
                    f"Unexpected ayah numbering in surah {surah_number}: "
                    f"expected {expected_ayah_number}, got {actual_ayah_number}"
                )
            expected_ayah_number += 1


def resolve_output_path(cli_arg: str | None) -> Path:
    if cli_arg is None:
        return DEFAULT_OUTPUT

    user_path = Path(cli_arg)
    if user_path.is_absolute():
        return user_path

    return (REPO_ROOT / user_path).resolve()


def write_snapshot(output_path: Path, snapshot: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) > 2:
        print("Usage: python build_quran_tajwid_snapshot.py [output_path]")
        return 1

    output_path = resolve_output_path(sys.argv[1] if len(sys.argv) == 2 else None)

    started_at = time.time()

    print("→ Downloading quran-tajweed from alquran.cloud...")
    api_payload = fetch_quran_tajwid()

    print("→ Building Sawt-AI compatible snapshot...")
    snapshot = build_compatible_snapshot(api_payload)

    print(f"→ Writing file: {output_path}")
    write_snapshot(output_path, snapshot)

    elapsed = time.time() - started_at
    print("")
    print("✅ Snapshot generated successfully")
    print(f"   Output: {output_path}")
    print(f"   Surahs: {snapshot['meta']['total_surahs']}")
    print(f"   Ayahs: {snapshot['meta']['total_ayahs']}")
    print(f"   Time: {elapsed:.2f}s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
