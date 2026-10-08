"""Connect the real import, storage services and private/public HTTP routes.

Only Supabase REST and thread dispatch are simulated. These fictitious texts
never enter a real database; SQL trigger protections have their own SQL tests.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from io import BytesIO
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import httpx

from app.main import app
import app.core.model_loader as loader
import app.routes.quran_content as content
import app.routes.tafsir_review as review
import app.services.quran_translation_service as translations
import app.services.tafsir_store as store
from scripts import import_tafsir_fr as importer

API_DIR = Path(__file__).resolve().parents[2]
PASSWORD = "fictitious-pilot-review-password"
HEADERS = {"Authorization": f"Bearer {PASSWORD}"}
PRIVATE_PATH = "/internal/tafsir/2/255/ibn_kathir"
PUBLIC_PARAMS = {"surah_id": 2, "start_verse": 254, "end_verse": 255}


def pilot_batch(source):
    return {
        "schema_version": 1, "source": source, "source_language": "fr",
        "source_edition": f"Édition française fictive de test {source}",
        "version": "test-1", "reuse_reference": "Conditions fictives du corpus de test",
        "entries": [
            {
                "surah_id": surah, "ayah": ayah, "source": source, "version": "test-1",
                "source_reference": f"Référence fictive {source}, {surah}:{ayah}",
                "source_text": f"Brouillon fictif {source}, {surah}:{ayah}, sans contenu religieux.",
                "text_fr": f"Brouillon fictif {source}, {surah}:{ayah}, sans contenu religieux.",
                "source_surah_id": surah, "source_start_ayah": ayah, "source_end_ayah": ayah,
            }
            for surah, ayah in ((1, 1), (2, 1), (2, 255))
        ],
    }


def fake_supabase(monkeypatch):
    rows = []
    requests = []
    clock = datetime(2026, 10, 7, 10, tzinfo=timezone.utc)

    def matches(row, params):
        for field, condition in params:
            if field in {"select", "order", "limit", "offset"}:
                continue
            operator, _, expected = condition.partition(".")
            actual = row[field]
            if field == "updated_at":
                actual = datetime.fromisoformat(actual)
                expected = datetime.fromisoformat(expected)
            elif isinstance(actual, int):
                expected = int(expected)
            if operator == "eq" and actual != expected:
                return False
            if operator == "gte" and actual < expected:
                return False
            if operator == "lte" and actual > expected:
                return False
            assert operator in {"eq", "gte", "lte"}
        return True

    def open_request(request, timeout):
        nonlocal clock
        requests.append(request)
        assert urlsplit(request.full_url).path == "/rest/v1/tafsir_entries"
        assert timeout == store.TIMEOUT_SECONDS
        method = request.get_method()
        clock += timedelta(seconds=1)
        if method == "POST":
            inserted = json.loads(request.data)
            assert request.get_header("Prefer") == "return=minimal"
            assert all(row["status"] == "need_review" and row["reviewed_at"] is None for row in inserted)
            rows.extend({**row, "updated_at": clock.isoformat()} for row in inserted)
            return BytesIO()

        params = parse_qsl(urlsplit(request.full_url).query)
        selected = [row for row in rows if matches(row, params)]
        if method == "PATCH":
            change = json.loads(request.data)
            for row in selected:
                row.update(change)
                row["updated_at"] = clock.isoformat()
                row["reviewed_at"] = clock.isoformat() if row["status"] == "verified" else None
        else:
            assert method == "GET"
        columns = dict(params)["select"].split(",")
        return BytesIO(json.dumps([{field: row[field] for field in columns} for row in selected]).encode())

    monkeypatch.setattr(store, "urlopen", open_request)
    return rows, requests


def test_import_review_and_publication_share_the_exact_source_verse_and_revision(monkeypatch, tmp_path):
    catalog = json.loads((API_DIR / "assets/quran_versets.json").read_text())
    monkeypatch.setattr(loader, "quran_versets", catalog)
    # Catalogue is already loaded; no Whisper or candidate-index setup is needed.
    monkeypatch.setattr(importer, "load_quran_catalog", lambda: None)
    monkeypatch.setenv("SUPABASE_URL", "https://fictitious-project.example.test")
    monkeypatch.setenv("SUPABASE_API_KEY", "sb_secret_fictitious_pilot_key")
    monkeypatch.setenv("TAFSIR_REVIEW_PASSWORD", PASSWORD)
    monkeypatch.setenv("QURAN_TRANSLATION_PATH", str(API_DIR / "assets/quran_translation_fr.json"))

    async def direct_call(operation, *args, **kwargs):
        return operation(*args, **kwargs)
    monkeypatch.setattr(content, "run_in_threadpool", direct_call)
    monkeypatch.setattr(review, "run_in_threadpool", direct_call)
    rows, requests = fake_supabase(monkeypatch)
    translations.clear_quran_translation_cache()

    for source in ("ibn_kathir", "as_saadi"):
        input_path = tmp_path / f"{source}-input.json"
        input_path.write_text(json.dumps(pilot_batch(source)), encoding="utf-8")
        output_path, _ = importer.import_drafts(input_path, tmp_path / f"{source}-snapshot.json")
        assert store.insert_tafsir_import(json.loads(output_path.read_text())) == 3
    assert len(rows) == 6

    async def workflow():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test",
        ) as client:
            async def public_read(expected_sources, expected_text=None):
                response = await client.get("/quran/content", params=PUBLIC_PARAMS)
                assert response.status_code == 200
                assert response.headers["Cache-Control"] == "no-store"
                body = response.json()
                assert body["tafsir_status"] == "available"
                assert body["ayahs"][0] == {"ayah": 254, "translation": None, "tafsirs": []}
                ayah = body["ayahs"][1]
                snapshot = json.loads((API_DIR / "assets/quran_translation_fr.json").read_text())
                translation = next(row for row in snapshot["translations"] if (row["surah_id"], row["ayah"]) == (2, 255))
                assert ayah["translation"] == translation
                assert {entry["source"] for entry in ayah["tafsirs"]} == set(expected_sources)
                assert all((entry["surah_id"], entry["ayah"], entry["status"]) == (2, 255, "verified") for entry in ayah["tafsirs"])
                if expected_text is not None:
                    assert ayah["tafsirs"][0]["text_fr"] == expected_text
                for private in ("PRIVATE_CORRECTION", "provenance", "source_text", "updated_at", "need_review", PASSWORD):
                    assert private not in response.text
                assert all(row["text_fr"] not in response.text for row in rows if row["status"] == "need_review")

            await public_read([])
            before = len(requests)
            denied = await client.post(f"{PRIVATE_PATH}/verify", json={"expected_updated_at": rows[2]["updated_at"]})
            assert denied.status_code == 401 and len(requests) == before
            listing = await client.get("/internal/tafsir", headers=HEADERS, params={"surah_id": 2, "status": "need_review"})
            assert listing.status_code == 200
            assert len(listing.json()) == 4
            kathir = next(row for row in listing.json() if row["ayah"] == 255 and row["source"] == "ibn_kathir")
            original = kathir["provenance"]
            corrected_text = "Correction fictive Ibn Kathir 2:255 après relecture de test."
            saved = await client.patch(PRIVATE_PATH, headers=HEADERS, json={
                "text_fr": corrected_text, "expected_updated_at": kathir["updated_at"],
            })
            assert saved.status_code == 200
            corrected = saved.json()
            assert corrected["status"] == "need_review" and corrected["reviewed_at"] is None
            assert corrected["provenance"] == original
            await public_read([])
            stale = await client.post(f"{PRIVATE_PATH}/verify", headers=HEADERS, json={"expected_updated_at": kathir["updated_at"]})
            assert stale.status_code == 409
            await public_read([])
            verified = await client.post(f"{PRIVATE_PATH}/verify", headers=HEADERS, json={"expected_updated_at": corrected["updated_at"]})
            assert verified.status_code == 200
            assert verified.json()["status"] == "verified" and verified.json()["reviewed_at"] is not None
            assert verified.json()["provenance"] == original
            await public_read(["ibn_kathir"], corrected_text)
            # A validation of 2:255 must not publish 1:1 or 2:1 from the same source.
            other = await client.get("/quran/content", params={"surah_id": 1, "start_verse": 1, "end_verse": 1})
            assert other.status_code == 200 and other.json()["ayahs"][0]["tafsirs"] == []
            saadi_path = "/internal/tafsir/2/255/as_saadi"
            saadi = next(row for row in listing.json() if row["ayah"] == 255 and row["source"] == "as_saadi")
            validated_saadi = await client.post(f"{saadi_path}/verify", headers=HEADERS, json={"expected_updated_at": saadi["updated_at"]})
            assert validated_saadi.status_code == 200
            # This draft's exact text may be exposed only after its own review.
            await public_read(["ibn_kathir", "as_saadi"])
            revised = await client.patch(PRIVATE_PATH, headers=HEADERS, json={
                "text_fr": "PRIVATE_CORRECTION fictive à relire.",
                "expected_updated_at": verified.json()["updated_at"],
            })
            assert revised.status_code == 200
            assert revised.json()["status"] == "need_review" and revised.json()["reviewed_at"] is None
            await public_read(["as_saadi"], saadi["text_fr"])
            public = await client.get("/quran/content", params=PUBLIC_PARAMS)
            assert corrected_text not in public.text and "PRIVATE_CORRECTION" not in public.text
            assert all(row["status"] == "need_review" for row in rows if row["ayah"] != 255)

    try:
        asyncio.run(workflow())
    finally:
        translations.clear_quran_translation_cache()
