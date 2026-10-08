from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError

import pytest

import app.services.quran_foundation_client as qf
import app.services.tafsir_generation_service as generation
import app.services.tafsir_import_service as french_import
import app.services.tafsir_source_import as source_import
from app.services.tafsir_generation_progress import generation_fingerprint
from app.services.tafsir_sources import TafsirSourceError
from scripts import generate_tafsir_fr as generator
from scripts import import_tafsir_sources as importer

API_DIR = Path(__file__).resolve().parents[2]
SYNC_STATE = {"sync_until_sequence": 990, "next_sync_token": "opaque-checkpoint"}


@pytest.fixture(autouse=True)
def local_configuration(monkeypatch):
    catalog = json.loads((API_DIR / "assets/quran_versets.json").read_text())
    surahs = [{"id": surah["id"], "total_verses": len(surah["verses"])} for surah in catalog]
    monkeypatch.setattr(source_import, "list_surah_metadata", lambda: surahs)
    counts = {surah["id"]: surah for surah in surahs}
    monkeypatch.setattr(generation, "get_surah_metadata", counts.get)
    monkeypatch.setattr(french_import, "get_surah_metadata", counts.get)
    monkeypatch.setattr(importer, "load_quran_catalog", lambda: None)
    monkeypatch.setattr(generator, "load_quran_catalog", lambda: None)
    monkeypatch.setenv("QF_CLIENT_ID", "test-client")
    monkeypatch.setenv("QF_CLIENT_SECRET", "test-secret")
    monkeypatch.setenv("QF_ENV", "prelive")


def fixture_data(source="ibn_kathir"):
    resource_id, slug = (14, "ar-tafsir-ibn-kathir") if source == "ibn_kathir" else (91, "ar-tafseer-al-saddi")
    resource = {"id": resource_id, "slug": slug, "language_name": "arabic", "name": "Édition fictive de test"}
    records = []
    # Fictional text only. The 1:7 anchor covers all Fatiha; Baqara extends beyond the pilot.
    for row_id, surah, start, end, anchor in [(101, 1, 1, 7, 7), (102, 2, 1, 6, 1), (103, 2, 255, 255, 255), (104, 3, 1, 1, 1)]:
        offset = {1: 0, 2: 7, 3: 293}[surah]
        records.append({
            "id": row_id, "resource_id": resource_id, "resource_content_id": resource_id,
            "verse_id": offset + anchor, "verse_key": f"{surah}:{anchor}",
            "group_verse_key_from": f"{surah}:{start}", "group_verse_key_to": f"{surah}:{end}",
            "group_verses_count": end - start + 1,
            "start_verse_id": offset + start, "end_verse_id": offset + end,
            "text": f"<h2>عنوان تجريبي {source}</h2><p>نص خيالي {row_id} &amp; اختبار <strong>كامل</strong>.</p>",
        })
    return resource, {
        "resource_group": "tafsirs", "resource_id": resource_id, "resource_content_id": resource_id,
        "schema_version": 1, "sync_sequence": 999, "records": records,
    }


def archive_for(source="ibn_kathir"):
    resource, snapshot = fixture_data(source)
    return source_import.build_source_archive(source, resource, snapshot, "prelive", SYNC_STATE.copy())


@pytest.mark.parametrize("source", ["ibn_kathir", "as_saadi"])
def test_groups_use_global_range_instead_of_anchor_and_preserve_raw_pilot_only(source):
    archive = archive_for(source)
    batch = archive["generation_batch"]
    assert batch["source"] == source and batch["source_language"] == "ar"
    assert archive["snapshot"]["sync_sequence"] == 999
    assert archive["sync"]["sync_until_sequence"] == 990  # Snapshot may be newer than bootstrap.
    assert [row["id"] for row in archive["snapshot"]["records"]] == [101, 102, 103]
    assert archive["missing_references"] == []
    assert batch["passages"][0]["ayahs"] == list(range(1, 8))
    assert batch["passages"][1]["ayahs"] == list(range(1, 6))
    assert batch["passages"][1]["source_end_ayah"] == 6
    assert "عنوان تجريبي" in batch["passages"][0]["source_text"]
    assert "& اختبار كامل." in batch["passages"][0]["source_text"]
    assert "<h2>" in archive["snapshot"]["records"][0]["text"]
    assert "#record-101" in batch["passages"][0]["source_reference"]
    assert batch["version"].startswith("sha256:")
    assert source_import.read_archive_batch(archive) == batch


def test_missing_and_empty_entries_are_reported_without_completing_tafsir():
    resource, snapshot = fixture_data()
    snapshot["records"][2]["text"] = ""
    # An empty per-verse placeholder must not override the real grouped passage.
    snapshot["records"].append({"id": 105, "resource_id": 14, "resource_content_id": 14, "text": None})
    archive = source_import.build_source_archive("ibn_kathir", resource, snapshot, "prelive", SYNC_STATE)
    assert archive["missing_references"] == ["2:255"]
    assert len(archive["generation_batch"]["passages"]) == 2


@pytest.mark.parametrize("change", [
    lambda resource, snapshot: resource.update(language_name="english"),
    lambda resource, snapshot: resource.update(id=169),
    lambda resource, snapshot: snapshot.update(resource_id=91),
    lambda resource, snapshot: snapshot.update(schema_version=2),
    lambda resource, snapshot: snapshot["records"][0].update(resource_id=91),
    lambda resource, snapshot: snapshot["records"][0].update(group_verse_key_to="1:6"),
    lambda resource, snapshot: snapshot["records"][0].update(group_verses_count=6),
    lambda resource, snapshot: snapshot["records"].append(deepcopy(snapshot["records"][0])),
    lambda resource, snapshot: snapshot["records"].append({**snapshot["records"][0], "id": 200}),
    lambda resource, snapshot: snapshot["records"][0].update(text="<script>bad</script>"),
])
def test_wrong_source_or_ambiguous_range_is_rejected(change):
    resource, snapshot = fixture_data()
    change(resource, snapshot)
    with pytest.raises(TafsirSourceError):
        source_import.build_source_archive("ibn_kathir", resource, snapshot, "prelive", SYNC_STATE)


def fake_requests(monkeypatch, responses):
    requests = []
    results = iter(responses)

    class FakeOpener:
        def open(self, request, timeout):
            requests.append(request)
            result = next(results)
            if isinstance(result, Exception):
                raise result
            return BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(qf, "build_opener", lambda *handlers: FakeOpener())
    return requests


def bootstrap_page(mutations, has_more=False, next_page=None):
    return {"sync": {
        "sync_until_sequence": 990, "has_more": has_more, "next_page_url": next_page,
        "next_sync_token": None if has_more else "opaque-checkpoint", "mutations": mutations,
    }}


def mutation():
    return {"type": "RESOURCE_CREATE", "resource_group": "tafsirs", "resource_id": 14,
            "snapshot_url": "/api/v4/resources/snapshots/tafsirs/14"}


def test_source_import_authenticates_and_completes_pagination_before_atomic_private_write(monkeypatch, tmp_path):
    resource, snapshot = fixture_data()
    requests = fake_requests(monkeypatch, [
        {"access_token": "fake-token"}, {"tafsirs": [resource]},
        bootstrap_page([mutation()], True, "/api/v4/resources/sync?cursor=opaque"),
        bootstrap_page([]), snapshot,
    ])
    output = tmp_path / "ibn_kathir-source.json"
    archive = importer.import_source("ibn_kathir", output)
    assert json.loads(output.read_text()) == archive
    assert output.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))
    assert requests[0].full_url == "https://prelive-oauth2.quran.foundation/oauth2/token"
    assert requests[0].data == b"grant_type=client_credentials&scope=content"
    assert all(request.get_header("User-agent") == qf.USER_AGENT for request in requests)
    assert all(request.get_header("X-auth-token") == "fake-token" for request in requests[1:])
    assert "snapshots/tafsirs/14" in requests[-1].full_url
    assert "fake-token" not in output.read_text() and "test-secret" not in output.read_text()
    assert generator.read_source_batch(output) == archive["generation_batch"]
    before = output.read_bytes()
    with pytest.raises(TafsirSourceError, match="existe déjà"):
        importer.import_source("ibn_kathir", output)
    assert len(requests) == 5 and output.read_bytes() == before


@pytest.mark.parametrize("page", [
    bootstrap_page([mutation()], True, "https://evil.example/api/v4/resources/sync"),
    bootstrap_page([{**mutation(), "snapshot_url": "https://evil.example/snapshot"}]),
    bootstrap_page([{**mutation(), "resource_id": 91}]),
    bootstrap_page([]),
    {"sync": {"has_more": False, "mutations": []}},
])
def test_unavailable_source_or_unsafe_sync_link_never_writes_an_archive(monkeypatch, tmp_path, page):
    resource, _ = fixture_data()
    requests = fake_requests(monkeypatch, [{"access_token": "fake-token"}, {"tafsirs": [resource]}, page])
    output = tmp_path / "source.json"
    with pytest.raises(TafsirSourceError):
        importer.import_source("ibn_kathir", output)
    assert not output.exists()
    assert all("evil.example" not in request.full_url for request in requests)


def test_401_reauthenticates_once_and_redirections_are_disabled(monkeypatch):
    denied = HTTPError("https://apis-prelive.quran.foundation", 401, "Denied", {}, None)
    requests = fake_requests(monkeypatch, [
        {"access_token": "first-token"}, denied, {"access_token": "second-token"}, denied,
    ])
    client = qf.QuranFoundationClient()
    with pytest.raises(TafsirSourceError, match="HTTP 401"):
        client.get("/api/v4/resources/tafsirs")
    assert len(requests) == 4
    assert qf._NoRedirect().redirect_request(None, None, 302, "Found", {}, "https://evil.example") is None


def test_missing_credentials_fail_before_network(monkeypatch):
    monkeypatch.delenv("QF_CLIENT_SECRET")
    monkeypatch.setattr(qf, "build_opener", lambda *args: pytest.fail("Network setup must not occur"))
    with pytest.raises(TafsirSourceError, match="QF_CLIENT_SECRET"):
        qf.QuranFoundationClient()


def test_missing_prelive_original_gives_production_guidance_without_fetching_another_tafsir(monkeypatch, tmp_path):
    requests = fake_requests(monkeypatch, [
        {"access_token": "fake-token"},
        {"tafsirs": [{"id": 169, "name": "Ibn Kathir (Abridged)", "language_name": "english"}]},
    ])
    output = tmp_path / "source.json"
    with pytest.raises(TafsirSourceError, match="ressource arabe 14.*prelive.*QF_ENV=production"):
        importer.import_source("ibn_kathir", output)
    assert len(requests) == 2 and not output.exists()


def test_archive_to_generation_keeps_review_status_and_checkpoint_identity(monkeypatch, tmp_path):
    archive = archive_for()
    path = tmp_path / "source.json"
    importer.write_source_archive(path, archive)
    batch = generation.validate_tafsir_generation_batch(generator.read_source_batch(path))
    fingerprint = generation_fingerprint(batch)
    archive["synced_at"] = "2026-10-08T12:00:00+00:00"
    archive["sync"]["next_sync_token"] = "another-checkpoint"
    path.write_text(json.dumps(archive))
    assert generation_fingerprint(generation.validate_tafsir_generation_batch(generator.read_source_batch(path))) == fingerprint

    calls = []
    def translate(request, timeout):
        calls.append(json.loads(request.data)["text"][0])
        return BytesIO(json.dumps({"translations": [{"text": "Traduction française fictive de test."}]}).encode())
    monkeypatch.setenv("DEEPL_API_KEY", "fake-key")
    monkeypatch.setenv("DEEPL_API_URL", generation.DEEPL_FREE_URL)
    monkeypatch.setattr(generation, "urlopen", translate)
    output, drafts = generator.generate_drafts(path, tmp_path / "drafts.json", tmp_path / "progress.json")
    assert len(calls) == 3 and len(drafts["entries"]) == 13
    assert all(entry["status"] == "need_review" and entry["reviewed_at"] is None for entry in drafts["entries"])
    assert all(entry["version"] == batch.version for entry in drafts["entries"])
    assert output.exists()

    archive["generation_batch"]["passages"][0]["source_text"] = "Texte modifié sans correspondance source."
    path.write_text(json.dumps(archive))
    with pytest.raises(generation.TafsirGenerationError, match="textes originaux"):
        generator.generate_drafts(path, tmp_path / "other-drafts.json", tmp_path / "other-progress.json")
    assert len(calls) == 3  # Reject tampering before any new DeepL call.
