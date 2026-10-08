from copy import deepcopy
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError, URLError

import pytest

import app.services.tafsir_generation_service as service
import app.services.tafsir_generation_progress as progress_store
import app.services.tafsir_import_service as importer
import app.services.tafsir_store as store
from scripts import generate_tafsir_fr as generator

API_DIR = Path(__file__).resolve().parents[2]


def source_batch(source="ibn_kathir"):
    # Explicitly fictional originals, unrelated to the actual tafsir corpus.
    return {
        "schema_version": 1, "source": source, "source_language": "ar",
        "source_edition": "Édition fictive de test, sans contenu religieux",
        "version": "test-1", "reuse_reference": "Autorisation fictive du corpus de test",
        "passages": [
            {
                "source_surah_id": surah, "source_start_ayah": start, "source_end_ayah": end,
                "source_reference": f"Édition fictive {source}, passage {surah}:{start}–{end}",
                "source_text": f"  نص تجريبي خيالي لا يتضمن تفسيرًا دينيًا. {source} {surah}:{start}\n  ",
                "ayahs": ayahs,
            }
            for surah, start, end, ayahs in [(1, 6, 7, [7, 6]), (2, 1, 6, [1, 5])]
        ],
    }


def write_input(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def fake_deepl(monkeypatch, results):
    requests = []
    responses = iter(results)

    def open_request(request, timeout):
        requests.append(request)
        assert timeout == service.TIMEOUT_SECONDS
        result = next(responses)
        if isinstance(result, BaseException):
            raise result
        return BytesIO(result if isinstance(result, bytes) else json.dumps(result).encode("utf-8"))

    monkeypatch.setattr(service, "urlopen", open_request)
    return requests


def french_response(text="  Traduction fictive de test.\n  "):
    return {"translations": [{"text": text, "detected_source_language": "AR"}]}


@pytest.fixture(autouse=True)
def local_configuration(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPL_API_KEY", "fake_test_key:fx")
    monkeypatch.delenv("DEEPL_API_URL", raising=False)
    catalog = json.loads((API_DIR / "assets/quran_versets.json").read_text())
    surahs = {surah["id"]: {"total_verses": len(surah["verses"])} for surah in catalog}
    for module in (service, importer, store):
        monkeypatch.setattr(module, "get_surah_metadata", surahs.get)
    monkeypatch.setattr(generator, "load_quran_catalog", lambda: None)
    monkeypatch.setattr(generator, "DEFAULT_IMPORT_DIRECTORY", tmp_path / "private")


def test_generation_preserves_complete_passages_and_sources_and_stores_only_pending_rows(monkeypatch, tmp_path):
    results = [french_response(f"  Traduction fictive {source} du passage {index}.\n  ")
               for source in ("ibn_kathir", "as_saadi") for index in (0, 1)]
    requests = fake_deepl(monkeypatch, results)
    rows = []

    def storage_request(method, params, payload=None):
        if method == "POST":
            rows.extend(payload)
            return None
        # Even if a provider returns pending rows, the public service filters them.
        return [{key: value for key, value in row.items() if key in store.TafsirEntry.model_fields}
                for row in rows if row["surah_id"] == 1]

    monkeypatch.setattr(store, "_request", storage_request)
    for source in ("ibn_kathir", "as_saadi"):
        payload = source_batch(source)
        input_path = tmp_path / f"{source}.json"
        write_input(input_path, payload)
        output_path, snapshot = generator.generate_drafts(input_path)

        assert output_path.parent == tmp_path / "private" / source
        assert output_path.stat().st_mode & 0o777 == 0o600
        assert json.loads(output_path.read_text()) == snapshot
        assert [(entry["surah_id"], entry["ayah"]) for entry in snapshot["entries"]] == [(1, 6), (1, 7), (2, 1), (2, 5)]
        assert all(entry["status"] == "need_review" and entry["reviewed_at"] is None for entry in snapshot["entries"])
        assert snapshot["generation"]["provider"] == "deepl"
        assert datetime.fromisoformat(snapshot["generation"]["generated_at"].replace("Z", "+00:00")).tzinfo is not None

        for index, passage in enumerate(payload["passages"]):
            entries = [entry for entry in snapshot["entries"] if entry["surah_id"] == passage["source_surah_id"]]
            assert all(entry["text_fr"] == f"  Traduction fictive {source} du passage {index}.\n  " for entry in entries)
            for entry in entries:
                assert entry["source"] == source
                for field in ("source_text", "source_reference", "source_start_ayah", "source_end_ayah"):
                    assert entry[field] == passage[field]
        assert store.insert_tafsir_import(snapshot) == 4
        metadata_by_verse = {(entry["surah_id"], entry["ayah"]): entry["generation"] for entry in snapshot["entries"]}
        assert all(row["provenance"]["generation"] == metadata_by_verse[(row["surah_id"], row["ayah"])]
                   for row in rows if row["source"] == source)
        assert all("generation" not in row for row in rows)

    assert len(requests) == 4  # Four whole passages, rather than eight ayah translations.
    assert store.fetch_verified_tafsirs(1, 6, 7) == []
    assert len({(row["surah_id"], row["ayah"], row["source"]) for row in rows}) == 8
    for request, passage in zip(requests, source_batch()["passages"] + source_batch("as_saadi")["passages"]):
        assert request.full_url == "https://api-free.deepl.com/v2/translate"
        assert request.get_header("Authorization") == "DeepL-Auth-Key fake_test_key:fx"
        assert json.loads(request.data) == {
            "text": [passage["source_text"]], "source_lang": "AR", "target_lang": "FR",
            "preserve_formatting": True,
        }
        assert b"fake_test_key" not in request.data


@pytest.mark.parametrize("case", ["out_of_pilot", "wrong_passage", "duplicate_ayah", "duplicate_passage", "mixed_source", "verified"])
def test_invalid_source_batch_is_rejected_before_any_api_call(monkeypatch, tmp_path, case):
    requests = fake_deepl(monkeypatch, [])
    payload = source_batch()
    passage = payload["passages"][1]
    if case == "out_of_pilot":
        passage["ayahs"] = [6]
    elif case == "wrong_passage":
        payload["passages"][0]["source_end_ayah"] = 8  # Al-Fatiha ends at seven.
    elif case == "duplicate_ayah":
        passage["ayahs"] = [1, 1]
    elif case == "duplicate_passage":
        payload["passages"].append(deepcopy(passage))
    elif case == "mixed_source":
        passage["source"] = "as_saadi"
    else:
        passage["status"] = "verified"

    with pytest.raises(service.TafsirGenerationError):
        service.generate_tafsir_snapshot(payload, checkpoint_path=tmp_path / "progress.json")
    assert requests == []


def test_later_oversized_passage_is_rejected_before_translating_the_first(monkeypatch, tmp_path):
    requests = fake_deepl(monkeypatch, [])
    payload = source_batch()
    payload["passages"][1]["source_text"] = "ن" * (service.MAX_REQUEST_BYTES // 2)

    with pytest.raises(service.TafsirGenerationError, match="128 Kio"):
        service.generate_tafsir_snapshot(payload, checkpoint_path=tmp_path / "progress.json")
    assert requests == []


@pytest.mark.parametrize("response", [
    b"invalid JSON", {"translations": []}, {"translations": [{"text": "  "}]},
    {"translations": [{"text": "Texte fictif", "detected_source_language": "EN"}]},
    {"translations": [{"text": "Premier texte fictif"}, {"text": "Second texte fictif"}]},
])
def test_invalid_deepl_response_never_creates_a_snapshot(monkeypatch, tmp_path, response):
    requests = fake_deepl(monkeypatch, [response])
    input_path, output_path = tmp_path / "input.json", tmp_path / "drafts.json"
    write_input(input_path, source_batch())

    with pytest.raises(service.TafsirGenerationError, match="Réponse DeepL invalide"):
        generator.generate_drafts(input_path, output_path)
    assert len(requests) == 1 and not output_path.exists()


@pytest.mark.parametrize("failure", [
    HTTPError("https://api-free.deepl.com", 456, "Quota exceeded", {}, BytesIO(b"private provider details")),
    URLError("private network details"),
    KeyboardInterrupt(),
])
def test_mid_batch_failure_keeps_progress_and_resumes_automatically(monkeypatch, tmp_path, capsys, failure):
    requests = fake_deepl(monkeypatch, [french_response(), failure])
    input_path, output_path = tmp_path / "input.json", tmp_path / "drafts.json"
    write_input(input_path, source_batch())
    monkeypatch.setattr("sys.argv", ["generate_tafsir_fr.py", "--input", str(input_path), "--output", str(output_path)])

    assert generator.main() == (130 if isinstance(failure, KeyboardInterrupt) else 1)
    error = capsys.readouterr().err
    assert "Génération interrompue" in error
    assert "private" not in error and "fake_test_key" not in error
    assert len(requests) == 2 and not output_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))
    checkpoint = next((tmp_path / "private" / "ibn_kathir").glob("progress-*.json"))
    saved = json.loads(checkpoint.read_text())
    assert len(saved["entries"]) == 2
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    assert all(entry["status"] == "need_review" and entry["reviewed_at"] is None for entry in saved["entries"])
    assert "fake_test_key" not in checkpoint.read_text()
    retry_requests = fake_deepl(monkeypatch, [french_response("Texte fictif du passage restant.")])

    assert generator.main() == 0
    assert len(retry_requests) == 1
    assert json.loads(retry_requests[0].data)["text"] == [source_batch()["passages"][1]["source_text"]]
    snapshot = json.loads(output_path.read_text())
    assert all(entry["text_fr"] == french_response()["translations"][0]["text"]
               for entry in snapshot["entries"] if entry["surah_id"] == 1)


def test_existing_destination_is_preserved_without_consuming_quota(monkeypatch, tmp_path):
    requests = fake_deepl(monkeypatch, [])
    input_path, output_path = tmp_path / "input.json", tmp_path / "reviewed.json"
    write_input(input_path, source_batch())
    previous = b'{"reviewed": true}\n'
    output_path.write_bytes(previous)

    with pytest.raises(service.TafsirGenerationError, match="existe déjà"):
        generator.generate_drafts(input_path, output_path)
    assert output_path.read_bytes() == previous and requests == []


@pytest.mark.parametrize("configuration", ["missing_key", "foreign_host"])
def test_configuration_failure_never_sends_a_secret_or_consumes_quota(monkeypatch, tmp_path, configuration):
    requests = fake_deepl(monkeypatch, [])
    if configuration == "missing_key":
        monkeypatch.delenv("DEEPL_API_KEY")
    else:
        monkeypatch.setenv("DEEPL_API_URL", "https://foreign.example.test")

    with pytest.raises(service.TafsirGenerationError, match="DEEPL_API"):
        service.generate_tafsir_snapshot(source_batch(), checkpoint_path=tmp_path / "progress.json")
    assert requests == []


def test_dry_run_validates_and_counts_once_per_passage_without_a_key_or_output(monkeypatch, tmp_path, capsys):
    requests = fake_deepl(monkeypatch, [])
    monkeypatch.delenv("DEEPL_API_KEY")
    payload = source_batch()
    input_path, output_path = tmp_path / "input.json", tmp_path / "drafts.json"
    write_input(input_path, payload)
    monkeypatch.setattr("sys.argv", [
        "generate_tafsir_fr.py", "--input", str(input_path), "--output", str(output_path), "--dry-run",
    ])

    assert generator.main() == 0
    output = capsys.readouterr().out
    assert "4 versets, 2 passages" in output
    assert f"{sum(len(p['source_text']) for p in payload['passages'])} caractères" in output
    assert requests == [] and not output_path.exists()


def test_unfilled_generation_template_fails_without_calling_deepl(monkeypatch, capsys):
    requests = fake_deepl(monkeypatch, [])
    monkeypatch.setattr("sys.argv", [
        "generate_tafsir_fr.py", "--input", str(API_DIR / "examples/tafsir_generation.example.json"), "--dry-run",
    ])

    assert generator.main() == 1
    assert "Lot source invalide" in capsys.readouterr().err
    assert requests == []


def test_quota_resume_counts_only_remaining_text_and_preserves_each_passages_generation(monkeypatch, tmp_path, capsys):
    payload = source_batch()
    input_path, output_path, checkpoint = (tmp_path / name for name in ("source.json", "drafts.json", "progress.json"))
    write_input(input_path, payload)
    requests = fake_deepl(monkeypatch, [
        french_response("  Premier passage fictif sauvegardé.\n"),
        HTTPError("https://api-free.deepl.com", 456, "Quota exceeded", {}, BytesIO()),
    ])
    with pytest.raises(service.TafsirGenerationError, match="Quota DeepL épuisé"):
        generator.generate_drafts(input_path, output_path, checkpoint)
    saved_bytes = checkpoint.read_bytes()
    saved = json.loads(saved_bytes)
    assert len(requests) == 2 and len(saved["entries"]) == 2
    assert not output_path.exists()

    monkeypatch.delenv("DEEPL_API_KEY")
    monkeypatch.setattr("sys.argv", [
        "generate_tafsir_fr.py", "--input", str(input_path), "--checkpoint", str(checkpoint), "--dry-run",
    ])
    capsys.readouterr()
    assert generator.main() == 0
    counts = capsys.readouterr().out
    assert "1 passages enregistrés, 1 passages restants" in counts
    assert f"{len(payload['passages'][1]['source_text'])} caractères sources restants" in counts
    assert len(requests) == 2 and checkpoint.read_bytes() == saved_bytes

    # A manual plan/key change can resume the job without rewriting old provenance.
    monkeypatch.setenv("DEEPL_API_KEY", "new_fake_test_key")
    monkeypatch.setenv("DEEPL_API_URL", "https://api.deepl.com")
    retry_requests = fake_deepl(monkeypatch, [french_response("  Passage restant fictif.\n")])
    _, snapshot = generator.generate_drafts(input_path, output_path, checkpoint)
    assert len(retry_requests) == 1
    assert retry_requests[0].full_url == "https://api.deepl.com/v2/translate"
    assert retry_requests[0].get_header("Authorization") == "DeepL-Auth-Key new_fake_test_key"
    for entry in snapshot["entries"]:
        if entry["surah_id"] == 1:
            original = next(row for row in saved["entries"] if row["ayah"] == entry["ayah"])
            assert entry == original
        else:
            assert entry["generation"]["api_url"] == "https://api.deepl.com"
        assert entry["status"] == "need_review" and entry["reviewed_at"] is None

    monkeypatch.delenv("DEEPL_API_KEY")
    requests = fake_deepl(monkeypatch, [])
    _, cached_snapshot = generator.generate_drafts(input_path, tmp_path / "another-final.json", checkpoint)
    assert cached_snapshot["entries"] == snapshot["entries"]
    assert requests == []


@pytest.mark.parametrize("change", ["source", "version", "original", "targets"])
def test_progress_from_a_changed_source_batch_is_rejected_without_api_calls(monkeypatch, tmp_path, change):
    payload = source_batch()
    input_path, checkpoint = tmp_path / "source.json", tmp_path / "progress.json"
    write_input(input_path, payload)
    fake_deepl(monkeypatch, [french_response(), french_response()])
    generator.generate_drafts(input_path, tmp_path / "first.json", checkpoint)
    previous = checkpoint.read_bytes()
    if change == "source":
        payload["source"] = "as_saadi"
    elif change == "version":
        payload["version"] = "test-2"
    elif change == "original":
        payload["passages"][0]["source_text"] += " Texte fictif modifié."
    else:
        payload["passages"][1]["ayahs"] = [1, 2]
    write_input(input_path, payload)
    requests = fake_deepl(monkeypatch, [])

    with pytest.raises(service.TafsirGenerationError, match="autre lot source"):
        generator.generate_drafts(input_path, tmp_path / "next.json", checkpoint)
    assert requests == [] and checkpoint.read_bytes() == previous


@pytest.mark.parametrize("damage", ["truncated_json", "verified", "partial_group", "wrong_reference", "request_version"])
def test_invalid_progress_is_preserved_and_cannot_be_used_or_retranslated_silently(monkeypatch, tmp_path, damage):
    input_path, checkpoint = tmp_path / "source.json", tmp_path / "progress.json"
    write_input(input_path, source_batch())
    fake_deepl(monkeypatch, [french_response(), french_response()])
    generator.generate_drafts(input_path, tmp_path / "first.json", checkpoint)
    progress = json.loads(checkpoint.read_text())
    if damage == "truncated_json":
        checkpoint.write_text('{"partial":')
    else:
        if damage == "verified":
            progress["entries"][0].update({"status": "verified", "reviewed_at": "2026-10-08T10:00:00+00:00"})
        elif damage == "partial_group":
            progress["entries"].pop(0)
        elif damage == "wrong_reference":
            progress["entries"][0]["source_reference"] = "Autre référence fictive."
        else:
            progress["request_version"] = "unrelated-request-version"
        write_input(checkpoint, progress)
    previous = checkpoint.read_bytes()
    requests = fake_deepl(monkeypatch, [])

    with pytest.raises(service.TafsirGenerationError):
        generator.generate_drafts(input_path, tmp_path / "next.json", checkpoint)
    assert requests == [] and checkpoint.read_bytes() == previous


def test_concurrent_job_cannot_consume_quota_for_the_same_checkpoint(monkeypatch, tmp_path):
    input_path, checkpoint = tmp_path / "source.json", tmp_path / "progress.json"
    payload = source_batch()
    write_input(input_path, payload)
    batch = service.validate_tafsir_generation_batch(payload)
    requests = fake_deepl(monkeypatch, [])

    with progress_store.locked_generation_progress(checkpoint, batch):
        with pytest.raises(service.TafsirGenerationError, match="utilise déjà"):
            generator.generate_drafts(input_path, tmp_path / "final.json", checkpoint)
    assert requests == []


def test_progress_write_failure_preserves_previous_groups_and_stops_further_requests(monkeypatch, tmp_path):
    input_path, checkpoint, output_path = (tmp_path / name for name in ("source.json", "progress.json", "final.json"))
    write_input(input_path, source_batch())
    requests = fake_deepl(monkeypatch, [french_response(), french_response()])
    original_replace = progress_store.os.replace
    writes = []

    def fail_third_write(source, destination):
        writes.append(destination)
        if len(writes) == 3:
            raise OSError("Fictitious disk failure")
        return original_replace(source, destination)

    monkeypatch.setattr(progress_store.os, "replace", fail_third_write)
    with pytest.raises(service.TafsirGenerationError, match="sauvegarder la progression"):
        generator.generate_drafts(input_path, output_path, checkpoint)
    progress = progress_store.load_generation_progress(checkpoint, service.validate_tafsir_generation_batch(source_batch()))
    assert len(progress.entries) == 2 and len(requests) == 2
    assert not output_path.exists() and not list(tmp_path.glob(".*.tmp"))


def test_final_output_failure_can_be_retried_without_retranslating(monkeypatch, tmp_path):
    input_path, checkpoint, output_path = (tmp_path / name for name in ("source.json", "progress.json", "final.json"))
    write_input(input_path, source_batch())
    requests = fake_deepl(monkeypatch, [french_response(), french_response()])
    original_write = generator.write_new_snapshot

    def fail_final_write(*args):
        raise OSError("Fictitious final output failure")

    monkeypatch.setattr(generator, "write_new_snapshot", fail_final_write)
    with pytest.raises(OSError):
        generator.generate_drafts(input_path, output_path, checkpoint)
    assert len(requests) == 2 and not output_path.exists()
    monkeypatch.setattr(generator, "write_new_snapshot", original_write)
    monkeypatch.delenv("DEEPL_API_KEY")
    retry_requests = fake_deepl(monkeypatch, [])

    _, snapshot = generator.generate_drafts(input_path, output_path, checkpoint)
    assert len(snapshot["entries"]) == 4 and retry_requests == []
