from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

import app.services.tafsir_import_service as service
from scripts import import_tafsir_fr as importer

API_DIR = Path(__file__).resolve().parents[2]


def draft(source="ibn_kathir", surah_id=1, ayah=1):
    return {
        "surah_id": surah_id,
        "ayah": ayah,
        "source": source,
        "version": "test-1",
        "source_reference": f"Édition fictive de test, passage {surah_id}:{ayah}",
        "source_text": f"  Passage source fictif de test {source}, {surah_id}:{ayah}.\n  ",
        "source_surah_id": surah_id,
        "source_start_ayah": ayah,
        "source_end_ayah": ayah,
        "text_fr": f"  Brouillon fictif de test {source}, {surah_id}:{ayah}.\n  ",
    }


def batch(source="ibn_kathir"):
    return {
        "schema_version": 1,
        "source": source,
        "source_language": "ar",
        "source_edition": "Édition fictive de test, sans contenu religieux",
        "version": "test-1",
        "reuse_reference": "Référence fictive de test pour le corpus de test",
        "entries": [draft(source, 2, 255), draft(source, 1, 1), draft(source, 2, 1)],
    }


def write_input(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(autouse=True)
def local_catalog(monkeypatch, tmp_path):
    # Use the real Quran's counts without starting the recognition models.
    catalog = json.loads((API_DIR / "assets" / "quran_versets.json").read_text())
    surahs = {surah["id"]: {"total_verses": len(surah["verses"])} for surah in catalog}
    monkeypatch.setattr(service, "get_surah_metadata", surahs.get)
    monkeypatch.setattr(importer, "load_quran_catalog", lambda: None)
    monkeypatch.setattr(importer, "DEFAULT_IMPORT_DIRECTORY", tmp_path / "private")


def test_local_pipeline_keeps_sources_verses_and_original_texts_separate(tmp_path):
    outputs = {}
    for source in ("ibn_kathir", "as_saadi"):
        payload = batch(source)
        input_path = tmp_path / f"{source}-input.json"
        write_input(input_path, payload)
        output_path, snapshot = importer.import_drafts(input_path)
        original_entries = {(entry["surah_id"], entry["ayah"]): entry for entry in payload["entries"]}

        assert output_path.parent == tmp_path / "private" / source
        assert json.loads(output_path.read_text()) == snapshot
        assert [(entry["surah_id"], entry["ayah"]) for entry in snapshot["entries"]] == [(1, 1), (2, 1), (2, 255)]
        for entry in snapshot["entries"]:
            original = original_entries[(entry["surah_id"], entry["ayah"])]
            assert entry == {**original, "status": "need_review", "reviewed_at": None}
        for field in ("source", "source_language", "source_edition", "version", "reuse_reference"):
            assert snapshot[field] == payload[field]
        assert datetime.fromisoformat(snapshot["imported_at"]).utcoffset().total_seconds() == 0
        assert output_path.stat().st_mode & 0o777 == 0o600
        outputs[source] = snapshot

    assert outputs["ibn_kathir"]["entries"][0]["text_fr"] != outputs["as_saadi"]["entries"][0]["text_fr"]


@pytest.mark.parametrize("review_metadata", [
    {"status": "verified", "reviewed_at": "2026-10-07T10:00:00+00:00"},
    {"reviewed_at": "2026-10-07T10:00:00+00:00"},
])
def test_import_claiming_manual_review_is_rejected_without_creating_output(tmp_path, review_metadata):
    payload = batch()
    payload["entries"][0].update(review_metadata)
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "drafts.json"
    write_input(input_path, payload)

    with pytest.raises(service.TafsirImportError):
        importer.import_drafts(input_path, output_path)
    assert not output_path.exists()


@pytest.mark.parametrize("mismatch", [{"source": "as_saadi"}, {"version": "other-edition"}])
def test_mixed_sources_or_versions_are_rejected(mismatch):
    payload = batch()
    payload["entries"][0].update(mismatch)

    with pytest.raises(service.TafsirImportError, match="une seule source et une seule version"):
        service.build_tafsir_import_snapshot(payload)


@pytest.mark.parametrize("mismatch", [
    {"source_surah_id": 2},
    {"source_start_ayah": 2},
    {"source_end_ayah": 8},  # Al-Fatiha has only seven verses.
    {"surah_id": 1, "ayah": 8, "source_start_ayah": 8, "source_end_ayah": 8},
])
def test_draft_must_belong_to_an_existing_original_passage(mismatch):
    payload = batch()
    payload["entries"] = [{**draft(), **mismatch}]

    with pytest.raises(service.TafsirImportError):
        service.build_tafsir_import_snapshot(payload)


def test_duplicate_verse_in_one_source_is_rejected():
    payload = batch()
    payload["entries"].append(deepcopy(payload["entries"][0]))

    with pytest.raises(service.TafsirImportError, match="doublon"):
        service.build_tafsir_import_snapshot(payload)


def test_import_is_limited_to_the_pilot_references():
    payload = batch()
    payload["entries"] = [draft(surah_id=2, ayah=6)]

    with pytest.raises(service.TafsirImportError, match="hors du pilote"):
        service.build_tafsir_import_snapshot(payload)

    payload["entries"] = [draft(surah_id=surah, ayah=ayah) for surah, ayah in service.PILOT_REFERENCES]
    snapshot = service.build_tafsir_import_snapshot(payload)
    assert len(snapshot["entries"]) == 13

    payload["entries"].append(draft(surah_id=2, ayah=7))
    with pytest.raises(service.TafsirImportError):
        service.build_tafsir_import_snapshot(payload)


def test_existing_french_source_is_preserved_without_rewriting():
    payload = batch()
    payload["source_language"] = "fr"
    for entry in payload["entries"]:
        entry["source_text"] = entry["text_fr"]

    snapshot = service.build_tafsir_import_snapshot(payload)
    assert all(entry["source_text"] == entry["text_fr"] for entry in snapshot["entries"])
    assert all(entry["status"] == "need_review" for entry in snapshot["entries"])

    payload["entries"][0]["text_fr"] += " Ajout fictif."
    with pytest.raises(service.TafsirImportError, match="à l'identique"):
        service.build_tafsir_import_snapshot(payload)


def test_grouped_original_passage_is_kept_whole_and_consistent():
    payload = batch()
    passage = "  Passage source fictif couvrant deux versets.\n  "
    payload["entries"] = [
        {
            **draft(ayah=ayah),
            "source_start_ayah": 6,
            "source_end_ayah": 7,
            "source_reference": "Édition fictive de test, groupe 1:6–7",
            "source_text": passage,
        }
        for ayah in (6, 7)
    ]

    snapshot = service.build_tafsir_import_snapshot(payload)
    assert [entry["ayah"] for entry in snapshot["entries"]] == [6, 7]
    assert all(entry["source_text"] == passage for entry in snapshot["entries"])
    assert all((entry["source_start_ayah"], entry["source_end_ayah"]) == (6, 7) for entry in snapshot["entries"])

    payload["entries"][1]["source_text"] = "Autre passage fictif."
    with pytest.raises(service.TafsirImportError, match="deux contenus"):
        service.build_tafsir_import_snapshot(payload)


def test_reimport_never_overwrites_an_existing_reviewed_file(tmp_path):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "existing.json"
    write_input(input_path, batch())
    reviewed = service.build_tafsir_import_snapshot(batch())
    reviewed["entries"][0].update({
        "status": "verified", "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "text_fr": "Texte fictif relu et corrigé pour ce test.",
    })
    write_input(output_path, reviewed)
    previous = output_path.read_bytes()

    with pytest.raises(service.TafsirImportError, match="existe déjà"):
        importer.import_drafts(input_path, output_path)

    assert output_path.read_bytes() == previous
    assert not list(tmp_path.glob(".*.tmp"))


def test_write_failure_leaves_no_partial_snapshot(monkeypatch, tmp_path):
    snapshot = service.build_tafsir_import_snapshot(batch())
    output_path = tmp_path / "drafts.json"

    def failing_dump(payload, file, **kwargs):
        file.write('{"incomplete":')
        raise OSError("Échec d'écriture fictif de test.")

    monkeypatch.setattr(importer.json, "dump", failing_dump)

    with pytest.raises(OSError):
        importer.write_new_snapshot(output_path, snapshot)
    assert not output_path.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_cli_rejects_the_unfilled_template_with_a_clear_error(monkeypatch, tmp_path, capsys):
    output_path = tmp_path / "drafts.json"
    monkeypatch.setattr("sys.argv", [
        "import_tafsir_fr.py", "--input", str(API_DIR / "examples" / "tafsir_import.example.json"),
        "--output", str(output_path),
    ])

    assert importer.main() == 1
    assert "Import interrompu" in capsys.readouterr().err
    assert not output_path.exists()


@pytest.mark.parametrize("input_content", [None, "{invalid json"])
def test_unreadable_or_invalid_input_preserves_the_existing_output(tmp_path, input_content):
    input_path = tmp_path / "input.json"
    output_path = tmp_path / "previous.json"
    if input_content is not None:
        input_path.write_text(input_content)
    previous = b'{"previous": true}\n'
    output_path.write_bytes(previous)

    with pytest.raises(service.TafsirImportError, match="Impossible de lire"):
        importer.import_drafts(input_path, output_path)
    assert output_path.read_bytes() == previous
