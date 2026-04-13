import json

import app.core.model_loader as model_loader


def build_quran_payload():
    return [
        {
            "id": 112,
            "name": "الإخلاص",
            "transliteration": "Al-Ikhlas",
            "verses": [
                {"id": 1, "text": "قُلْ"},
                {"id": 2, "text": "هُوَ اللَّهُ"},
            ],
        },
    ]


def test_load_all_models_precomputes_quran_verse_candidates_once(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "quran_versets.json"
    snapshot_path.write_text(json.dumps(build_quran_payload()), encoding="utf-8")

    original_builder = model_loader._build_quran_verse_candidates
    build_calls = []

    def counted_builder(versets_data):
        build_calls.append(len(versets_data))
        return original_builder(versets_data)

    monkeypatch.setattr(model_loader, "QURAN_VERSETS_PATH", snapshot_path)
    monkeypatch.setattr(model_loader, "whisper_model", object())
    monkeypatch.setattr(model_loader, "quran_versets", None)
    monkeypatch.setattr(model_loader, "quran_verse_candidates", None)
    monkeypatch.setattr(model_loader, "_build_quran_verse_candidates", counted_builder)

    model_loader.load_all_models()
    model_loader.load_all_models()

    candidates = model_loader.get_quran_verse_candidates()

    assert build_calls == [1]
    assert len(candidates) == 3
    assert candidates[0].normalized_text == "قل"
    assert candidates[-1].normalized_text == "قل هو الله"
