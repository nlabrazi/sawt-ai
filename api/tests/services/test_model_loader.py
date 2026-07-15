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


def test_load_quran_catalog_does_not_load_whisper(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "quran_versets.json"
    snapshot_path.write_text(json.dumps(build_quran_payload()), encoding="utf-8")

    monkeypatch.setattr(model_loader, "QURAN_VERSETS_PATH", snapshot_path)
    monkeypatch.setattr(model_loader, "whisper_model", None)
    monkeypatch.setattr(model_loader, "quran_versets", None)
    monkeypatch.setattr(model_loader, "quran_verse_candidates", None)

    model_loader.load_quran_catalog()

    assert model_loader.whisper_model is None
    assert len(model_loader.get_quran_verse_candidates()) == 3


def test_candidate_catalog_includes_complete_seven_verse_passage():
    payload = [
        {
            "id": 1,
            "name": "الفاتحة",
            "transliteration": "Al-Fatihah",
            "verses": [
                {"id": verse_id, "text": f"كلمة اية {verse_id}"}
                for verse_id in range(1, 8)
            ],
        },
    ]

    candidates = model_loader._build_quran_verse_candidates(payload)

    complete_passage = next(
        candidate
        for candidate in candidates
        if candidate.start_verse == 1 and candidate.end_verse == 7
    )
    assert complete_passage.sourate_id == 1


def test_candidate_catalog_keeps_long_surah_growth_bounded():
    verse_count = 100
    payload = [
        {
            "id": 2,
            "name": "سورة طويلة",
            "transliteration": "Long Surah",
            "verses": [
                {"id": verse_id, "text": "كلمة"}
                for verse_id in range(1, verse_count + 1)
            ],
        },
    ]

    candidates = model_loader._build_quran_verse_candidates(payload)
    expected_count = sum(
        verse_count - window_size + 1
        for window_size in range(1, model_loader.MAX_VERSE_DETECTION_WINDOW_SIZE + 1)
    )

    assert len(candidates) == expected_count
    assert max(
        candidate.end_verse - candidate.start_verse + 1
        for candidate in candidates
    ) == model_loader.MAX_VERSE_DETECTION_WINDOW_SIZE


def test_candidate_catalog_caps_multi_verse_passages_by_word_count():
    long_verse = " ".join(["كلمة"] * 40)
    payload = [
        {
            "id": 2,
            "name": "سورة طويلة",
            "transliteration": "Long Surah",
            "verses": [
                {"id": 1, "text": long_verse},
                {"id": 2, "text": long_verse},
            ],
        },
    ]

    candidates = model_loader._build_quran_verse_candidates(payload)

    assert [(candidate.start_verse, candidate.end_verse) for candidate in candidates] == [
        (1, 1),
        (2, 2),
    ]
