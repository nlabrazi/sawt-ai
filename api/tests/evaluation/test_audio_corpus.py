import json
import hashlib
from pathlib import Path

import pytest

import evaluation.audio_benchmark.corpus as corpus_module
from evaluation.audio_benchmark.audio import (
    FrenchTtsUnavailableError,
    generate_tone,
    write_pcm16_wav,
)
from evaluation.audio_benchmark.corpus import (
    AudioCorpusError,
    build_audio_corpus,
    load_built_corpus,
)


def write_manifest(path: Path, cases, *, privacy="public", seed=123):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "privacy": privacy,
                "sample_rate": 8_000,
                "seed": seed,
                "cases": cases,
            }
        ),
        encoding="utf-8",
    )


def synthetic_negative(case_id="white_noise"):
    return {
        "id": case_id,
        "label": "negative",
        "category": "noise",
        "expected_verse": None,
        "source": {
            "type": "synthetic",
            "generator": "white_noise",
            "duration_seconds": 0.1,
        },
    }


def test_private_example_never_preconfirms_consent_or_usage_rights():
    example_path = (
        Path(__file__).resolve().parents[2]
        / "evaluation"
        / "audio_corpus.private.example.json"
    )
    payload = json.loads(example_path.read_text(encoding="utf-8"))
    local_sources = []

    for case in payload["cases"]:
        source = case["source"]
        if source.get("type") == "local_file":
            local_sources.append(source)
        for variant in case.get("variants", []):
            noise = variant.get("noise", {})
            if noise.get("type") == "local_file":
                local_sources.append(noise)

    assert local_sources
    assert all(source["consent_confirmed"] is False for source in local_sources)
    assert all(source["usage_rights_confirmed"] is False for source in local_sources)
    assert all("consented" not in case.get("tags", []) for case in payload["cases"])


def test_build_audio_corpus_is_reproducible_and_loadable(tmp_path):
    manifest = tmp_path / "public.json"
    write_manifest(manifest, [synthetic_negative()])

    first_path = build_audio_corpus([manifest], tmp_path / "build-1")
    second_path = build_audio_corpus([manifest], tmp_path / "build-2")
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second_path.read_text(encoding="utf-8"))
    corpus = load_built_corpus(first_path)

    assert first_payload == second_payload
    assert len(corpus.cases) == 1
    assert corpus.cases[0].audio_path.is_file()
    assert corpus.cases[0].duration_seconds == pytest.approx(0.1)
    assert corpus.privacy == "public"
    assert len(corpus.manifest_sha256) == 64
    assert corpus.cases[0].audio_sha256 == first_payload["cases"][0]["sha256"]
    assert first_payload["cases"][0]["sha256"] == second_payload["cases"][0]["sha256"]


def test_private_source_builds_clean_and_noisy_variants_without_leaking_source_path(tmp_path):
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    source_path = private_dir / "person-name-recitation.wav"
    background_path = private_dir / "children-private-background.wav"
    write_pcm16_wav(source_path, generate_tone(0.2, 8_000, frequency_hz=220), 8_000)
    write_pcm16_wav(
        background_path,
        generate_tone(0.1, 8_000, frequency_hz=330),
        8_000,
    )
    manifest = tmp_path / "audio_corpus.private.json"
    write_manifest(
        manifest,
        [
            {
                "id": "fatiha_private",
                "label": "positive",
                "category": "quran_recitation",
                "expected_verse": {
                    "sourate_id": 1,
                    "start_verse": 1,
                    "end_verse": 7,
                },
                "source": {
                    "type": "local_file",
                    "path": str(source_path),
                    "consent_confirmed": True,
                    "usage_rights_confirmed": True,
                    "redistributable": False,
                },
                "variants": [
                    {"id": "clean"},
                    {"id": "white_snr10", "noise": {"type": "white", "snr_db": 10}},
                    {
                        "id": "children_snr5",
                        "noise": {
                            "type": "local_file",
                            "path": str(background_path),
                            "consent_confirmed": True,
                            "usage_rights_confirmed": True,
                            "redistributable": False,
                            "snr_db": 5,
                        },
                    },
                ],
            }
        ],
        privacy="private",
    )

    built_path = build_audio_corpus([manifest], tmp_path / "built")
    built_text = built_path.read_text(encoding="utf-8")
    corpus = load_built_corpus(built_path)

    assert len(corpus.cases) == 3
    assert str(source_path) not in built_text
    assert str(background_path) not in built_text
    assert "person-name" not in built_text
    assert {case.variant["id"] for case in corpus.cases} == {
        "clean",
        "white_snr10",
        "children_snr5",
    }
    assert built_path.stat().st_mode & 0o777 == 0o600
    assert built_path.parent.stat().st_mode & 0o777 == 0o700
    assert all(case.audio_path.stat().st_mode & 0o777 == 0o600 for case in corpus.cases)
    assert {case.source_case_id for case in corpus.cases} == {"fatiha_private"}
    assert corpus.privacy == "private"


def test_local_voice_requires_explicit_consent(tmp_path):
    source_path = tmp_path / "voice.wav"
    write_pcm16_wav(source_path, generate_tone(0.1, 8_000), 8_000)
    manifest = tmp_path / "private.json"
    write_manifest(
        manifest,
        [
            {
                "id": "voice_without_consent",
                "label": "negative",
                "category": "french_speech",
                "expected_verse": None,
                "source": {"type": "local_file", "path": str(source_path)},
            }
        ],
        privacy="private",
    )

    with pytest.raises(AudioCorpusError, match="consent_confirmed=true"):
        build_audio_corpus([manifest], tmp_path / "built")


def test_local_source_requires_explicit_usage_rights(tmp_path):
    source_path = tmp_path / "voice.wav"
    write_pcm16_wav(source_path, generate_tone(0.1, 8_000), 8_000)
    manifest = tmp_path / "private.json"
    write_manifest(
        manifest,
        [
            {
                "id": "voice_without_rights",
                "label": "negative",
                "category": "french_speech",
                "expected_verse": None,
                "source": {
                    "type": "local_file",
                    "path": str(source_path),
                    "consent_confirmed": True,
                },
            }
        ],
        privacy="private",
    )

    with pytest.raises(AudioCorpusError, match="usage_rights_confirmed=true"):
        build_audio_corpus([manifest], tmp_path / "built")


def test_optional_french_tts_is_reported_as_skipped(monkeypatch, tmp_path):
    manifest = tmp_path / "public.json"
    write_manifest(
        manifest,
        [
            {
                "id": "french_tts",
                "label": "negative",
                "category": "french_speech",
                "expected_verse": None,
                "optional": True,
                "source": {"type": "french_tts", "text": "Bonjour à tous."},
            }
        ],
    )
    monkeypatch.setattr(
        corpus_module,
        "render_french_tts",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            FrenchTtsUnavailableError("unavailable")
        ),
    )

    built_path = build_audio_corpus([manifest], tmp_path / "built")
    corpus = load_built_corpus(built_path)

    assert corpus.cases == ()
    assert corpus.skipped == (
        {
            "category": "french_speech",
            "id": "french_tts",
            "reason": "french_tts_unavailable",
        },
    )


def test_built_manifest_rejects_path_traversal(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_rate": 16_000,
                "cases": [
                    {
                        "id": "unsafe_case",
                        "label": "negative",
                        "category": "noise",
                        "audio_path": "../private/voice.wav",
                        "duration_seconds": 1,
                        "expected_verse": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(AudioCorpusError, match="sortir du corpus"):
        load_built_corpus(manifest, require_audio=False)


def test_built_manifest_detects_audio_changed_after_generation(tmp_path):
    manifest = tmp_path / "public.json"
    write_manifest(manifest, [synthetic_negative()])
    built_path = build_audio_corpus([manifest], tmp_path / "built")
    corpus = load_built_corpus(built_path)
    audio_path = corpus.cases[0].audio_path
    audio_path.write_bytes(audio_path.read_bytes() + b"tampered")

    with pytest.raises(AudioCorpusError, match="empreinte audio"):
        load_built_corpus(built_path)


def test_corpus_rejects_audio_above_production_duration(monkeypatch, tmp_path):
    manifest = tmp_path / "public.json"
    write_manifest(manifest, [synthetic_negative()])
    monkeypatch.setattr(corpus_module, "MAX_AUDIO_DURATION_SECONDS", 0.05)

    with pytest.raises(AudioCorpusError, match="limite de production"):
        build_audio_corpus([manifest], tmp_path / "built")


def test_private_build_is_restricted_even_when_a_later_variant_fails(tmp_path):
    source_path = tmp_path / "private-source.wav"
    write_pcm16_wav(source_path, generate_tone(0.1, 8_000), 8_000)
    manifest = tmp_path / "private.json"
    write_manifest(
        manifest,
        [
            {
                "id": "private_recitation",
                "label": "positive",
                "category": "quran_recitation",
                "expected_verse": {
                    "sourate_id": 112,
                    "start_verse": 1,
                    "end_verse": 4,
                },
                "source": {
                    "type": "local_file",
                    "path": str(source_path),
                    "consent_confirmed": True,
                    "usage_rights_confirmed": True,
                    "redistributable": False,
                },
                "variants": [
                    {"id": "clean"},
                    {"id": "broken", "noise": {"type": "unknown", "snr_db": 5}},
                ],
            }
        ],
        privacy="private",
    )
    output_dir = tmp_path / "built"
    output_dir.mkdir(mode=0o755)

    with pytest.raises(AudioCorpusError, match="Type de bruit inconnu"):
        build_audio_corpus([manifest], output_dir)

    clean_path = output_dir / "audio" / "private_recitation--clean.wav"
    assert output_dir.stat().st_mode & 0o777 == 0o700
    assert (output_dir / "audio").stat().st_mode & 0o777 == 0o700
    assert clean_path.stat().st_mode & 0o777 == 0o600
    assert not (output_dir / "manifest.json").exists()
    assert not list((output_dir / "audio").glob("*.tmp"))


def test_preflight_rejects_case_variant_collision_before_writing(tmp_path):
    manifest = tmp_path / "public.json"
    first = synthetic_negative("foo--bar")
    first["variants"] = [{"id": "baz"}]
    second = synthetic_negative("foo")
    second["variants"] = [{"id": "bar--baz"}]
    write_manifest(manifest, [first, second])
    output_dir = tmp_path / "built"

    with pytest.raises(AudioCorpusError, match="Collision case/variante"):
        build_audio_corpus([manifest], output_dir)

    assert not output_dir.exists()


def test_built_manifest_validates_wav_header_and_real_duration(tmp_path):
    source_manifest = tmp_path / "public.json"
    write_manifest(source_manifest, [synthetic_negative()])
    built_path = build_audio_corpus([source_manifest], tmp_path / "built")
    payload = json.loads(built_path.read_text(encoding="utf-8"))

    payload["cases"][0]["duration_seconds"] = 0.2
    built_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AudioCorpusError, match="durée audio"):
        load_built_corpus(built_path)

    audio_path = tmp_path / "built" / payload["cases"][0]["audio_path"]
    audio_path.write_bytes(b"not-a-wav")
    payload["cases"][0]["duration_seconds"] = 0.1
    payload["cases"][0]["sha256"] = hashlib.sha256(b"not-a-wav").hexdigest()
    built_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(AudioCorpusError, match="WAV construit est invalide"):
        load_built_corpus(built_path)
