from __future__ import annotations

import hashlib
import json
import math
import re
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.core.upload_policy import MAX_AUDIO_DURATION_SECONDS
from evaluation.audio_benchmark.audio import (
    DEFAULT_SAMPLE_RATE,
    AudioGenerationError,
    FrenchTtsUnavailableError,
    generate_background_noise,
    generate_pink_noise,
    generate_silence,
    generate_synthetic_song,
    generate_tone,
    generate_white_noise,
    inspect_pcm16_wav,
    load_local_audio,
    mix_at_snr,
    peak_limit,
    render_french_tts,
    write_pcm16_wav,
)
from evaluation.audio_benchmark.secure_io import atomic_write_text, prepare_directory

SCHEMA_VERSION = 1
SAFE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{1,79}$")
VALID_LABELS = {"positive", "negative"}
VALID_PRIVACY_LEVELS = {"public", "private"}


class AudioCorpusError(ValueError):
    """Le manifeste du corpus audio est invalide."""


@dataclass(frozen=True, slots=True)
class ExpectedVerse:
    sourate_id: int
    start_verse: int
    end_verse: int


@dataclass(frozen=True, slots=True)
class BuiltAudioCase:
    case_id: str
    label: str
    category: str
    audio_path: Path
    duration_seconds: float
    expected_verse: ExpectedVerse | None
    variant: Mapping[str, Any]
    tags: tuple[str, ...]
    source_case_id: str = ""
    audio_sha256: str = ""


@dataclass(frozen=True, slots=True)
class BuiltAudioCorpus:
    manifest_path: Path
    sample_rate: int
    cases: tuple[BuiltAudioCase, ...]
    skipped: tuple[Mapping[str, str], ...]
    manifest_sha256: str = ""
    privacy: str = "public"


def _load_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as manifest_file:
            payload = json.load(manifest_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise AudioCorpusError(f"Impossible de lire le manifeste {path.name}.") from exc

    if not isinstance(payload, dict):
        raise AudioCorpusError("La racine du manifeste doit être un objet JSON.")

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AudioCorpusError(f"schema_version doit valoir {SCHEMA_VERSION}.")

    return payload


def _validate_id(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or SAFE_ID_PATTERN.fullmatch(value) is None:
        raise AudioCorpusError(
            f"{field_name} doit contenir 2 à 80 caractères [a-z0-9_-]."
        )
    return value


def _validate_positive_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AudioCorpusError(f"{field_name} doit être un nombre.")

    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise AudioCorpusError(f"{field_name} doit être strictement positif.")
    return normalized


def _parse_expected_verse(value: Any) -> ExpectedVerse | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise AudioCorpusError("expected_verse doit être un objet ou null.")

    fields = []
    for field_name in ("sourate_id", "start_verse", "end_verse"):
        field_value = value.get(field_name)
        if isinstance(field_value, bool) or not isinstance(field_value, int) or field_value <= 0:
            raise AudioCorpusError(f"expected_verse.{field_name} doit être un entier positif.")
        fields.append(field_value)

    expected = ExpectedVerse(*fields)
    if expected.start_verse > expected.end_verse:
        raise AudioCorpusError("Le début du passage doit précéder sa fin.")
    return expected


def _validate_tags(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(tag, str) or not tag for tag in value):
        raise AudioCorpusError("tags doit être une liste de chaînes non vides.")
    return tuple(value)


def _stable_seed(seed: int, *identifiers: str) -> int:
    material = ":".join((str(seed), *identifiers)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _generate_synthetic_source(
    source: Mapping[str, Any],
    sample_rate: int,
    seed: int,
):
    generator = source.get("generator")
    duration_seconds = _validate_positive_number(
        source.get("duration_seconds"),
        "source.duration_seconds",
    )

    if generator == "silence":
        return generate_silence(duration_seconds, sample_rate)
    if generator == "white_noise":
        return generate_white_noise(duration_seconds, sample_rate, seed=seed)
    if generator == "pink_noise":
        return generate_pink_noise(duration_seconds, sample_rate, seed=seed)
    if generator == "background_noise":
        return generate_background_noise(duration_seconds, sample_rate, seed=seed)
    if generator == "tone":
        frequency_hz = _validate_positive_number(
            source.get("frequency_hz", 440.0),
            "source.frequency_hz",
        )
        return generate_tone(
            duration_seconds,
            sample_rate,
            frequency_hz=frequency_hz,
        )
    if generator == "synthetic_song":
        tempo_bpm = _validate_positive_number(
            source.get("tempo_bpm", 108.0),
            "source.tempo_bpm",
        )
        return generate_synthetic_song(
            duration_seconds,
            sample_rate,
            tempo_bpm=tempo_bpm,
        )

    raise AudioCorpusError(f"Générateur synthétique inconnu : {generator!r}.")


def _resolve_local_source(
    source: Mapping[str, Any],
    manifest_path: Path,
    privacy: str,
) -> Path:
    source_path_value = source.get("path")
    if not isinstance(source_path_value, str) or not source_path_value.strip():
        raise AudioCorpusError("source.path doit référencer un fichier audio local.")

    if source.get("consent_confirmed") is not True:
        raise AudioCorpusError(
            "Toute voix locale doit avoir consent_confirmed=true avant son évaluation."
        )

    if source.get("usage_rights_confirmed") is not True:
        raise AudioCorpusError(
            "Toute source locale doit avoir usage_rights_confirmed=true avant son évaluation."
        )

    if privacy == "public" and source.get("redistributable") is not True:
        raise AudioCorpusError(
            "Un manifeste public ne peut référencer qu'un audio redistribuable."
        )

    source_path = Path(source_path_value)
    if not source_path.is_absolute():
        source_path = manifest_path.parent / source_path
    return source_path.resolve()


def _generate_source(
    source: Mapping[str, Any],
    *,
    sample_rate: int,
    seed: int,
    manifest_path: Path,
    privacy: str,
):
    source_type = source.get("type")

    if source_type == "synthetic":
        return _generate_synthetic_source(source, sample_rate, seed)
    if source_type == "local_file":
        source_path = _resolve_local_source(source, manifest_path, privacy)
        return load_local_audio(source_path, sample_rate)
    if source_type == "french_tts":
        text = source.get("text")
        if not isinstance(text, str):
            raise AudioCorpusError("source.text doit contenir le texte français à synthétiser.")
        return render_french_tts(text, sample_rate)

    raise AudioCorpusError(f"Type de source audio inconnu : {source_type!r}.")


def _fit_noise_duration(samples, frame_count: int):
    if not samples:
        raise AudioGenerationError("La source de bruit locale est vide.")
    if len(samples) >= frame_count:
        return samples[:frame_count]
    return array("f", (samples[index % len(samples)] for index in range(frame_count)))


def _generate_variant_noise(
    noise: Mapping[str, Any],
    duration_seconds: float,
    sample_rate: int,
    seed: int,
    *,
    manifest_path: Path,
    privacy: str,
):
    noise_type = noise.get("type")
    if noise_type == "white":
        return generate_white_noise(duration_seconds, sample_rate, seed=seed)
    if noise_type == "pink":
        return generate_pink_noise(duration_seconds, sample_rate, seed=seed)
    if noise_type == "background":
        return generate_background_noise(duration_seconds, sample_rate, seed=seed)
    if noise_type == "local_file":
        source_path = _resolve_local_source(noise, manifest_path, privacy)
        samples = load_local_audio(source_path, sample_rate)
        return _fit_noise_duration(samples, round(duration_seconds * sample_rate))
    raise AudioCorpusError(f"Type de bruit inconnu : {noise_type!r}.")


def _apply_variant(
    samples,
    variant: Mapping[str, Any],
    *,
    sample_rate: int,
    seed: int,
    manifest_path: Path,
    privacy: str,
):
    noise = variant.get("noise")
    if noise is None:
        return peak_limit(samples)
    if not isinstance(noise, dict):
        raise AudioCorpusError("variant.noise doit être un objet.")

    noise_type = noise.get("type")
    if not isinstance(noise_type, str):
        raise AudioCorpusError("variant.noise.type est requis.")

    snr_db_value = noise.get("snr_db")
    if isinstance(snr_db_value, bool) or not isinstance(snr_db_value, (int, float)):
        raise AudioCorpusError("variant.noise.snr_db doit être un nombre.")
    snr_db = float(snr_db_value)
    duration_seconds = len(samples) / sample_rate
    generated_noise = _generate_variant_noise(
        noise,
        duration_seconds,
        sample_rate,
        seed,
        manifest_path=manifest_path,
        privacy=privacy,
    )
    return mix_at_snr(samples, generated_noise, snr_db)


def _serialize_expected(expected: ExpectedVerse | None) -> dict[str, int] | None:
    if expected is None:
        return None
    return {
        "sourate_id": expected.sourate_id,
        "start_verse": expected.start_verse,
        "end_verse": expected.end_verse,
    }


def _audio_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as audio_file:
        while chunk := audio_file.read(128 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_variants(case: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    variants = case.get("variants", [{"id": "clean"}])
    if not isinstance(variants, list) or not variants:
        raise AudioCorpusError("variants doit être une liste non vide.")

    result = []
    seen_ids = set()
    for variant in variants:
        if not isinstance(variant, dict):
            raise AudioCorpusError("Chaque variante doit être un objet.")
        variant_id = _validate_id(variant.get("id"), "variant.id")
        if variant_id in seen_ids:
            raise AudioCorpusError(f"Variante dupliquée : {variant_id}.")
        seen_ids.add(variant_id)
        result.append(variant)
    return result


def _preflight_case_collisions(
    manifests: Iterable[tuple[Path, Mapping[str, Any], str]],
) -> None:
    seen_case_ids = set()
    seen_built_ids = set()

    for _manifest_path, payload, _privacy in manifests:
        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise AudioCorpusError("cases doit être une liste.")

        for case in cases:
            if not isinstance(case, dict):
                raise AudioCorpusError("Chaque cas doit être un objet.")

            case_id = _validate_id(case.get("id"), "case.id")
            if case_id in seen_case_ids:
                raise AudioCorpusError(f"Cas dupliqué entre les manifestes : {case_id}.")
            seen_case_ids.add(case_id)

            for variant in _validated_variants(case):
                built_id = f"{case_id}--{variant['id']}"
                if built_id in seen_built_ids:
                    raise AudioCorpusError(
                        f"Collision case/variante avant génération : {built_id}."
                    )
                seen_built_ids.add(built_id)


def build_audio_corpus(
    manifest_paths: Iterable[str | Path],
    output_dir: str | Path,
) -> Path:
    """Construit les WAV et un manifeste d'exécution sans chemin de source privée."""
    normalized_manifest_paths = tuple(Path(path).resolve() for path in manifest_paths)
    if not normalized_manifest_paths:
        raise AudioCorpusError("Au moins un manifeste source est requis.")

    loaded_manifests: list[tuple[Path, Mapping[str, Any], str]] = []
    contains_private_sources = False

    for manifest_path in normalized_manifest_paths:
        payload = _load_json(manifest_path)
        privacy = payload.get("privacy")
        if privacy not in VALID_PRIVACY_LEVELS:
            raise AudioCorpusError("privacy doit valoir 'public' ou 'private'.")
        contains_private_sources = contains_private_sources or privacy == "private"
        loaded_manifests.append((manifest_path, payload, privacy))

    _preflight_case_collisions(loaded_manifests)

    destination = Path(output_dir).resolve()
    audio_dir = destination / "audio"
    prepare_directory(
        destination,
        private=contains_private_sources,
        tighten_existing=contains_private_sources,
    )
    prepare_directory(
        audio_dir,
        private=contains_private_sources,
        tighten_existing=contains_private_sources,
    )

    built_cases: list[dict[str, Any]] = []
    skipped_cases: list[dict[str, str]] = []
    seen_case_ids = set()
    corpus_sample_rate: int | None = None

    for manifest_path, payload, privacy in loaded_manifests:

        seed = payload.get("seed", 0)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AudioCorpusError("seed doit être un entier.")

        sample_rate = payload.get("sample_rate", DEFAULT_SAMPLE_RATE)
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
            raise AudioCorpusError("sample_rate doit être un entier positif.")
        if corpus_sample_rate is None:
            corpus_sample_rate = sample_rate
        elif corpus_sample_rate != sample_rate:
            raise AudioCorpusError("Tous les manifestes doivent utiliser le même sample_rate.")

        cases = payload.get("cases")
        if not isinstance(cases, list):
            raise AudioCorpusError("cases doit être une liste.")

        for case in cases:
            if not isinstance(case, dict):
                raise AudioCorpusError("Chaque cas doit être un objet.")

            case_id = _validate_id(case.get("id"), "case.id")
            if case_id in seen_case_ids:
                raise AudioCorpusError(f"Cas dupliqué entre les manifestes : {case_id}.")
            seen_case_ids.add(case_id)

            label = case.get("label")
            if label not in VALID_LABELS:
                raise AudioCorpusError("case.label doit valoir 'positive' ou 'negative'.")

            category = _validate_id(case.get("category"), "case.category")
            expected = _parse_expected_verse(case.get("expected_verse"))
            if label == "positive" and expected is None:
                raise AudioCorpusError(f"Le cas positif {case_id} doit définir expected_verse.")
            if label == "negative" and expected is not None:
                raise AudioCorpusError(f"Le cas négatif {case_id} doit avoir expected_verse=null.")

            tags = _validate_tags(case.get("tags"))
            source = case.get("source")
            if not isinstance(source, dict):
                raise AudioCorpusError(f"Le cas {case_id} doit définir source.")

            optional = case.get("optional", False)
            if not isinstance(optional, bool):
                raise AudioCorpusError("case.optional doit être un booléen.")

            try:
                base_samples = _generate_source(
                    source,
                    sample_rate=sample_rate,
                    seed=_stable_seed(seed, case_id, "source"),
                    manifest_path=manifest_path,
                    privacy=privacy,
                )
                if not base_samples:
                    raise AudioGenerationError("La source audio ne contient aucun échantillon.")
                duration_seconds = len(base_samples) / sample_rate
                if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
                    raise AudioGenerationError(
                        "La source dépasse la limite de production de "
                        f"{MAX_AUDIO_DURATION_SECONDS} secondes."
                    )
                variants = _validated_variants(case)

                for variant in variants:
                    variant_id = variant["id"]
                    output_name = f"{case_id}--{variant_id}.wav"
                    output_path = audio_dir / output_name
                    variant_samples = _apply_variant(
                        base_samples,
                        variant,
                        sample_rate=sample_rate,
                        seed=_stable_seed(seed, case_id, variant_id),
                        manifest_path=manifest_path,
                        privacy=privacy,
                    )
                    write_pcm16_wav(
                        output_path,
                        variant_samples,
                        sample_rate,
                        private=contains_private_sources,
                    )

                    variant_metadata: dict[str, Any] = {"id": variant_id}
                    if isinstance(variant.get("noise"), dict):
                        variant_metadata["noise"] = {
                            "type": variant["noise"].get("type"),
                            "snr_db": float(variant["noise"].get("snr_db")),
                        }

                    built_cases.append(
                        {
                            "id": f"{case_id}--{variant_id}",
                            "source_case_id": case_id,
                            "label": label,
                            "category": category,
                            "audio_path": f"audio/{output_name}",
                            "duration_seconds": len(variant_samples) / sample_rate,
                            "expected_verse": _serialize_expected(expected),
                            "variant": variant_metadata,
                            "tags": list(tags),
                            "sha256": _audio_sha256(output_path),
                        }
                    )
            except FrenchTtsUnavailableError:
                if not optional:
                    raise
                skipped_cases.append(
                    {
                        "id": case_id,
                        "category": category,
                        "reason": "french_tts_unavailable",
                    }
                )
            except AudioGenerationError as exc:
                raise AudioCorpusError(f"Impossible de construire le cas {case_id}: {exc}") from exc

    built_manifest = {
        "schema_version": SCHEMA_VERSION,
        "privacy": "private" if contains_private_sources else "public",
        "sample_rate": corpus_sample_rate or DEFAULT_SAMPLE_RATE,
        "cases": built_cases,
        "skipped": skipped_cases,
    }
    built_manifest_path = destination / "manifest.json"
    atomic_write_text(
        built_manifest_path,
        json.dumps(built_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        private=contains_private_sources,
    )
    return built_manifest_path


def load_built_corpus(path: str | Path, *, require_audio: bool = True) -> BuiltAudioCorpus:
    manifest_path = Path(path).resolve()
    payload = _load_json(manifest_path)
    privacy = payload.get("privacy", "public")
    if privacy not in VALID_PRIVACY_LEVELS:
        raise AudioCorpusError("privacy doit valoir 'public' ou 'private'.")
    sample_rate = payload.get("sample_rate")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or sample_rate <= 0:
        raise AudioCorpusError("sample_rate doit être un entier positif.")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise AudioCorpusError("cases doit être une liste.")

    manifest_dir = manifest_path.parent.resolve()
    cases = []
    seen_ids = set()

    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise AudioCorpusError("Chaque cas construit doit être un objet.")

        case_id = _validate_id(raw_case.get("id"), "case.id")
        if case_id in seen_ids:
            raise AudioCorpusError(f"Cas construit dupliqué : {case_id}.")
        seen_ids.add(case_id)

        label = raw_case.get("label")
        if label not in VALID_LABELS:
            raise AudioCorpusError("case.label doit valoir 'positive' ou 'negative'.")
        category = _validate_id(raw_case.get("category"), "case.category")
        expected = _parse_expected_verse(raw_case.get("expected_verse"))
        if (label == "positive") != (expected is not None):
            raise AudioCorpusError(f"Le label et expected_verse divergent pour {case_id}.")

        audio_path_value = raw_case.get("audio_path")
        if not isinstance(audio_path_value, str) or not audio_path_value:
            raise AudioCorpusError("audio_path doit être un chemin relatif non vide.")
        relative_audio_path = Path(audio_path_value)
        if relative_audio_path.is_absolute():
            raise AudioCorpusError("audio_path doit rester relatif pour ne pas divulguer de chemin privé.")
        audio_path = (manifest_dir / relative_audio_path).resolve()
        if not audio_path.is_relative_to(manifest_dir):
            raise AudioCorpusError("audio_path ne peut pas sortir du corpus construit.")
        if require_audio and not audio_path.is_file():
            raise AudioCorpusError(f"Fichier audio construit absent : {audio_path.name}.")

        duration_seconds = _validate_positive_number(
            raw_case.get("duration_seconds"),
            "duration_seconds",
        )
        variant = raw_case.get("variant", {})
        if not isinstance(variant, dict):
            raise AudioCorpusError("variant doit être un objet.")

        source_case_id = _validate_id(
            raw_case.get("source_case_id", case_id),
            "source_case_id",
        )
        audio_sha256 = raw_case.get("sha256")
        if (
            not isinstance(audio_sha256, str)
            or len(audio_sha256) != 64
            or any(character not in "0123456789abcdef" for character in audio_sha256)
        ):
            raise AudioCorpusError("sha256 doit être une empreinte hexadécimale valide.")
        if require_audio and _audio_sha256(audio_path) != audio_sha256:
            raise AudioCorpusError(f"L'empreinte audio ne correspond plus pour {case_id}.")
        if require_audio:
            try:
                actual_sample_rate, _channel_count, frame_count = inspect_pcm16_wav(
                    audio_path
                )
            except AudioGenerationError as exc:
                raise AudioCorpusError(
                    f"Le WAV construit est invalide pour {case_id}: {exc}"
                ) from exc

            if actual_sample_rate != sample_rate:
                raise AudioCorpusError(
                    f"La fréquence audio ne correspond plus pour {case_id}."
                )
            actual_duration_seconds = frame_count / actual_sample_rate
            if not math.isclose(
                actual_duration_seconds,
                duration_seconds,
                rel_tol=0.0,
                abs_tol=0.5 / actual_sample_rate,
            ):
                raise AudioCorpusError(
                    f"La durée audio ne correspond plus pour {case_id}."
                )

        cases.append(
            BuiltAudioCase(
                case_id=case_id,
                label=label,
                category=category,
                audio_path=audio_path,
                duration_seconds=duration_seconds,
                expected_verse=expected,
                variant=variant,
                tags=_validate_tags(raw_case.get("tags")),
                source_case_id=source_case_id,
                audio_sha256=audio_sha256,
            )
        )

    raw_skipped = payload.get("skipped", [])
    if not isinstance(raw_skipped, list) or any(not isinstance(item, dict) for item in raw_skipped):
        raise AudioCorpusError("skipped doit être une liste d'objets.")

    return BuiltAudioCorpus(
        manifest_path=manifest_path,
        sample_rate=sample_rate,
        cases=tuple(cases),
        skipped=tuple(raw_skipped),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        privacy=privacy,
    )
