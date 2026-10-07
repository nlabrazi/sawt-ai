"""Supabase REST storage and review operations, called only by the backend."""

import base64
import json
from datetime import datetime
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from app.schemas.tafsir import TafsirEntry, TafsirReviewEntry, TafsirSource, TafsirStatus
from app.services.feedback_store import (
    FeedbackStoreConfigError,
    _get_supabase_api_key,
    _get_supabase_url,
)
from app.services.quran_catalog_service import get_surah_metadata
from app.services.tafsir_import_service import validate_tafsir_import_batch
from app.services.tafsir_sources import TafsirSourceError, get_tafsir_source

TABLE = "tafsir_entries"
TIMEOUT_SECONDS = 15
PUBLIC_COLUMNS = ",".join(TafsirEntry.model_fields)
REVIEW_COLUMNS = ",".join(TafsirReviewEntry.model_fields)


class TafsirStoreError(Exception):
    pass


class TafsirStoreConfigError(TafsirStoreError):
    pass


class TafsirStoreConflict(TafsirStoreError):
    pass


def _connection() -> tuple[str, dict[str, str]]:
    try:
        base_url, key = _get_supabase_url(), _get_supabase_api_key()
    except FeedbackStoreConfigError as exc:
        raise TafsirStoreConfigError(str(exc)) from exc
    url = urlsplit(base_url)
    if url.scheme not in ("https", "http") or not url.netloc or url.query or url.fragment or url.username:
        raise TafsirStoreConfigError("SUPABASE_URL doit être une URL de projet HTTP(S).")
    headers = {"apikey": key, "Content-Type": "application/json", "Accept": "application/json"}
    if key.startswith("sb_secret_"):
        # Opaque secret keys go only in apikey, not in a JWT Authorization header.
        return base_url, headers
    try:
        parts = key.split(".")
        if len(parts) != 3 or not all(parts):
            raise ValueError()
        claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=" * (-len(parts[1]) % 4)))
        if not isinstance(claims, dict) or claims.get("role") != "service_role":
            raise ValueError()
    except (ValueError, UnicodeError) as exc:
        raise TafsirStoreConfigError("La clé Supabase doit être sb_secret ou service_role.") from exc
    # This check rejects the wrong key type; Supabase verifies the JWT signature.
    headers["Authorization"] = f"Bearer {key}"
    return base_url, headers


def _request(method: str, params: list[tuple[str, str]], payload: Any = None) -> Any:
    base_url, headers = _connection()
    if method in ("POST", "PATCH"):
        headers["Prefer"] = "return=minimal" if method == "POST" else "return=representation"
    endpoint = f"{base_url}/rest/v1/{TABLE}"
    if params:
        endpoint += "?" + urlencode(params)
    request = Request(
        endpoint, method=method, headers=headers,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None,
    )
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            data = response.read()
    except HTTPError as exc:
        if exc.code == 409:
            raise TafsirStoreConflict("Un tafsir existe déjà ; aucun texte n'a été remplacé.") from exc
        raise TafsirStoreError(f"Stockage tafsir indisponible (HTTP {exc.code}).") from exc
    except (URLError, OSError) as exc:
        raise TafsirStoreError("Impossible de joindre le stockage tafsir.") from exc
    if method == "POST":
        return None
    try:
        rows = json.loads(data)
        if not isinstance(rows, list):
            raise ValueError()
        return rows
    except (ValueError, UnicodeError) as exc:
        raise TafsirStoreError("Réponse du stockage tafsir invalide.") from exc


def _validate_range(surah_id: int, start_ayah: int, end_ayah: int) -> None:
    if any(type(value) is not int for value in (surah_id, start_ayah, end_ayah)):
        raise TafsirStoreError("Les références de versets doivent être des entiers.")
    surah = get_surah_metadata(surah_id)
    if surah is None or not 1 <= start_ayah <= end_ayah <= surah["total_verses"]:
        raise TafsirStoreError("Référence de verset absente du catalogue.")


def _reference_params(surah_id: int, ayah: int, source: TafsirSource) -> list[tuple[str, str]]:
    _validate_range(surah_id, ayah, ayah)
    try:
        get_tafsir_source(source)
    except TafsirSourceError as exc:
        raise TafsirStoreError(str(exc)) from exc
    return [("surah_id", f"eq.{surah_id}"), ("ayah", f"eq.{ayah}"), ("source", f"eq.{source}")]


def insert_tafsir_import(payload: Any) -> int:
    batch = validate_tafsir_import_batch(payload)
    if batch.imported_at is None:
        raise TafsirStoreError("Utiliser un snapshot horodaté produit par import_tafsir_fr.py.")
    common = batch.model_dump(mode="json", exclude={"entries", "source", "version"})
    rows = []
    for entry in batch.entries:
        data = entry.model_dump(mode="json")
        provenance = {**common, **{field: data.pop(field) for field in (
            "source_text", "source_surah_id", "source_start_ayah", "source_end_ayah",
        )}}
        rows.append({**data, "provenance": provenance})
    # One bulk POST is transactional. No upsert or merge can overwrite a review.
    _request("POST", [], rows)
    return len(rows)


def list_tafsirs_for_review(
    *, surah_id: int | None = None, source: TafsirSource | None = None,
    status: TafsirStatus | None = "need_review", limit: int = 50, offset: int = 0,
) -> list[TafsirReviewEntry]:
    if type(limit) is not int or not 1 <= limit <= 100 or type(offset) is not int or offset < 0:
        raise TafsirStoreError("Pagination tafsir invalide.")
    params = [("select", REVIEW_COLUMNS), ("order", "surah_id.asc,ayah.asc,source.asc"),
              ("limit", str(limit)), ("offset", str(offset))]
    if surah_id is not None:
        _validate_range(surah_id, 1, 1)
        params.append(("surah_id", f"eq.{surah_id}"))
    if source is not None:
        try:
            get_tafsir_source(source)
        except TafsirSourceError as exc:
            raise TafsirStoreError(str(exc)) from exc
        params.append(("source", f"eq.{source}"))
    if status is not None:
        if status not in ("need_review", "verified"):
            raise TafsirStoreError("Statut tafsir invalide.")
        params.append(("status", f"eq.{status}"))
    try:
        return [TafsirReviewEntry.model_validate(row) for row in _request("GET", params)]
    except ValidationError as exc:
        raise TafsirStoreError("Entrée interne tafsir invalide.") from exc


def _patch_review(
    surah_id: int, ayah: int, source: TafsirSource, expected_updated_at: datetime,
    payload: dict, expected_status: TafsirStatus,
) -> TafsirReviewEntry:
    params = _reference_params(surah_id, ayah, source)
    try:
        timestamp = TypeAdapter(AwareDatetime).validate_python(expected_updated_at)
    except ValidationError as exc:
        raise TafsirStoreError("La date de la version relue doit inclure un fuseau horaire.") from exc
    params += [("updated_at", f"eq.{timestamp.isoformat()}"), ("select", REVIEW_COLUMNS)]
    if expected_status == "verified":
        params.append(("status", "eq.need_review"))
    rows = _request("PATCH", params, payload)
    if not rows:
        raise TafsirStoreConflict("Le tafsir a changé ou n'est plus à relire ; recharger son contenu.")
    try:
        if len(rows) != 1:
            raise ValueError()
        entry = TafsirReviewEntry.model_validate(rows[0])
        if (entry.surah_id, entry.ayah, entry.source, entry.status) != (surah_id, ayah, source, expected_status):
            raise ValueError()
        return entry
    except (ValidationError, ValueError) as exc:
        raise TafsirStoreError("Résultat de validation ou correction tafsir incohérent.") from exc


def update_tafsir_text(
    surah_id: int, ayah: int, source: TafsirSource, text_fr: str, *, expected_updated_at: datetime,
) -> TafsirReviewEntry:
    if not isinstance(text_fr, str) or not text_fr.strip():
        raise TafsirStoreError("Le texte français ne doit pas être vide.")
    return _patch_review(surah_id, ayah, source, expected_updated_at,
                         {"text_fr": text_fr, "status": "need_review"}, "need_review")


def verify_tafsir(
    surah_id: int, ayah: int, source: TafsirSource, *, expected_updated_at: datetime,
) -> TafsirReviewEntry:
    return _patch_review(surah_id, ayah, source, expected_updated_at, {"status": "verified"}, "verified")


def fetch_verified_tafsirs(surah_id: int, start_ayah: int, end_ayah: int) -> list[TafsirEntry]:
    _validate_range(surah_id, start_ayah, end_ayah)
    params = [("select", PUBLIC_COLUMNS), ("surah_id", f"eq.{surah_id}"),
              ("ayah", f"gte.{start_ayah}"), ("ayah", f"lte.{end_ayah}"),
              ("status", "eq.verified"), ("order", "ayah.asc,source.asc")]
    entries = []
    try:
        for row in _request("GET", params):
            if not isinstance(row, dict):
                raise ValueError()
            if row.get("status") != "verified":
                continue
            entry = TafsirEntry.model_validate(row)
            if entry.surah_id != surah_id or not start_ayah <= entry.ayah <= end_ayah:
                raise ValueError()
            entries.append(entry)
    except (ValidationError, ValueError) as exc:
        raise TafsirStoreError("Entrée publique tafsir incohérente.") from exc
    return entries
