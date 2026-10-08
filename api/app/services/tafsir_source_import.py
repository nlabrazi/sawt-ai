"""Map original Content Sync passages to the existing Arabic generation pilot."""

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import re

from app.services.quran_catalog_service import list_surah_metadata
from app.services.quran_foundation_client import ENVIRONMENTS
from app.services.tafsir_generation_service import validate_tafsir_generation_batch
from app.services.tafsir_import_service import PILOT_REFERENCES
from app.services.tafsir_sources import (
    PROVIDER, TERMS_URL, TafsirSourceError, get_tafsir_source,
    resolve_quran_foundation_resource,
)


class _SourceText(HTMLParser):
    """Remove presentation markup, retaining every visible word and heading."""

    BLOCKS = {"p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "tr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            raise TafsirSourceError("Balise active inattendue dans le texte original.")
        if tag in self.BLOCKS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.BLOCKS and self.parts and not self.parts[-1].endswith("\n"):
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def _plain_text(text):
    parser = _SourceText()
    parser.feed(text)
    parser.close()
    return "".join(parser.parts).strip("\n")


def _digest(payload):
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_source_archive(source, resource, snapshot, environment, sync_state):
    definition = get_tafsir_source(source)
    resolve_quran_foundation_resource(source, {"tafsirs": [resource]})
    if environment not in ENVIRONMENTS or not isinstance(resource.get("name"), str) or not resource["name"].strip():
        raise TafsirSourceError("Environnement ou titre de la ressource invalide.")
    if not isinstance(sync_state, dict) or (
        type(sync_state.get("sync_until_sequence")) is not int or sync_state["sync_until_sequence"] < 0
        or not isinstance(sync_state.get("next_sync_token"), str) or not sync_state["next_sync_token"].strip()
    ):
        raise TafsirSourceError("Checkpoint Content Sync absent ou invalide.")
    if not isinstance(snapshot, dict) or (
        snapshot.get("resource_group") != "tafsirs"
        or type(snapshot.get("resource_id")) is not int or snapshot["resource_id"] != definition.resource_id
        or type(snapshot.get("resource_content_id")) is not int or snapshot["resource_content_id"] != definition.resource_id
        or type(snapshot.get("schema_version")) is not int or snapshot["schema_version"] != 1
        or type(snapshot.get("sync_sequence")) is not int or snapshot["sync_sequence"] < 0
        or not isinstance(snapshot.get("records"), list)
    ):
        raise TafsirSourceError("Identité ou format du snapshot Content Sync invalide.")

    # Quran Foundation ranges use global verse IDs, while Sawt-AI uses surah/ayah.
    verse_ids, offset = {}, 0
    for surah in list_surah_metadata():
        for ayah in range(1, surah["total_verses"] + 1):
            verse_ids[(surah["id"], ayah)] = offset + ayah
        offset += surah["total_verses"]
    if len(verse_ids) != 6236 or not PILOT_REFERENCES.issubset(verse_ids):
        raise TafsirSourceError("Catalogue coranique incomplet pour les plages fournisseur.")

    records, passages, covered, seen_ids = [], [], set(), set()
    snapshot_url = ENVIRONMENTS[environment][0] + f"/api/v4/resources/snapshots/tafsirs/{definition.resource_id}"
    for row in snapshot["records"]:
        if not isinstance(row, dict) or (
            type(row.get("id")) is not int or row["id"] <= 0 or row["id"] in seen_ids
            or type(row.get("resource_id")) is not int or row["resource_id"] != definition.resource_id
            or type(row.get("resource_content_id")) is not int or row["resource_content_id"] != definition.resource_id
        ):
            raise TafsirSourceError("Ligne tafsir invalide, dupliquée ou provenant d'un autre ouvrage.")
        seen_ids.add(row["id"])
        text = row.get("text")
        if text is None or isinstance(text, str) and not text.strip():
            continue  # Some providers put empty placeholders beside a grouped passage.
        if not isinstance(text, str) or (
            type(row.get("start_verse_id")) is not int or type(row.get("end_verse_id")) is not int
            or not 1 <= row["start_verse_id"] <= row["end_verse_id"] <= 6236
        ):
            raise TafsirSourceError("Texte ou plage globale d'un tafsir invalide.")
        targets = sorted(reference for reference in PILOT_REFERENCES
                         if row["start_verse_id"] <= verse_ids[reference] <= row["end_verse_id"])
        if not targets:
            continue
        bounds = []
        for key in ("group_verse_key_from", "group_verse_key_to"):
            value = row.get(key)
            if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*:[1-9][0-9]*", value):
                raise TafsirSourceError("Références du passage original absentes ou invalides.")
            bounds.append(tuple(map(int, value.split(":"))))
        start, end = bounds
        if (
            start[0] != end[0] or verse_ids.get(start) != row["start_verse_id"]
            or verse_ids.get(end) != row["end_verse_id"]
            or type(row.get("group_verses_count")) is not int
            or row["group_verses_count"] != row["end_verse_id"] - row["start_verse_id"] + 1
        ):
            raise TafsirSourceError("La plage du passage source est incohérente ou traverse deux sourates.")
        if covered.intersection(targets):
            raise TafsirSourceError("Plusieurs textes originaux couvrent le même verset ; import interrompu.")
        plain = _plain_text(text)
        if not plain.strip():
            continue
        covered.update(targets)
        records.append(row)  # Preserve the original HTML and all row metadata privately.
        passages.append({
            "source_surah_id": start[0], "source_start_ayah": start[1], "source_end_ayah": end[1],
            "source_reference": f"{snapshot_url}#record-{row['id']}", "source_text": plain,
            "ayahs": [ayah for _, ayah in targets],
        })
    if not passages:
        raise TafsirSourceError("Aucun passage original disponible pour le pilote ; aucun texte n'est inventé.")
    records.sort(key=lambda row: row["id"])
    passages.sort(key=lambda passage: (passage["source_surah_id"], passage["source_start_ayah"]))
    version = "sha256:" + _digest({"resource": resource, "records": records})
    batch = validate_tafsir_generation_batch({
        "schema_version": 1, "source": source, "source_language": "ar",
        "source_edition": f"Quran Foundation — {resource['name']} (resource {definition.resource_id}, arabic)",
        "version": version, "reuse_reference": TERMS_URL, "passages": passages,
    })
    return {
        "schema_version": 1, "provider": PROVIDER, "source": source, "environment": environment,
        "synced_at": datetime.now(timezone.utc).isoformat(), "sync": sync_state,
        "catalog_resource": resource,
        "snapshot": {**{key: snapshot[key] for key in (
            "resource_group", "resource_id", "resource_content_id", "schema_version", "sync_sequence",
        )}, "records": records},
        "missing_references": [f"{surah}:{ayah}" for surah, ayah in sorted(PILOT_REFERENCES - covered)],
        "generation_batch": batch.model_dump(mode="json"),
    }


def read_archive_batch(archive):
    """Recheck source identity and raw rows before any chargeable translation."""
    try:
        if type(archive["schema_version"]) is not int or archive["schema_version"] != 1 or archive["provider"] != PROVIDER:
            raise TafsirSourceError("Archive source invalide.")
        rebuilt = build_source_archive(
            archive["source"], archive["catalog_resource"], archive["snapshot"],
            archive["environment"], archive["sync"],
        )
        if rebuilt["generation_batch"] != archive["generation_batch"]:
            raise TafsirSourceError("Le lot de génération ne correspond plus aux textes originaux archivés.")
        return rebuilt["generation_batch"]
    except (KeyError, TypeError) as exc:
        raise TafsirSourceError("Archive source incomplète ou invalide.") from exc
