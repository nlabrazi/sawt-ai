"""Private, atomic progress files for the manual tafsir translation command."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from pydantic import ValidationError

from app.schemas.tafsir import TafsirGenerationBatch, TafsirGenerationProgress

REQUEST_VERSION = "deepl-ar-fr-v1"


class TafsirProgressError(Exception):
    pass


def generation_fingerprint(batch: TafsirGenerationBatch) -> str:
    original = json.dumps(batch.model_dump(mode="json"), ensure_ascii=False,
                          sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(original).hexdigest()


def load_generation_progress(path: Path, batch: TafsirGenerationBatch) -> TafsirGenerationProgress:
    fingerprint = generation_fingerprint(batch)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        progress = TafsirGenerationProgress.model_validate(payload)
    except FileNotFoundError:
        return TafsirGenerationProgress(batch_sha256=fingerprint, request_version=REQUEST_VERSION)
    except (OSError, ValueError, ValidationError) as exc:
        raise TafsirProgressError("Fichier de progression illisible ou invalide ; il est conservé.") from exc
    if progress.batch_sha256 != fingerprint:
        raise TafsirProgressError("Ce fichier de progression appartient à un autre lot source ; aucun appel DeepL.")

    expected = {}
    for passage in batch.passages:
        original = passage.model_dump(exclude={"ayahs"})
        for ayah in passage.ayahs:
            expected[(passage.source_surah_id, ayah)] = {
                **original, "surah_id": passage.source_surah_id, "ayah": ayah,
                "source": batch.source, "version": batch.version,
            }
    completed = {}
    for entry in progress.entries:
        reference = (entry.surah_id, entry.ayah)
        original = entry.model_dump(exclude={"text_fr", "status", "reviewed_at", "generation"})
        if reference in completed or expected.get(reference) != original or entry.generation is None:
            raise TafsirProgressError("La progression ne correspond pas aux passages du lot ; aucun appel DeepL.")
        completed[reference] = entry
    for passage in batch.passages:
        group = [completed.get((passage.source_surah_id, ayah)) for ayah in passage.ayahs]
        present = [entry for entry in group if entry is not None]
        if present and (len(present) != len(group) or any(
            (entry.text_fr, entry.generation) != (present[0].text_fr, present[0].generation)
            for entry in present
        )):
            raise TafsirProgressError("Un passage sauvegardé est incomplet ou incohérent ; aucun appel DeepL.")
    return progress


def save_generation_progress(path: Path, progress: TafsirGenerationProgress) -> None:
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(progress.model_dump(mode="json"), file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise TafsirProgressError("Impossible de sauvegarder la progression ; génération arrêtée.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@contextmanager
def locked_generation_progress(path: Path, batch: TafsirGenerationBatch):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the lock file in place: unlinking it would allow concurrent lock inodes.
    descriptor = os.open(str(path) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    with os.fdopen(descriptor, "a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise TafsirProgressError("Une génération utilise déjà ce fichier de progression.") from exc
        yield load_generation_progress(path, batch)
