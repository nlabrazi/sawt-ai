"""Validate audio signatures, size and duration before inference."""

from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException, UploadFile

from app.core.upload_policy import (
    MAX_AUDIO_DURATION_SECONDS, MAX_FILE_SIZE_BYTES, resolve_temp_extension,
)
from app.services.audio_metadata_service import AudioMetadataError, get_audio_duration_seconds

HEADER_SNIFF_BYTES = 4096
READ_CHUNK_SIZE_BYTES = 1024 * 1024


def sniff_audio_content_type(header_bytes: bytes) -> str | None:
    if (
        len(header_bytes) >= 12
        and header_bytes.startswith(b"RIFF")
        and header_bytes[8:12] == b"WAVE"
    ):
        return "audio/wav"

    if header_bytes.startswith(b"OggS"):
        return "audio/ogg"

    if len(header_bytes) >= 12 and header_bytes[4:8] == b"ftyp":
        return "audio/mp4"

    if (
        len(header_bytes) >= 4
        and header_bytes[:4] == b"\x1a\x45\xdf\xa3"
        and b"webm" in header_bytes.lower()
    ):
        return "audio/webm"

    if header_bytes.startswith(b"ID3"):
        return "audio/mpeg"

    if (
        len(header_bytes) >= 2
        and header_bytes[0] == 0xFF
        and (header_bytes[1] & 0xE0) == 0xE0
    ):
        return "audio/mpeg"

    return None


async def persist_upload_to_temp_file(file: UploadFile) -> tuple[Path, int, str]:
    header_bytes = await file.read(HEADER_SNIFF_BYTES)
    total_bytes = len(header_bytes)

    if total_bytes == 0:
        raise HTTPException(status_code=400, detail="Fichier audio vide.")

    if total_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

    detected_content_type = sniff_audio_content_type(header_bytes)

    if detected_content_type is None:
        raise HTTPException(
            status_code=415,
            detail="Format audio invalide ou non pris en charge.",
        )

    with NamedTemporaryFile(
        mode="wb",
        suffix=resolve_temp_extension(detected_content_type),
        delete=False,
        dir="/tmp",
    ) as temp_buffer:
        temp_path = Path(temp_buffer.name)

        try:
            temp_buffer.write(header_bytes)

            while True:
                chunk = await file.read(READ_CHUNK_SIZE_BYTES)
                if not chunk:
                    break

                total_bytes += len(chunk)

                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(status_code=413, detail="Fichier trop volumineux.")

                temp_buffer.write(chunk)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return temp_path, total_bytes, detected_content_type


def enforce_audio_duration_limit(
    audio_path: Path, max_duration_seconds: int = MAX_AUDIO_DURATION_SECONDS,
) -> float:
    try:
        duration_seconds = get_audio_duration_seconds(audio_path)
    except AudioMetadataError as exc:
        raise HTTPException(
            status_code=415,
            detail="Impossible de lire la durée du fichier audio.",
        ) from exc

    if duration_seconds > max_duration_seconds:
        raise HTTPException(
            status_code=413,
            detail=f"Audio trop long. Maximum {max_duration_seconds} secondes.",
        )

    return duration_seconds


