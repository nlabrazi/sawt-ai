from __future__ import annotations

MAX_FILE_SIZE_BYTES = 12 * 1024 * 1024
MAX_AUDIO_DURATION_SECONDS = 90

CONTENT_TYPE_ALIASES = {
    "audio/wav": "audio/wav",
    "audio/x-wav": "audio/wav",
    "audio/mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/mp4": "audio/mp4",
    "audio/x-m4a": "audio/mp4",
    "audio/ogg": "audio/ogg",
    "audio/webm": "audio/webm",
}
CONTENT_TYPE_TO_EXTENSION = {
    "audio/wav": ".wav",
    "audio/mpeg": ".mp3",
    "audio/mp4": ".m4a",
    "audio/ogg": ".ogg",
    "audio/webm": ".webm",
}
ACCEPTED_MIME_TYPES = tuple(CONTENT_TYPE_ALIASES.keys())
ACCEPTED_FILE_EXTENSIONS = tuple(dict.fromkeys(CONTENT_TYPE_TO_EXTENSION.values()))


def canonicalize_content_type(content_type: str | None) -> str:
    if not content_type:
        return ""

    normalized_content_type = content_type.strip().lower()
    return CONTENT_TYPE_ALIASES.get(normalized_content_type, normalized_content_type)


def resolve_temp_extension(content_type: str) -> str:
    return CONTENT_TYPE_TO_EXTENSION.get(content_type, ".bin")


def build_upload_policy() -> dict[str, int | list[str]]:
    return {
        "max_file_size_bytes": MAX_FILE_SIZE_BYTES,
        "max_audio_duration_seconds": MAX_AUDIO_DURATION_SECONDS,
        "accepted_mime_types": list(ACCEPTED_MIME_TYPES),
        "accepted_file_extensions": list(ACCEPTED_FILE_EXTENSIONS),
    }
