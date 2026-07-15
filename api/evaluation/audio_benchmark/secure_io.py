from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import mkstemp
from typing import Iterator

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
PUBLIC_FILE_MODE = 0o644


def prepare_directory(
    path: str | Path,
    *,
    private: bool,
    tighten_existing: bool = False,
) -> Path:
    directory = Path(path)
    already_existed = directory.exists()
    directory.mkdir(
        mode=PRIVATE_DIRECTORY_MODE if private else 0o755,
        parents=True,
        exist_ok=True,
    )

    if private and (tighten_existing or not already_existed):
        directory.chmod(PRIVATE_DIRECTORY_MODE)

    return directory


@contextmanager
def atomic_output_path(
    destination: str | Path,
    *,
    private: bool,
) -> Iterator[Path]:
    """Expose un temporaire 0600, puis le remplace atomiquement une fois complet."""
    output_path = Path(destination)
    prepare_directory(output_path.parent, private=private)
    file_mode = PRIVATE_FILE_MODE if private else PUBLIC_FILE_MODE

    if private and output_path.exists() and not output_path.is_symlink():
        output_path.chmod(PRIVATE_FILE_MODE)

    file_descriptor, temp_name = mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    temp_path = Path(temp_name)

    try:
        os.fchmod(file_descriptor, file_mode)
        os.close(file_descriptor)
        file_descriptor = -1
        yield temp_path

        with temp_path.open("rb") as temp_file:
            os.fsync(temp_file.fileno())

        temp_path.chmod(file_mode)
        os.replace(temp_path, output_path)
    finally:
        if file_descriptor >= 0:
            os.close(file_descriptor)
        temp_path.unlink(missing_ok=True)


def atomic_write_text(
    destination: str | Path,
    content: str,
    *,
    private: bool,
) -> None:
    with atomic_output_path(destination, private=private) as temp_path:
        with temp_path.open("w", encoding="utf-8") as output_file:
            output_file.write(content)
            output_file.flush()
            os.fsync(output_file.fileno())
