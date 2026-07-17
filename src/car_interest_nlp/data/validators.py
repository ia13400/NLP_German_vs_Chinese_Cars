from __future__ import annotations

import hashlib
import json
from pathlib import Path


def file_sha256(path: str | Path) -> str:
    """Calculate a source-file checksum without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_local_cache(path: str | Path, metadata_path: str | Path | None = None) -> bool:
    """Return whether a cached dataset exists and seems valid."""
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        return False
    if metadata_path is None:
        return True
    metadata_file = Path(metadata_path)
    if not metadata_file.is_file():
        return False
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    required = {"download_timestamp", "source_url", "file_hash", "record_count", "schema_version"}
    return required.issubset(metadata) and metadata["file_hash"] == file_sha256(source)
