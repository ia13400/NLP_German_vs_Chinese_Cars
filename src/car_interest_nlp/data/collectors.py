from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..logging_utils import configure_logging
from .validators import file_sha256, validate_local_cache

logger = configure_logging()


def _record_count(path: Path) -> int:
    """Best-effort row count for the cache metadata sidecar (not part of the analysis pipeline).

    Excel files are binary and must not be read as text. For any other format that
    fails UTF-8 decoding (e.g. an unexpected binary download), this reports 0 rather
    than crashing the caller -- it only affects the informational metadata record.
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return len(pd.read_csv(path, encoding="utf-8"))
    if suffix in (".xlsx", ".xls"):
        return len(pd.read_excel(path))
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except UnicodeDecodeError:
        logger.warning("Could not count records in %s as text; recording 0.", path)
        return 0


def write_cache_metadata(
    source_path: str | Path,
    source_url: str = "",
    schema_version: str = "1.0",
) -> Path:
    """Write reproducibility metadata next to a valid local source."""
    source = Path(source_path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"Cannot describe missing or empty source: {source}")
    metadata_path = source.with_suffix(source.suffix + ".metadata.json")
    payload = {
        "download_timestamp": datetime.now(UTC).isoformat(),
        "source_url": source_url,
        "file_hash": file_sha256(source),
        "record_count": _record_count(source),
        "schema_version": schema_version,
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def collect_source(
    source_name: str,
    source_path: str | Path,
    force_refresh: bool = False,
    source_url: str = "",
) -> Path:
    """Return a local source path and log collection reuse decisions."""
    source_path = Path(source_path)
    metadata_path = source_path.with_suffix(source_path.suffix + ".metadata.json")
    if source_path.exists() and not metadata_path.exists():
        write_cache_metadata(source_path, source_url)
    if validate_local_cache(source_path, metadata_path) and not force_refresh:
        logger.info("Reusing cached source for %s at %s", source_name, source_path)
        return source_path
    if force_refresh and source_path.exists():
        logger.warning(
            "Refresh requested for %s, but no remote adapter is configured; preserving raw data",
            source_name,
        )
        write_cache_metadata(source_path, source_url)
        return source_path
    raise FileNotFoundError(
        f"Source {source_name!r} is unavailable at {source_path}; configure an adapter or supply it locally"
    )
