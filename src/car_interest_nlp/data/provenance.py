from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils import stable_hash
from .validators import file_sha256

PROVENANCE_COLUMNS: tuple[str, ...] = (
    "source_type",
    "source_name",
    "source_url",
    "source_record_id",
    "retrieved_at",
    "reporting_period",
    "parser_version",
    "raw_file_path",
    "raw_file_hash",
    "license_or_usage_note",
    "collection_method",
)


def attach_provenance(
    frame: pd.DataFrame,
    *,
    source_type: str,
    source_name: str,
    source_url: str,
    parser_version: str,
    collection_method: str,
    license_note: str,
    record_id_column: str | None = None,
    reporting_period_column: str | None = None,
    raw_file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Stamp every row of `frame` with the ten required provenance fields.

    Called once, right after normalizing raw KBA records, so the resulting registration-
    share table always carries the same provenance shape.
    """
    result = frame.copy()
    retrieved_at = datetime.now(UTC).isoformat()
    result["source_type"] = source_type
    result["source_name"] = source_name
    result["source_url"] = source_url
    if record_id_column and record_id_column in result:
        result["source_record_id"] = result[record_id_column].astype(str)
    else:
        result["source_record_id"] = [
            stable_hash(f"{source_name}:{index}") for index in range(len(result))
        ]
    result["retrieved_at"] = retrieved_at
    result["reporting_period"] = (
        result[reporting_period_column]
        if reporting_period_column and reporting_period_column in result
        else pd.NA
    )
    result["parser_version"] = parser_version
    raw_path = Path(raw_file_path) if raw_file_path is not None else None
    result["raw_file_path"] = str(raw_path) if raw_path is not None else pd.NA
    result["raw_file_hash"] = (
        file_sha256(raw_path) if raw_path is not None and raw_path.is_file() else pd.NA
    )
    result["license_or_usage_note"] = license_note
    result["collection_method"] = collection_method
    return result


def validate_provenance(frame: pd.DataFrame) -> None:
    """Raise if a processed frame is missing any required provenance column."""
    missing = [column for column in PROVENANCE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required provenance columns: {missing}")


@dataclass
class SourceRegistryRow:
    """One row of artifacts/tables/data_source_registry.csv."""

    source_name: str
    official_url: str
    data_category: str
    access_method: str
    enabled: bool
    latest_collection: str | None
    record_count: int | None
    reporting_start: str | None
    reporting_end: str | None
    license_note: str
    limitations: str


# Static, human-authored description for the registry. Counts/dates are filled in
# dynamically from whatever has actually been collected on disk (see build_data_source_registry).
_SOURCE_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "kba": {
        "official_url": "https://www.kba.de/",
        "data_category": "vehicle_registrations_germany",
        "license_note": "Kraftfahrt-Bundesamt open data, official German government statistics.",
        "limitations": (
            "Distinguishes new registrations from vehicle stock; monthly new-registration files "
            "must be located manually on kba.de subpages and supplied via a listing URL."
        ),
    },
    "switzerland": {
        "official_url": "https://www.stats.swiss/",
        "data_category": "vehicle_registrations_switzerland",
        "license_note": "Swiss Federal Statistical Office open data (stats.swiss); underlying "
        "source is the Federal Roads Office (FEDRO/ASTRA).",
        "limitations": (
            "Same measurement type as KBA (national new-registration flow), so the two are "
            "meaningfully comparable in kind -- but Switzerland is a much smaller market than "
            "Germany, so absolute registration counts should not be compared directly, only "
            "market-share percentages."
        ),
    },
    "google_trends": {
        "official_url": "https://trends.google.com/",
        "data_category": "search_interest_germany",
        "license_note": "Google Trends relative search-interest index; fetched via the "
        "unofficial pytrends client (Google publishes no official Trends API).",
        "limitations": (
            "Measures search interest, not a market outcome -- not comparable in kind to "
            "KBA/Switzerland registrations. Google Trends returns each request's values "
            "normalized 0-100 independently, and allows at most 5 keywords per request, so "
            "values from different batched requests are only comparable after rescaling "
            "them onto one shared anchor brand (see chain_trends_batches()). The unofficial "
            "endpoint is also frequently rate-limited/blocked by Google, independent of "
            "this project's own request pacing."
        ),
    },
    "gdelt": {
        "official_url": "https://www.gdeltproject.org/",
        "data_category": "news_media_coverage",
        "license_note": "GDELT Project DOC 2.0 API; article metadata (title/url/date/source) "
        "is publicly published under GDELT's open access terms.",
        "limitations": (
            "Measures English-language news media coverage, not a market outcome or consumer "
            "intent -- not comparable in kind to KBA/Switzerland/Google Trends. mode=artlist "
            "returns article metadata and titles only, never full article body text (see "
            "data/article_text.py for the separate, opt-in full-text scraping module). "
            "GDELT's real rate-limit tolerance is stricter in practice than its documented "
            "'one request per 5 seconds,' confirmed directly while building this adapter; "
            "full 10-brand x 5-year x 3-mode coverage is a multi-hour, multi-run operation."
        ),
    },
}


def _metadata_files(raw_directory: Path) -> list[Path]:
    if not raw_directory.exists():
        return []
    return sorted(raw_directory.rglob("*.metadata.json"))


def build_data_source_registry(
    sources_config: dict[str, dict[str, Any]], project_root: Path
) -> list[SourceRegistryRow]:
    """Build registry rows from configs/sources.yaml plus whatever is actually on disk.

    Never invents record counts or dates: sources with nothing collected yet report
    `None`/empty rather than a guessed value.
    """
    import json

    rows: list[SourceRegistryRow] = []
    for source_name, source_config in sources_config.items():
        description = _SOURCE_DESCRIPTIONS.get(
            source_name,
            {"official_url": "", "data_category": "unknown", "license_note": "", "limitations": ""},
        )
        raw_directory_value = source_config.get("raw_directory") or source_config.get("path", "")
        raw_directory = project_root / raw_directory_value if raw_directory_value else project_root
        if raw_directory.is_file():
            raw_directory = raw_directory.parent
        metadata_files = _metadata_files(raw_directory)
        record_count: int | None = None
        latest_collection: str | None = None
        if metadata_files:
            record_count = 0
            timestamps: list[str] = []
            for metadata_file in metadata_files:
                try:
                    payload = json.loads(metadata_file.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                record_count += int(payload.get("record_count", 0))
                if payload.get("download_timestamp"):
                    timestamps.append(payload["download_timestamp"])
            latest_collection = max(timestamps) if timestamps else None
        rows.append(
            SourceRegistryRow(
                source_name=source_name,
                official_url=source_config.get("base_url") or description["official_url"],
                data_category=description["data_category"],
                access_method=source_config.get(
                    "access_method", source_config.get("source_type", "")
                ),
                enabled=bool(source_config.get("enabled", False)),
                latest_collection=latest_collection,
                record_count=record_count,
                reporting_start=source_config.get("start_date"),
                reporting_end=source_config.get("end_date"),
                license_note=description["license_note"],
                limitations=description["limitations"],
            )
        )
    return rows


def write_data_source_registry(rows: list[SourceRegistryRow], path: str | Path) -> Path:
    """Write the data source registry to artifacts/tables/data_source_registry.csv."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([asdict(row) for row in rows])
    frame.to_csv(destination, index=False)
    return destination
