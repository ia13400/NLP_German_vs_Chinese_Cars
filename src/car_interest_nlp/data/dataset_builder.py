from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_project_config
from ..logging_utils import configure_logging
from ..paths import ARTIFACTS_DIR, ensure_directory
from ..preprocessing.brand_matching import build_brand_alias_map
from .errors import SourceUnavailableError
from .kba import (
    FZ10_DEFAULT_MONTH,
    FZ10_DEFAULT_YEARS,
    FZ10_LANDING_URL_TEMPLATE,
    FZ10_RAW_FILENAME_PATTERN,
    build_kba_monthly_series,
    discover_kba_files,
    download_kba_file,
    load_kba_xlsx,
    normalize_kba_brands,
    tidy_fz10_annual_totals,
)

logger = configure_logging()

RAW_FILE_SUFFIXES = (".xlsx", ".xls", ".csv")

# Standard output location for scripts/build_kba_annual_totals.py. Real KBA downloads
# under data/raw/registrations/kba/ are never already in the tidy shape this pipeline
# needs (see kba.py::load_kba_xlsx), so once this tidied file exists it is always
# preferred over scanning the raw directory directly.
DEFAULT_KBA_INTERIM_FILE = PROJECT_ROOT / "data" / "interim" / "kba" / "kba_annual_brand_totals.csv"


def _find_local_kba_files(raw_directory: Path) -> list[Path]:
    if not raw_directory.exists():
        return []
    return sorted(p for p in raw_directory.iterdir() if p.suffix.lower() in RAW_FILE_SUFFIXES)


def ensure_kba_dataset(
    *,
    years: Sequence[int] = FZ10_DEFAULT_YEARS,
    month: int = FZ10_DEFAULT_MONTH,
    raw_directory: str | Path | None = None,
    interim_file: str | Path | None = None,
) -> Path:
    """Ensure the tidied KBA interim CSV exists, downloading/tidying only what's missing.

    If `interim_file` (default `DEFAULT_KBA_INTERIM_FILE`) already exists, it is reused
    as-is -- nothing is downloaded or re-tidied. Otherwise, only the raw FZ10 files not
    already present under `raw_directory` are downloaded from kba.de (already-downloaded
    files are reused, never re-fetched), and the interim CSV is built from whatever raw
    files end up present.
    """
    interim_path = Path(interim_file) if interim_file is not None else DEFAULT_KBA_INTERIM_FILE
    if interim_path.is_file():
        logger.info("KBA interim file already present at %s; nothing to download", interim_path)
        return interim_path

    raw_dir = (
        Path(raw_directory)
        if raw_directory is not None
        else PROJECT_ROOT / "data" / "raw" / "registrations" / "kba"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    for year in years:
        target = raw_dir / f"fz10_{year}_{month:02d}.xlsx"
        if target.exists():
            logger.info("Raw KBA file already present, skipping download: %s", target)
            continue
        landing_url = FZ10_LANDING_URL_TEMPLATE.format(year=year, month=month)
        links = discover_kba_files(landing_url)
        if not links:
            logger.warning("No FZ10 download link found at %s", landing_url)
            continue
        download_kba_file(links[0]["url"], raw_dir)

    frames: list[pd.DataFrame] = []
    for file_path in sorted(raw_dir.glob("fz10_*.xlsx")):
        match = FZ10_RAW_FILENAME_PATTERN.match(file_path.name)
        if not match:
            continue
        try:
            frames.append(tidy_fz10_annual_totals(file_path, int(match.group(1))))
        except Exception:
            logger.exception("Failed to tidy %s", file_path)

    if not frames:
        raise SourceUnavailableError(
            source="kba",
            reason=f"No FZ10 files could be downloaded or found under {raw_dir}.",
            required_action="Check kba.de availability, or place real FZ10 files manually "
            f"under {raw_dir} and re-run.",
            accepted_fallback="There is no synthetic fallback; real KBA data is required.",
        )

    combined = pd.concat(frames, ignore_index=True)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(interim_path, index=False, encoding="utf-8")
    logger.info("Wrote tidied KBA interim file with %s rows to %s", len(combined), interim_path)
    return interim_path


def build_analysis_dataset(
    *,
    mode: str = "cached",
    kba_raw_directory: str | Path | None = None,
    listing_url: str | None = None,
    kba_file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the KBA registration-share dataset. There is no synthetic fallback of any kind.

    - mode="cached" (default): use the tidied file at `DEFAULT_KBA_INTERIM_FILE` if it
      exists (produced by `scripts/build_kba_annual_totals.py`); otherwise fall back to
      previously downloaded real KBA files under `kba_raw_directory` (default
      `data/raw/registrations/kba/`). Raises `SourceUnavailableError` if neither exists.
    - mode="live": discover and download files from `listing_url` -- a specific kba.de
      subpage, since KBA's overview pages link to further subpages rather than files
      directly (see `data/kba.py`) -- into `kba_raw_directory`, then load them. Always
      does a fresh discovery/download regardless of any existing tidied file.
    - mode="manual_import": use `kba_file_path` if given, else the tidied
      `DEFAULT_KBA_INTERIM_FILE` if it exists, else the first file already present
      under `kba_raw_directory`.

    Files loaded directly from `kba_raw_directory` (i.e. when no tidied file exists yet)
    are expected to already be tidied into the `reporting_period`/`brand`/`value_type`/
    `registrations` shape `kba.validate_kba_data` requires -- real KBA downloads are not
    already in that shape (see `data/kba.py::load_kba_xlsx` /
    `data/kba.py::tidy_fz10_annual_totals`); an untidy file raises a clear `KbaDataError`
    listing the missing columns rather than being silently reinterpreted.

    Rows whose raw KBA brand string cannot be confidently mapped to a canonical brand are
    written to `artifacts/tables/unresolved_kba_brands.csv` for manual review rather than
    being guessed or silently dropped.
    """
    config = load_project_config()
    kba_config = config["sources"].get("kba", {})
    raw_directory = (
        Path(kba_raw_directory)
        if kba_raw_directory is not None
        else PROJECT_ROOT / kba_config.get("raw_directory", "data/raw/registrations/kba")
    )
    listing_url = listing_url if listing_url is not None else kba_config.get("listing_url")

    if kba_file_path is not None:
        files = [Path(kba_file_path)]
    elif mode == "live":
        if not listing_url:
            raise SourceUnavailableError(
                source="kba",
                reason="mode='live' requires an explicit listing_url (a specific kba.de subpage).",
                required_action=(
                    "Locate the relevant KBA subpage manually (see README) and pass "
                    "listing_url=... ."
                ),
                accepted_fallback="mode='cached' if files were already downloaded, or mode='manual_import'.",
            )
        links = discover_kba_files(listing_url)
        if not links:
            raise SourceUnavailableError(
                source="kba",
                reason=f"No downloadable XLSX/CSV files found at {listing_url}.",
                required_action=(
                    "Verify listing_url still contains direct file links; KBA page "
                    "structure changes over time."
                ),
                accepted_fallback="mode='cached', if files were already downloaded.",
            )
        for link in links:
            download_kba_file(link["url"], raw_directory)
        files = _find_local_kba_files(raw_directory)
    elif DEFAULT_KBA_INTERIM_FILE.is_file():
        logger.info("Using tidied KBA file at %s", DEFAULT_KBA_INTERIM_FILE)
        files = [DEFAULT_KBA_INTERIM_FILE]
    elif mode == "manual_import":
        files = _find_local_kba_files(raw_directory)
        if not files or not all(file_path.is_file() for file_path in files):
            raise SourceUnavailableError(
                source="kba",
                reason=f"No manually supplied KBA file found under {raw_directory} and no "
                f"tidied file at {DEFAULT_KBA_INTERIM_FILE}.",
                required_action=(
                    "Run scripts/build_kba_annual_totals.py after downloading raw KBA "
                    "files, or pass kba_file_path=... directly."
                ),
                accepted_fallback="There is no synthetic fallback; a real KBA file is required.",
            )
    elif mode == "cached":
        files = _find_local_kba_files(raw_directory)
        if not files:
            raise SourceUnavailableError(
                source="kba",
                reason=f"No cached KBA data found under {raw_directory} and no tidied file "
                f"at {DEFAULT_KBA_INTERIM_FILE}.",
                required_action=(
                    "Run build_analysis_dataset(mode='live', listing_url=...) once, then "
                    "scripts/build_kba_annual_totals.py, or use mode='manual_import'."
                ),
                accepted_fallback="There is no synthetic fallback; real KBA data is required.",
            )
    else:
        raise ValueError(f"Unknown execution mode: {mode!r}")

    raw_frames = [load_kba_xlsx(file_path) for file_path in files]
    combined = pd.concat(raw_frames, ignore_index=True, sort=False)

    alias_map = build_brand_alias_map(config["brands"])
    resolved, unresolved = normalize_kba_brands(combined, alias_map)

    if not unresolved.empty:
        unresolved_path = ARTIFACTS_DIR / "tables" / "unresolved_kba_brands.csv"
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, index=False)
        logger.warning(
            "%s KBA rows had unresolved brand names; written to %s for manual review",
            len(unresolved),
            unresolved_path,
        )

    if resolved.empty:
        raise SourceUnavailableError(
            source="kba",
            reason="No KBA rows could be mapped to a canonical brand.",
            required_action=(
                "Review artifacts/tables/unresolved_kba_brands.csv and extend "
                "configs/brands.yaml or kba.KBA_BRAND_SPELLING_OVERRIDES."
            ),
            accepted_fallback="There is no synthetic fallback; resolvable KBA data is required.",
        )

    series = build_kba_monthly_series(resolved)
    logger.info("Built KBA registration series with %s rows (mode=%s)", len(series), mode)

    ensure_directory(Path(config["project"]["output_paths"]["processed_dir"]))
    return series
