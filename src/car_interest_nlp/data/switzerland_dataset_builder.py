from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_project_config
from ..logging_utils import configure_logging
from ..paths import ARTIFACTS_DIR, ensure_directory
from ..preprocessing.brand_matching import build_brand_alias_map
from .errors import SourceUnavailableError
from .switzerland import (
    CH_CODELIST_RAW_FILENAME,
    CH_DATA_RAW_FILENAME,
    build_ch_annual_series,
    download_ch_codelist_file,
    download_ch_data_file,
    normalize_ch_brands,
    tidy_ch_new_registrations,
)

logger = configure_logging()

DEFAULT_CH_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "registrations" / "switzerland"

# Standard output location for ensure_ch_dataset(). Once this tidied file exists it is
# always preferred over re-scanning/re-tidying the raw data+codelist file pair (mirrors
# dataset_builder.DEFAULT_KBA_INTERIM_FILE).
DEFAULT_CH_INTERIM_FILE = (
    PROJECT_ROOT / "data" / "interim" / "switzerland" / "ch_annual_brand_totals.csv"
)


def ensure_ch_dataset(
    *,
    raw_directory: str | Path | None = None,
    interim_file: str | Path | None = None,
) -> Path:
    """Ensure the tidied Swiss interim CSV exists, downloading/tidying only what's missing.

    If `interim_file` (default `DEFAULT_CH_INTERIM_FILE`) already exists, it is reused
    as-is -- nothing is downloaded or re-tidied. Otherwise the data CSV and MAKE codelist
    are each downloaded from the SDMX REST API only if not already present under
    `raw_directory` (already-downloaded files are reused, never re-fetched), then tidied
    into the interim file.
    """
    interim_path = Path(interim_file) if interim_file is not None else DEFAULT_CH_INTERIM_FILE
    if interim_path.is_file():
        logger.info("Swiss interim file already present at %s; nothing to download", interim_path)
        return interim_path

    raw_dir = Path(raw_directory) if raw_directory is not None else DEFAULT_CH_RAW_DIRECTORY
    data_path = raw_dir / CH_DATA_RAW_FILENAME
    if data_path.exists():
        logger.info("Raw Swiss data file already present, skipping download: %s", data_path)
    else:
        data_path = download_ch_data_file(raw_dir)

    codelist_path = raw_dir / CH_CODELIST_RAW_FILENAME
    if codelist_path.exists():
        logger.info("Raw Swiss codelist file already present, skipping download: %s", codelist_path)
    else:
        codelist_path = download_ch_codelist_file(raw_dir)

    combined = tidy_ch_new_registrations(data_path, codelist_path)
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(interim_path, index=False, encoding="utf-8")
    logger.info("Wrote tidied Swiss interim file with %s rows to %s", len(combined), interim_path)
    return interim_path


def build_ch_analysis_dataset(
    *,
    mode: str = "cached",
    ch_raw_directory: str | Path | None = None,
) -> pd.DataFrame:
    """Build the Swiss new-registration-share dataset. There is no synthetic fallback of any kind.

    - mode="cached" (default): use the tidied file at `DEFAULT_CH_INTERIM_FILE` if it
      exists; otherwise fall back to a previously downloaded raw data+codelist file pair
      under `ch_raw_directory` (default `data/raw/registrations/switzerland`), tidying on
      the fly. Never performs a network call. Raises `SourceUnavailableError` if neither
      exists.
    - mode="live": always downloads both files fresh from the SDMX REST API regardless of
      any existing tidied file, then tidies them.
    - mode="manual_import": same as "cached" but intended for manually placed raw files
      under `ch_raw_directory`.

    Rows whose raw Swiss brand string cannot be confidently mapped to a canonical brand
    are written to `artifacts/tables/unresolved_switzerland_brands.csv` for manual review
    rather than being guessed or silently dropped.
    """
    config = load_project_config()
    ch_config = config["sources"].get("switzerland", {})
    raw_directory = (
        Path(ch_raw_directory)
        if ch_raw_directory is not None
        else PROJECT_ROOT / ch_config.get("raw_directory", "data/raw/registrations/switzerland")
    )

    if mode == "live":
        data_path = download_ch_data_file(raw_directory)
        codelist_path = download_ch_codelist_file(raw_directory)
        combined = tidy_ch_new_registrations(data_path, codelist_path)
    elif DEFAULT_CH_INTERIM_FILE.is_file():
        logger.info("Using tidied Swiss file at %s", DEFAULT_CH_INTERIM_FILE)
        combined = pd.read_csv(
            DEFAULT_CH_INTERIM_FILE, encoding="utf-8", dtype={"reporting_period": str}
        )
    elif mode in ("cached", "manual_import"):
        data_path = raw_directory / CH_DATA_RAW_FILENAME
        codelist_path = raw_directory / CH_CODELIST_RAW_FILENAME
        if not data_path.is_file() or not codelist_path.is_file():
            raise SourceUnavailableError(
                source="switzerland",
                reason=f"No cached Swiss data found under {raw_directory} and no tidied "
                f"file at {DEFAULT_CH_INTERIM_FILE}.",
                required_action=(
                    "Run ensure_ch_dataset() (or scripts/download_ch_registrations.py) once, "
                    "or use mode='live'."
                ),
                accepted_fallback="There is no synthetic fallback; real Swiss data is required.",
            )
        combined = tidy_ch_new_registrations(data_path, codelist_path)
    else:
        raise ValueError(f"Unknown execution mode: {mode!r}")

    alias_map = build_brand_alias_map(config["brands"])
    resolved, unresolved = normalize_ch_brands(combined, alias_map)

    if not unresolved.empty:
        unresolved_path = ARTIFACTS_DIR / "tables" / "unresolved_switzerland_brands.csv"
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, index=False)
        logger.warning(
            "%s Swiss rows had unresolved brand names; written to %s for manual review",
            len(unresolved),
            unresolved_path,
        )

    if resolved.empty:
        raise SourceUnavailableError(
            source="switzerland",
            reason="No Swiss rows could be mapped to a canonical brand.",
            required_action=(
                "Review artifacts/tables/unresolved_switzerland_brands.csv and extend "
                "configs/brands.yaml."
            ),
            accepted_fallback="There is no synthetic fallback; resolvable Swiss data is required.",
        )

    series = build_ch_annual_series(resolved)
    logger.info("Built Swiss registration series with %s rows (mode=%s)", len(series), mode)

    ensure_directory(Path(config["project"]["output_paths"]["processed_dir"]))
    return series
