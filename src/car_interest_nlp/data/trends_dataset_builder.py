from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, load_project_config
from ..logging_utils import configure_logging
from ..paths import ARTIFACTS_DIR, ensure_directory
from ..preprocessing.brand_matching import build_brand_alias_map
from .errors import SourceUnavailableError
from .google_trends import (
    TRENDS_DEFAULT_ANCHOR_BRAND,
    TRENDS_DEFAULT_END_YEAR,
    TRENDS_DEFAULT_GEO,
    TRENDS_DEFAULT_HL,
    TRENDS_DEFAULT_TIMEFRAME,
    TRENDS_DEFAULT_TRACKED_BRANDS,
    TRENDS_MANUAL_EXPORT_FILENAME,
    GoogleTrendsAccessError,
    GoogleTrendsDataError,
    annualize_trends_series,
    batch_raw_path,
    build_keyword_batches,
    build_trends_annual_series,
    chain_trends_batches,
    download_trends_batch,
    get_trend_brands,
    normalize_trends_brands,
    tidy_manual_trends_export,
    tidy_trends_batch,
)
from .validators import validate_local_cache

logger = configure_logging()

DEFAULT_TRENDS_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "trends"

# Standard output location for ensure_trends_dataset(). Once this tidied file exists it is
# always preferred over re-scanning/re-chaining the raw per-batch files (mirrors
# switzerland_dataset_builder.DEFAULT_CH_INTERIM_FILE).
DEFAULT_TRENDS_INTERIM_FILE = (
    PROJECT_ROOT / "data" / "interim" / "trends" / "google_trends_brand_interest.csv"
)


def _build_from_manual_export(
    path: Path, anchor_brand: str, *, end_year: int | None = TRENDS_DEFAULT_END_YEAR
) -> pd.DataFrame:
    """Build the annual series from a real, manually exported Trends CSV.

    Used instead of `_fetch_and_tidy_all_batches` whenever a manual export is found --
    see `TRENDS_MANUAL_EXPORT_FILENAME` and README's "Google Trends" section ("Option B")
    for why a manual export is sometimes the only practical way to get real data (the
    unofficial pytrends endpoint's anonymous quota can be exhausted for extended periods,
    confirmed directly while building this adapter). `chain_trends_batches` is still
    called (with a single "batch") purely to reuse its anchor-brand-column validation;
    with only one source frame it has no rescaling effect. `end_year` is forwarded to
    `annualize_trends_series` to drop any trailing partial year (see its docstring).
    """
    long_frame = tidy_manual_trends_export(path)
    chained = chain_trends_batches([long_frame], anchor_brand)
    return annualize_trends_series(chained, end_year=end_year)


def _fetch_and_tidy_all_batches(
    config: dict,
    raw_directory: Path,
    *,
    geo: str,
    timeframe: str,
    hl: str,
    anchor_brand: str,
    tracked_brands: Sequence[str],
    rate_limit_seconds: float,
    allow_download: bool,
    end_year: int | None = TRENDS_DEFAULT_END_YEAR,
) -> pd.DataFrame:
    """Tidy every cached Google Trends batch, downloading missing ones if `allow_download`.

    Batches already cached under `raw_directory` (a valid file + metadata sidecar) are
    always reused, never re-fetched -- important given how aggressively Google's
    unofficial Trends endpoint rate-limits/blocks automated requests: a run that only
    manages to fetch some batches before failing still leaves later re-runs able to pick
    up exactly where it left off. Raises `SourceUnavailableError` (listing exactly which
    keyword batches are still missing) if any batch cannot be resolved -- there is no
    partial/synthetic fallback.
    """
    brands = get_trend_brands(config["brands"], tracked_brands=tracked_brands)
    batches = build_keyword_batches(brands, anchor_brand)

    tidy_batches: list[pd.DataFrame] = []
    missing_batches: list[list[str]] = []
    for index, keywords in enumerate(batches):
        raw_path = batch_raw_path(raw_directory, keywords, geo, timeframe)
        metadata_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
        if validate_local_cache(raw_path, metadata_path):
            logger.info(
                "Reusing cached Google Trends batch %s/%s at %s", index + 1, len(batches), raw_path
            )
        elif allow_download:
            if index > 0:
                time.sleep(rate_limit_seconds)
            try:
                download_trends_batch(keywords, raw_directory, geo=geo, timeframe=timeframe, hl=hl)
            except (GoogleTrendsAccessError, GoogleTrendsDataError) as exc:
                logger.warning(
                    "Google Trends batch %s/%s (%s) failed: %s",
                    index + 1,
                    len(batches),
                    keywords,
                    exc,
                )
                missing_batches.append(keywords)
                continue
        else:
            missing_batches.append(keywords)
            continue
        tidy_batches.append(tidy_trends_batch(raw_path, keywords))

    if missing_batches:
        raise SourceUnavailableError(
            source="google_trends",
            reason=f"{len(missing_batches)} of {len(batches)} Google Trends keyword batches are "
            f"unavailable under {raw_directory} (missing keyword batches: {missing_batches}).",
            required_action=(
                "Run ensure_trends_dataset() (or scripts/download_google_trends.py) again -- "
                "already-downloaded batches are cached under data/raw/trends/ and are never "
                "re-fetched -- or use mode='live'. Google Trends' unofficial endpoint "
                "rate-limits/blocks automated requests independent of this project's own "
                "request pacing; if it keeps failing, wait and retry later."
            ),
            accepted_fallback="There is no synthetic fallback; real Google Trends data is required.",
        )

    chained = chain_trends_batches(tidy_batches, anchor_brand)
    return annualize_trends_series(chained, end_year=end_year)


def ensure_trends_dataset(
    *,
    raw_directory: str | Path | None = None,
    interim_file: str | Path | None = None,
    geo: str | None = None,
    timeframe: str | None = None,
    hl: str | None = None,
    anchor_brand: str | None = None,
    tracked_brands: Sequence[str] | None = None,
    rate_limit_seconds: float | None = None,
    manual_file_path: str | Path | None = None,
    end_year: int | None = None,
) -> Path:
    """Ensure the tidied Google Trends interim CSV exists, fetching only what's missing.

    If `interim_file` (default `DEFAULT_TRENDS_INTERIM_FILE`) already exists, it is reused
    as-is -- nothing is fetched. Otherwise, a manually exported Trends CSV is used if one
    is found: at `manual_file_path` if given, else at the conventional
    `raw_directory / TRENDS_MANUAL_EXPORT_FILENAME`. Only if neither is present is every
    keyword batch not already cached under `raw_directory` fetched from Google Trends
    (with its own retry/backoff on rate limiting), then chained onto one common scale and
    aggregated into an annual series.
    """
    interim_path = Path(interim_file) if interim_file is not None else DEFAULT_TRENDS_INTERIM_FILE
    if interim_path.is_file():
        logger.info(
            "Google Trends interim file already present at %s; nothing to download", interim_path
        )
        return interim_path

    config = load_project_config()
    trends_config = config["sources"].get("google_trends", {})
    raw_dir = Path(raw_directory) if raw_directory is not None else DEFAULT_TRENDS_RAW_DIRECTORY
    resolved_geo = geo if geo is not None else trends_config.get("geo", TRENDS_DEFAULT_GEO)
    resolved_timeframe = (
        timeframe
        if timeframe is not None
        else trends_config.get("timeframe", TRENDS_DEFAULT_TIMEFRAME)
    )
    resolved_hl = hl if hl is not None else trends_config.get("hl", TRENDS_DEFAULT_HL)
    resolved_anchor = (
        anchor_brand
        if anchor_brand is not None
        else trends_config.get("anchor_brand", TRENDS_DEFAULT_ANCHOR_BRAND)
    )
    resolved_tracked_brands = (
        tracked_brands
        if tracked_brands is not None
        else trends_config.get("tracked_brands", TRENDS_DEFAULT_TRACKED_BRANDS)
    )
    resolved_rate_limit = (
        rate_limit_seconds
        if rate_limit_seconds is not None
        else float(trends_config.get("rate_limit_seconds", 20.0))
    )
    resolved_end_year = (
        end_year if end_year is not None else trends_config.get("end_year", TRENDS_DEFAULT_END_YEAR)
    )

    resolved_manual_path = (
        Path(manual_file_path)
        if manual_file_path is not None
        else raw_dir / TRENDS_MANUAL_EXPORT_FILENAME
    )
    if resolved_manual_path.is_file():
        logger.info("Using manually exported Google Trends file at %s", resolved_manual_path)
        combined = _build_from_manual_export(
            resolved_manual_path, resolved_anchor, end_year=resolved_end_year
        )
    else:
        combined = _fetch_and_tidy_all_batches(
            config,
            raw_dir,
            geo=resolved_geo,
            timeframe=resolved_timeframe,
            hl=resolved_hl,
            anchor_brand=resolved_anchor,
            tracked_brands=resolved_tracked_brands,
            end_year=resolved_end_year,
            rate_limit_seconds=resolved_rate_limit,
            allow_download=True,
        )
    interim_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(interim_path, index=False, encoding="utf-8")
    logger.info(
        "Wrote tidied Google Trends interim file with %s rows to %s", len(combined), interim_path
    )
    return interim_path


def build_trends_analysis_dataset(
    *,
    mode: str = "cached",
    trends_raw_directory: str | Path | None = None,
    manual_file_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build the Google Trends search-interest dataset. There is no synthetic fallback of any kind.

    - mode="cached" (default): use the tidied file at `DEFAULT_TRENDS_INTERIM_FILE` if it
      exists; otherwise use a manually exported Trends CSV if one is found (at
      `manual_file_path`, or the conventional `trends_raw_directory /
      TRENDS_MANUAL_EXPORT_FILENAME`), else tidy whatever keyword batches are already
      cached under `trends_raw_directory` (default `data/raw/trends`), performing no
      network calls. Raises `SourceUnavailableError` if none of these are available.
    - mode="live": always fetches any keyword batch not already cached, regardless of any
      existing tidied file or manual export (already-cached batches are still reused,
      never re-fetched).
    - mode="manual_import": same as "cached" but intended for a manually placed export.

    Every Trends keyword was chosen directly from `configs/brands.yaml` (see
    `get_trend_brands`), so unlike KBA/Switzerland there should be no unresolved brand
    strings in practice; any row that fails to resolve is still written to
    `artifacts/tables/unresolved_google_trends_brands.csv` rather than silently dropped,
    since it would indicate a real mismatch between the fetched keywords and the current
    brands.yaml.
    """
    config = load_project_config()
    trends_config = config["sources"].get("google_trends", {})
    raw_directory = (
        Path(trends_raw_directory)
        if trends_raw_directory is not None
        else PROJECT_ROOT / trends_config.get("raw_directory", "data/raw/trends")
    )
    geo = trends_config.get("geo", TRENDS_DEFAULT_GEO)
    timeframe = trends_config.get("timeframe", TRENDS_DEFAULT_TIMEFRAME)
    hl = trends_config.get("hl", TRENDS_DEFAULT_HL)
    anchor_brand = trends_config.get("anchor_brand", TRENDS_DEFAULT_ANCHOR_BRAND)
    tracked_brands = trends_config.get("tracked_brands", TRENDS_DEFAULT_TRACKED_BRANDS)
    rate_limit_seconds = float(trends_config.get("rate_limit_seconds", 20.0))
    end_year = trends_config.get("end_year", TRENDS_DEFAULT_END_YEAR)
    resolved_manual_path = (
        Path(manual_file_path)
        if manual_file_path is not None
        else raw_directory / TRENDS_MANUAL_EXPORT_FILENAME
    )

    if mode == "live":
        combined = _fetch_and_tidy_all_batches(
            config,
            raw_directory,
            geo=geo,
            timeframe=timeframe,
            hl=hl,
            anchor_brand=anchor_brand,
            tracked_brands=tracked_brands,
            rate_limit_seconds=rate_limit_seconds,
            allow_download=True,
            end_year=end_year,
        )
    elif DEFAULT_TRENDS_INTERIM_FILE.is_file():
        logger.info("Using tidied Google Trends file at %s", DEFAULT_TRENDS_INTERIM_FILE)
        combined = pd.read_csv(
            DEFAULT_TRENDS_INTERIM_FILE, encoding="utf-8", dtype={"reporting_period": str}
        )
    elif mode in ("cached", "manual_import") and resolved_manual_path.is_file():
        logger.info("Using manually exported Google Trends file at %s", resolved_manual_path)
        combined = _build_from_manual_export(resolved_manual_path, anchor_brand, end_year=end_year)
    elif mode in ("cached", "manual_import"):
        combined = _fetch_and_tidy_all_batches(
            config,
            raw_directory,
            geo=geo,
            timeframe=timeframe,
            hl=hl,
            anchor_brand=anchor_brand,
            tracked_brands=tracked_brands,
            rate_limit_seconds=rate_limit_seconds,
            allow_download=False,
            end_year=end_year,
        )
    else:
        raise ValueError(f"Unknown execution mode: {mode!r}")

    alias_map = build_brand_alias_map(config["brands"])
    resolved, unresolved = normalize_trends_brands(combined, alias_map)

    if not unresolved.empty:
        unresolved_path = ARTIFACTS_DIR / "tables" / "unresolved_google_trends_brands.csv"
        unresolved_path.parent.mkdir(parents=True, exist_ok=True)
        unresolved.to_csv(unresolved_path, index=False)
        logger.warning(
            "%s Google Trends rows had unresolved brand names; written to %s for manual review",
            len(unresolved),
            unresolved_path,
        )

    if resolved.empty:
        raise SourceUnavailableError(
            source="google_trends",
            reason="No Google Trends rows could be mapped to a canonical brand.",
            required_action=(
                "Review artifacts/tables/unresolved_google_trends_brands.csv -- this "
                "indicates fetched keywords no longer match configs/brands.yaml."
            ),
            accepted_fallback="There is no synthetic fallback; resolvable Google Trends data is required.",
        )

    series = build_trends_annual_series(resolved, geo=geo)
    logger.info(
        "Built Google Trends search-interest series with %s rows (mode=%s)", len(series), mode
    )

    ensure_directory(Path(config["project"]["output_paths"]["processed_dir"]))
    return series
