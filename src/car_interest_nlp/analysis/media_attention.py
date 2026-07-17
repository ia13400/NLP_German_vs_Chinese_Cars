from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..data.gdelt import (
    GDELT_DEFAULT_TOP_N_BRANDS,
    GDELT_MODE_TIMELINE_VOLUME,
    GdeltDataError,
    tidy_timeline_chunk,
)
from ..data.gdelt_dataset_builder import (
    brand_queries,
    gdelt_config_defaults,
    resolve_scope,
    timelinevol_paths_for_brand,
)
from ..logging_utils import configure_logging
from .top_brands import get_top_brands

logger = configure_logging()


def _load_timeline_frame(
    raw_dir: Path, start: date, end: date, top_n_brands: int
) -> tuple[pd.DataFrame, int, int]:
    """Load every real cached `timelinevol` chunk for this scope, per brand.

    Uses `timelinevol_paths_for_brand()` rather than a fixed windowing scheme, since a live
    fetch may have used the primary full-range-per-brand strategy for some brands and the
    yearly-chunk fallback for others (see `gdelt_dataset_builder.ensure_gdelt_timelinevol_dataset`'s
    docstring) -- reading has to check for whichever one actually succeeded, per brand.
    `found`/`total` count *brands* covered, not raw chunk files, since the two strategies use
    a different number of files per brand (1 vs. 5) for the same real coverage.
    """
    queries = brand_queries(top_n_brands)
    frames: list[pd.DataFrame] = []
    found = 0
    for brand, query in queries.items():
        paths = timelinevol_paths_for_brand(raw_dir, query, start, end)
        brand_frames = []
        for path in paths:
            if not path.is_file():
                continue
            try:
                brand_frames.append(
                    tidy_timeline_chunk(path, brand=brand, mode=GDELT_MODE_TIMELINE_VOLUME)
                )
            except GdeltDataError as exc:
                logger.warning("Skipping unparsable GDELT timelinevol chunk %s: %s", path, exc)
        if brand_frames:
            found += 1
            frames.extend(brand_frames)
    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    return combined, found, len(queries)


def build_attention_over_time(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
) -> pd.DataFrame:
    """Real media-attention-over-time series per brand, from GDELT's own `timelinevol` mode.

    Built entirely from GDELT's own aggregate endpoint -- never requires fetching or
    scraping individual articles, so this half of the media-attention analysis stays cheap
    (a handful of cached chunks) regardless of how article/scraping coverage is going.
    Works over whatever `timelinevol` chunks are cached (full-range or yearly-fallback, see
    `_load_timeline_frame`) -- an empty frame simply means no `timelinevol` chunks are cached
    yet for any brand.
    """
    raw_dir, start, end, top_n = resolve_scope(raw_directory, start_date, end_date, top_n_brands)
    combined, found, total = _load_timeline_frame(raw_dir, start, end, top_n)
    logger.info(
        "Built media-attention-over-time from %s/%s brands with cached timelinevol data",
        found,
        total,
    )
    if combined.empty:
        return combined
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    unparsed = combined["date"].isna().sum()
    if unparsed:
        logger.warning(
            "%s/%s timelinevol rows had an unparsable date and were dropped",
            unparsed,
            len(combined),
        )
        combined = combined.dropna(subset=["date"])
    return combined.sort_values(["brand", "date"]).reset_index(drop=True)


def summarize_attention_by_year(
    attention_frame: pd.DataFrame,
    *,
    value_column: str = "value",
    top_n_brands: int | None = None,
    start_year: int = 2021,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Aggregate `build_attention_over_time()`'s daily `timelinevol` rows to one row per
    (brand, year): the *mean* daily coverage-intensity share across that year.

    The mean, not the sum: `timelinevol`'s `value` is already a percentage ("share of all
    GDELT-monitored global news that day mentioning this brand+context") -- summing 365 of
    those has no clean interpretation (it doesn't correspond to a real "yearly share" and can
    trivially exceed 100%), whereas the mean answers a real question: "on a typical day that
    year, what share of monitored global coverage mentioned this brand?" GDELT's DOC 2.0 API
    has no documented parameter to request pre-aggregated yearly buckets directly from
    `timelinevol` itself (unlike, say, a `timelinesmooth` moving-average window) -- this
    aggregation is done client-side, offline, over data already fetched, needing no
    additional GDELT requests.

    Rows outside `[start_year, end_year]` are dropped before aggregating: a yearly window's
    end date is inclusive (e.g. the 2025 window runs 2025-01-01 to 2026-01-01), so its final
    real daily datapoint is dated exactly 2026-01-01 -- confirmed directly in real cached
    data -- which would otherwise show up as a misleading stray one-day "2026" row (the same
    reasoning `plot_gdelt_article_count_trend`'s `start_year`/`end_year` filtering uses).
    """
    if attention_frame.empty:
        return attention_frame
    resolved_top_n = (
        top_n_brands
        if top_n_brands is not None
        else gdelt_config_defaults().get("top_n_brands", GDELT_DEFAULT_TOP_N_BRANDS)
    )
    top_brands = get_top_brands(resolved_top_n)
    brand_group_map = {brand: group for group, brands in top_brands.items() for brand in brands}

    yearly = attention_frame.copy()
    yearly["reporting_period"] = yearly["date"].dt.year.astype(str)
    yearly = yearly[(yearly["date"].dt.year >= start_year) & (yearly["date"].dt.year <= end_year)]
    grouped = (
        yearly.groupby(["reporting_period", "brand"])[value_column]
        .mean()
        .reset_index()
        .rename(columns={"brand": "canonical_brand", value_column: "mean_daily_share"})
    )
    grouped["brand_group"] = grouped["canonical_brand"].map(brand_group_map)
    return grouped.sort_values(["canonical_brand", "reporting_period"]).reset_index(drop=True)
