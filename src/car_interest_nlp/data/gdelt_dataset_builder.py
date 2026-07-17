from __future__ import annotations

import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from ..analysis.top_brands import get_top_brands
from ..config import PROJECT_ROOT, load_project_config
from ..logging_utils import configure_logging
from ..progress import TimeBudget, iter_with_progress
from .errors import SourceUnavailableError
from .gdelt import (
    GDELT_DEFAULT_TOP_N_BRANDS,
    GDELT_MODE_ARTICLES,
    GDELT_MODE_TIMELINE_VOLUME,
    GdeltAccessError,
    GdeltDataError,
    build_gdelt_annual_series,
    build_query,
    chunk_raw_path,
    download_gdelt_chunk,
    full_range_window,
    month_windows,
    tidy_article_chunk,
    year_windows,
)
from .validators import validate_local_cache

logger = configure_logging()

DEFAULT_GDELT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "gdelt"

# `artlist` (real article metadata, the text/media corpus) is chunked monthly -- see
# `enumerate_chunks`'s docstring for why. `timelinevol` (GDELT's own aggregate
# volume-over-time mode, used by analysis/media_attention.py so that part of the analysis
# never needs to fetch or scrape individual articles) is fetched separately by
# `ensure_gdelt_timelinevol_dataset` -- one request per brand for the whole configured range,
# with a yearly-chunk fallback per brand if that single request fails (see its docstring).
# The two phases have deliberately separate fetch functions (and separate notebook sections)
# since `timelinevol` is the much smaller, much faster phase (~10 requests vs. `artlist`'s
# ~600) and this project's notebook shows `timelinevol`'s results before starting `artlist`.


def brand_queries(top_n_brands: int, *, brands: Sequence[str] | None = None) -> dict[str, str]:
    """Map brands to their GDELT query (same `build_query()` criteria either way).

    `brands`, if given, overrides the default top-N-per-group selection (`get_top_brands`)
    with an explicit brand list -- e.g. restricting a slow `artlist` fetch to just Volkswagen
    and BYD instead of the full top-5+5, without changing how each brand's query is built.
    """
    if brands is not None:
        return {brand: build_query(brand) for brand in brands}
    top_brands = get_top_brands(top_n_brands)
    return {
        brand: build_query(brand) for group_brands in top_brands.values() for brand in group_brands
    }


def enumerate_chunks(
    queries: dict[str, str], start: date, end: date, modes: Sequence[str]
) -> list[tuple[str, str, str, date, date]]:
    """List every (brand, query, mode, month_start, month_end) chunk this scope covers.

    Used for `artlist` only (see module docstring -- `timelinevol` has its own dedicated
    fetch/read logic). `artlist` is capped at `maxrecords=250` per request -- real cached
    monthly chunks for Volkswagen/Mercedes-Benz already return exactly 250/250 articles every
    month, so a wider window would silently drop real articles -- it stays chunked monthly
    via `month_windows`. Chunks are ordered window-major then brand-minor (every brand for
    the earliest window, then every brand for the next, ...) rather than brand-major --
    **confirmed directly**: brand-major ordering left 8 of 10 brands (all Chinese brands but
    one) with zero real coverage after 6 hours of real fetching, since GDELT's rate limiting
    meant the loop never got past the first two brands. Round-robin guarantees every brand
    gets touched before any single brand goes deep, which matters most for a German-vs-
    Chinese comparison specifically.
    """
    chunks: list[tuple[str, str, str, date, date]] = []
    for mode in modes:
        windows = month_windows(start, end)
        for window_start, window_end in windows:
            for brand, query in queries.items():
                chunks.append((brand, query, mode, window_start, window_end))
    return chunks


def timelinevol_paths_for_brand(raw_dir: Path, query: str, start: date, end: date) -> list[Path]:
    """The real cached `timelinevol` file path(s) to use for one brand's query.

    Mirrors whichever strategy a live fetch actually used for this brand (see
    `ensure_gdelt_timelinevol_dataset`'s docstring): the single full-range chunk if that's
    what's cached (the primary strategy -- one request per brand for the whole configured
    range), or the five yearly chunks if a full-range request previously failed and fell back
    to them for this brand specifically. If neither exists yet, returns the full-range path
    alone (the primary strategy's expected path), so coverage reporting says "0/1 found" for
    that brand rather than a misleading "0/5".
    """
    full_start, full_end = full_range_window(start, end)[0]
    full_path = chunk_raw_path(raw_dir, query, GDELT_MODE_TIMELINE_VOLUME, full_start, full_end)
    if full_path.is_file():
        return [full_path]
    yearly_paths = [
        chunk_raw_path(raw_dir, query, GDELT_MODE_TIMELINE_VOLUME, window_start, window_end)
        for window_start, window_end in year_windows(start, end)
    ]
    if any(path.is_file() for path in yearly_paths):
        return yearly_paths
    return [full_path]


def gdelt_config_defaults() -> dict:
    return load_project_config()["sources"].get("gdelt", {})


def _resolve_gdelt_scope(
    raw_directory: str | Path | None,
    start_date: date | None,
    end_date: date | None,
    top_n_brands: int | None,
) -> tuple[Path, date, date, int, float]:
    gdelt_config = gdelt_config_defaults()
    raw_dir = Path(raw_directory) if raw_directory is not None else DEFAULT_GDELT_RAW_DIRECTORY
    resolved_start = start_date or date.fromisoformat(gdelt_config.get("start_date", "2021-01-01"))
    resolved_end = end_date or date.fromisoformat(gdelt_config.get("end_date", "2025-12-31"))
    resolved_top_n = (
        top_n_brands
        if top_n_brands is not None
        else gdelt_config.get("top_n_brands", GDELT_DEFAULT_TOP_N_BRANDS)
    )
    resolved_rate_limit = float(gdelt_config.get("rate_limit_seconds", 6.0))
    return raw_dir, resolved_start, resolved_end, resolved_top_n, resolved_rate_limit


def ensure_gdelt_timelinevol_dataset(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
    brands: Sequence[str] | None = None,
    rate_limit_seconds: float | None = None,
    max_attempts: int | None = None,
    fetch_mode: str = "live",
    time_budget: TimeBudget | None = None,
) -> dict[str, int]:
    """Fetch (or, in `fetch_mode="cached"`, just report) real GDELT `timelinevol` coverage.

    One request per brand for the whole configured date range (10 brands = 10 requests
    total) -- **confirmed directly** that this real, uncapped GDELT mode returns full daily
    data across a full 5-year window, with only the same small real data gaps already present
    in yearly-chunked fetches (an 18-day genuine GDELT-side gap was found in *both* a
    full-range request and the already-cached yearly chunks for the same brand/dates -- not a
    request-width-specific truncation). If a brand's full-range request itself fails after
    all retries, falls back to five separate one-year requests for that brand only
    (`year_windows`, the already-confirmed-safe granularity) rather than giving up on the
    brand entirely.

    `fetch_mode="cached"` never sends a request -- it only checks what's already on disk and
    reports coverage, for re-running the notebook/analysis without spending any more of
    GDELT's rate-limit tolerance once real data collection is done. `fetch_mode="live"`
    (default) fetches whatever is still missing -- already-cached brands are always skipped
    first (whichever of the two window strategies is actually on disk), so a re-run resumes
    exactly where the previous one stopped rather than re-attempting brands that already
    succeeded. `max_attempts` overrides `GDELT_DEFAULT_MAX_ATTEMPTS` (default 15) for every
    request this call makes, including the yearly fallback requests -- useful to push higher
    for a brand that kept failing under persistent real rate limiting. `brands` overrides the
    default top-N-per-group brand selection (see `brand_queries`'s docstring).
    """
    raw_dir, resolved_start, resolved_end, resolved_top_n, default_rate_limit = (
        _resolve_gdelt_scope(raw_directory, start_date, end_date, top_n_brands)
    )
    resolved_rate_limit = (
        rate_limit_seconds if rate_limit_seconds is not None else default_rate_limit
    )
    chunk_kwargs = {} if max_attempts is None else {"max_attempts": max_attempts}
    raw_dir.mkdir(parents=True, exist_ok=True)

    queries = brand_queries(resolved_top_n, brands=brands)
    summary = {"fetched": 0, "cached": 0, "failed": 0, "fallback_used": 0, "total": len(queries)}

    for brand, query in iter_with_progress(
        list(queries.items()), total=len(queries), desc="GDELT timelinevol", time_budget=time_budget
    ):
        full_start, full_end = full_range_window(resolved_start, resolved_end)[0]
        full_path = chunk_raw_path(raw_dir, query, GDELT_MODE_TIMELINE_VOLUME, full_start, full_end)
        full_meta = full_path.with_suffix(full_path.suffix + ".metadata.json")
        if validate_local_cache(full_path, full_meta):
            summary["cached"] += 1
            continue

        yearly = year_windows(resolved_start, resolved_end)
        yearly_paths = [
            chunk_raw_path(raw_dir, query, GDELT_MODE_TIMELINE_VOLUME, window_start, window_end)
            for window_start, window_end in yearly
        ]
        if all(
            validate_local_cache(path, path.with_suffix(path.suffix + ".metadata.json"))
            for path in yearly_paths
        ):
            summary["cached"] += 1
            continue

        if fetch_mode == "cached":
            summary["failed"] += 1
            continue

        try:
            download_gdelt_chunk(
                query, GDELT_MODE_TIMELINE_VOLUME, full_start, full_end, raw_dir, **chunk_kwargs
            )
            summary["fetched"] += 1
        except (GdeltAccessError, GdeltDataError) as exc:
            logger.warning(
                "Full-range timelinevol request failed for %s, falling back to yearly chunks: %s",
                brand,
                exc,
            )
            summary["fallback_used"] += 1
            fallback_complete = True
            for window_start, window_end in yearly:
                path = chunk_raw_path(
                    raw_dir, query, GDELT_MODE_TIMELINE_VOLUME, window_start, window_end
                )
                if validate_local_cache(path, path.with_suffix(path.suffix + ".metadata.json")):
                    continue
                try:
                    download_gdelt_chunk(
                        query,
                        GDELT_MODE_TIMELINE_VOLUME,
                        window_start,
                        window_end,
                        raw_dir,
                        **chunk_kwargs,
                    )
                except (GdeltAccessError, GdeltDataError) as inner_exc:
                    logger.warning(
                        "Yearly fallback chunk also failed for %s (%s-%s): %s",
                        brand,
                        window_start,
                        window_end,
                        inner_exc,
                    )
                    fallback_complete = False
                time.sleep(resolved_rate_limit)
            summary["fetched" if fallback_complete else "failed"] += 1
        time.sleep(resolved_rate_limit)

    logger.info(
        "GDELT timelinevol fetch summary: %s cached, %s newly fetched, %s failed, "
        "%s used yearly fallback, %s total",
        summary["cached"],
        summary["fetched"],
        summary["failed"],
        summary["fallback_used"],
        summary["total"],
    )
    return summary


def ensure_gdelt_artlist_dataset(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
    brands: Sequence[str] | None = None,
    rate_limit_seconds: float | None = None,
    max_attempts: int | None = None,
    fetch_mode: str = "live",
    time_budget: TimeBudget | None = None,
) -> dict[str, int]:
    """Fetch (or, in `fetch_mode="cached"`, just report) real GDELT `artlist` coverage.

    Chunked monthly, round-robin across brands (10 brands x 60 months = 600 requests by
    default) -- see `enumerate_chunks`'s docstring for why. `fetch_mode="cached"` never sends
    a request, and `max_attempts` overrides `GDELT_DEFAULT_MAX_ATTEMPTS` -- see
    `ensure_gdelt_timelinevol_dataset`'s docstring for the same reasoning on both. `brands`
    overrides the default top-N-per-group brand selection with an explicit list (e.g. just
    `["Volkswagen", "BYD"]`, cutting this to 2 brands x 60 months = 120 requests) -- the same
    `build_query()` criteria is used either way, only which brands are queried changes.
    """
    raw_dir, resolved_start, resolved_end, resolved_top_n, default_rate_limit = (
        _resolve_gdelt_scope(raw_directory, start_date, end_date, top_n_brands)
    )
    resolved_rate_limit = (
        rate_limit_seconds if rate_limit_seconds is not None else default_rate_limit
    )
    chunk_kwargs = {} if max_attempts is None else {"max_attempts": max_attempts}
    raw_dir.mkdir(parents=True, exist_ok=True)

    queries = brand_queries(resolved_top_n, brands=brands)
    chunks = enumerate_chunks(queries, resolved_start, resolved_end, (GDELT_MODE_ARTICLES,))

    summary = {"fetched": 0, "cached": 0, "failed": 0, "total": len(chunks)}
    for brand, query, chunk_mode, window_start, window_end in iter_with_progress(
        chunks, total=len(chunks), desc="GDELT artlist chunks", time_budget=time_budget
    ):
        raw_path = chunk_raw_path(raw_dir, query, chunk_mode, window_start, window_end)
        metadata_path = raw_path.with_suffix(raw_path.suffix + ".metadata.json")
        if validate_local_cache(raw_path, metadata_path):
            summary["cached"] += 1
            continue
        if fetch_mode == "cached":
            summary["failed"] += 1
            continue
        try:
            download_gdelt_chunk(
                query, chunk_mode, window_start, window_end, raw_dir, **chunk_kwargs
            )
            summary["fetched"] += 1
        except (GdeltAccessError, GdeltDataError) as exc:
            logger.warning(
                "GDELT chunk failed (brand=%s, mode=%s, %s-%s): %s",
                brand,
                chunk_mode,
                window_start,
                window_end,
                exc,
            )
            summary["failed"] += 1
        time.sleep(resolved_rate_limit)

    logger.info(
        "GDELT artlist fetch summary: %s cached, %s newly fetched, %s failed, %s total",
        summary["cached"],
        summary["fetched"],
        summary["failed"],
        summary["total"],
    )
    return summary


def ensure_gdelt_dataset(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
    rate_limit_seconds: float | None = None,
    max_attempts: int | None = None,
    fetch_mode: str = "live",
    time_budget: TimeBudget | None = None,
) -> dict[str, dict[str, int]]:
    """Fetch both real GDELT modes: calls `ensure_gdelt_timelinevol_dataset()` then
    `ensure_gdelt_artlist_dataset()` in sequence and returns both summaries.

    The notebook calls the two phases separately instead (see their own docstrings), so
    `timelinevol`'s results/graphs can be shown before starting the much slower `artlist`
    phase -- this combined function exists for simpler callers that just want "fetch
    everything" (e.g. `scripts/download_gdelt_news.py`).
    """
    timelinevol_summary = ensure_gdelt_timelinevol_dataset(
        raw_directory=raw_directory,
        start_date=start_date,
        end_date=end_date,
        top_n_brands=top_n_brands,
        rate_limit_seconds=rate_limit_seconds,
        max_attempts=max_attempts,
        fetch_mode=fetch_mode,
        time_budget=time_budget,
    )
    artlist_summary = ensure_gdelt_artlist_dataset(
        raw_directory=raw_directory,
        start_date=start_date,
        end_date=end_date,
        top_n_brands=top_n_brands,
        rate_limit_seconds=rate_limit_seconds,
        max_attempts=max_attempts,
        fetch_mode=fetch_mode,
        time_budget=time_budget,
    )
    return {"timelinevol": timelinevol_summary, "artlist": artlist_summary}


def resolve_scope(
    raw_directory: str | Path | None,
    start_date: date | None,
    end_date: date | None,
    top_n_brands: int | None,
) -> tuple[Path, date, date, int]:
    raw_dir, resolved_start, resolved_end, resolved_top_n, _rate_limit = _resolve_gdelt_scope(
        raw_directory, start_date, end_date, top_n_brands
    )
    return raw_dir, resolved_start, resolved_end, resolved_top_n


def load_cached_article_frame(
    raw_dir: Path, start: date, end: date, top_n_brands: int, *, brands: Sequence[str] | None = None
) -> tuple[pd.DataFrame, dict[str, str], int, int]:
    """Load every real GDELT article chunk already cached under `raw_dir` for this scope.

    Returns `(combined_frame, brand_group_map, found_chunk_count, total_chunk_count)` --
    shared by `build_gdelt_analysis_dataset()` and `collect_article_urls()` so both report
    identical coverage without duplicating the chunk-scanning loop. `brand_group_map` always
    covers the full top-N-per-group selection regardless of `brands` (every brand this project
    queries is one of the top-N, so this stays correct), but `brands`, if given, narrows which
    chunks are actually scanned -- e.g. `analysis/media_geography.py` reuses this to load just
    Volkswagen/BYD's cached articles without scanning all 10 brands' chunk paths.
    """
    top_brands = get_top_brands(top_n_brands)
    brand_group_map = {
        brand: group for group, group_brands in top_brands.items() for brand in group_brands
    }
    queries = brand_queries(top_n_brands, brands=brands)
    chunks = enumerate_chunks(queries, start, end, (GDELT_MODE_ARTICLES,))

    frames: list[pd.DataFrame] = []
    found = 0
    for brand, query, mode, window_start, window_end in chunks:
        raw_path = chunk_raw_path(raw_dir, query, mode, window_start, window_end)
        if not raw_path.is_file():
            continue
        found += 1
        try:
            frames.append(tidy_article_chunk(raw_path, brand=brand))
        except GdeltDataError as exc:
            logger.warning("Skipping unparsable GDELT article chunk %s: %s", raw_path, exc)

    combined = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    if not combined.empty:
        combined["brand_group"] = combined["brand"].map(brand_group_map)
    return combined, brand_group_map, found, len(chunks)


def build_gdelt_analysis_dataset(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
) -> pd.DataFrame:
    """Build the GDELT annual article-count series from whatever article chunks are cached.

    Unlike KBA/Switzerland/Trends, this does not require every chunk to be present --
    real full 10-brand x 5-year GDELT coverage takes many resumed
    `ensure_gdelt_artlist_dataset()` runs (see its docstring), so this works over partial
    coverage and reports exactly how much of the full scope was actually found, rather than
    blocking until 100% complete. Raises `SourceUnavailableError` only if *no* article chunks
    are cached at all.
    """
    raw_dir, start, end, top_n = resolve_scope(raw_directory, start_date, end_date, top_n_brands)
    combined, brand_group_map, found, total = load_cached_article_frame(raw_dir, start, end, top_n)

    if found == 0:
        raise SourceUnavailableError(
            source="gdelt",
            reason=f"No cached GDELT article chunks found under {raw_dir}.",
            required_action=(
                "Run ensure_gdelt_artlist_dataset() (or scripts/download_gdelt_news.py) at "
                "least once -- pass a TimeBudget since real full coverage takes many resumed "
                "runs."
            ),
            accepted_fallback="There is no synthetic fallback; real GDELT data is required.",
        )

    logger.info(
        "Built GDELT article dataset from %s/%s expected (brand, month) chunks (%.0f%% coverage)",
        found,
        total,
        100 * found / total,
    )
    return build_gdelt_annual_series(combined, brand_group_map=brand_group_map)


def collect_article_urls(
    *,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
) -> list[str]:
    """Return every unique real article URL found in the cached GDELT article chunks.

    Feeds `article_text.fetch_article_texts()` -- kept separate from
    `build_gdelt_analysis_dataset()` since the latter returns the annual count series, not
    raw per-article rows.
    """
    raw_dir, start, end, top_n = resolve_scope(raw_directory, start_date, end_date, top_n_brands)
    combined, _brand_group_map, found, total = load_cached_article_frame(raw_dir, start, end, top_n)
    logger.info("Collected article URLs from %s/%s expected (brand, month) chunks", found, total)
    if combined.empty or "url" not in combined.columns:
        return []
    return combined["url"].dropna().drop_duplicates().tolist()
