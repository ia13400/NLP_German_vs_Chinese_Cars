from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from pytrends.exceptions import ResponseError
from pytrends.request import TrendReq
from requests.exceptions import RequestException

from ..logging_utils import configure_logging
from ..preprocessing.brand_matching import BrandAlias, resolve_brands
from ..utils import stable_hash
from .collectors import write_cache_metadata
from .provenance import attach_provenance

logger = configure_logging()

PARSER_VERSION = "google-trends-1.0"
GOOGLE_TRENDS_WEBSITE = "https://trends.google.com/trends/explore"
VALUE_TYPE = "search_interest"

TRENDS_DEFAULT_GEO = "DE"
TRENDS_DEFAULT_HL = "de-DE"
TRENDS_DEFAULT_TZ = 60
TRENDS_DEFAULT_TIMEFRAME = "today 5-y"
TRENDS_DEFAULT_ANCHOR_BRAND = "Volkswagen"

# Matches KBA's FZ10_DEFAULT_YEARS and Switzerland's CH_DEFAULT_END_YEAR (both end 2025),
# so all three chapters compare the same 2021-2025 window. "today 5-y" is a *relative*
# window resolved at request/export time, so it naturally includes whatever partial
# current year has elapsed so far (e.g. a fetch/export made in mid-2026 includes a
# partial Jan-Jul 2026) -- annualize_trends_series() drops any year after this one so
# that trailing partial year is never averaged in as if it were a full year.
TRENDS_DEFAULT_END_YEAR = 2025

# Deliberately narrow, not the full configs/brands.yaml German/Chinese list: the unofficial
# Trends endpoint's 5-keyword-per-request limit, aggressive rate limiting/blocking, and
# several tracked brand names being ambiguous as bare search terms (e.g. "Mini", "Smart",
# "MAN") make a full multi-batch sweep unreliable as a data source. Volkswagen and BYD are
# each group's clearest, least ambiguous flagship brand, so this pair is used as a
# directional search-interest indicator only -- see README's "Google Trends" section.
TRENDS_DEFAULT_TRACKED_BRANDS: tuple[str, ...] = ("Volkswagen", "BYD")

# Google Trends accepts at most 5 keywords per request.
TRENDS_MAX_KEYWORDS_PER_REQUEST = 5

TRENDS_RAW_FILENAME_PREFIX = "google_trends_batch_"

# Conventional filename for a manually exported Trends CSV (downloaded by hand from
# trends.google.com's own UI export button) under a source's raw_directory. Checked by
# ensure_trends_dataset() before attempting any automated fetch -- see google_trends
# section in README.md ("Option B") for why a manual export is sometimes necessary (the
# unofficial pytrends endpoint's anonymous quota can be exhausted for extended periods).
TRENDS_MANUAL_EXPORT_FILENAME = "google_trends_manual_export.csv"


class GoogleTrendsAccessError(RuntimeError):
    """Raised when Google Trends repeatedly refuses a request (rate limit/blocked)."""


class GoogleTrendsDataError(ValueError):
    """Raised when Google Trends data cannot be validated or chained onto a common scale."""


def get_trend_brands(
    brand_config: dict, *, tracked_brands: Sequence[str] | None = None
) -> list[tuple[str, str]]:
    """Return (canonical_name, origin_group) pairs for the tracked German/Chinese brands.

    Only the German and Chinese groups are used as Google Trends keywords -- unlike the
    KBA/Switzerland registration adapters, there is no "Other/Miscellaneous" catch-all
    search term to query, since Google Trends measures interest in specific searched
    terms, not a fixed universe of all searches.

    `tracked_brands` (default `TRENDS_DEFAULT_TRACKED_BRANDS`) restricts the result to
    just those canonical names, in the order given -- see `TRENDS_DEFAULT_TRACKED_BRANDS`
    for why this project only queries Volkswagen and BYD by default rather than every
    brand in `configs/brands.yaml`. Raises if a requested name isn't a real
    german/chinese entry, since that would otherwise silently query nothing for it.
    """
    all_brands = {
        entry["canonical_name"]: (entry["canonical_name"], entry.get("origin_group", group_name))
        for group_name in ("german", "chinese")
        for entry in brand_config.get(group_name, [])
    }
    names = tracked_brands if tracked_brands is not None else TRENDS_DEFAULT_TRACKED_BRANDS
    missing = [name for name in names if name not in all_brands]
    if missing:
        raise GoogleTrendsDataError(
            f"tracked_brands {missing} are not german/chinese entries in configs/brands.yaml."
        )
    return [all_brands[name] for name in names]


def build_keyword_batches(
    brands: Sequence[tuple[str, str]],
    anchor_brand: str,
    *,
    batch_size: int = TRENDS_MAX_KEYWORDS_PER_REQUEST,
) -> list[list[str]]:
    """Split tracked brands into <=`batch_size` keyword batches, each carrying the anchor.

    Google Trends normalizes each request's values to 0-100 independently and allows at
    most 5 keywords per request. Including the same `anchor_brand` in every batch is what
    later makes `chain_trends_batches` able to rescale all batches onto one common,
    comparable index (see its docstring).
    """
    brand_names = [name for name, _ in brands]
    if anchor_brand not in brand_names:
        raise GoogleTrendsDataError(
            f"Anchor brand {anchor_brand!r} is not among the tracked brands: {brand_names}"
        )
    others = [name for name in brand_names if name != anchor_brand]
    chunk_size = max(1, batch_size - 1)
    return [
        [anchor_brand] + others[start : start + chunk_size]
        for start in range(0, len(others), chunk_size)
    ]


def _batch_cache_key(keywords: Sequence[str], geo: str, timeframe: str) -> str:
    return stable_hash(f"{sorted(keywords)}|{geo}|{timeframe}")[:16]


def batch_raw_path(dest_dir: str | Path, keywords: Sequence[str], geo: str, timeframe: str) -> Path:
    key = _batch_cache_key(keywords, geo, timeframe)
    return Path(dest_dir) / f"{TRENDS_RAW_FILENAME_PREFIX}{key}.csv"


def _fetch_interest_over_time(
    keywords: Sequence[str],
    *,
    geo: str,
    timeframe: str,
    hl: str,
    tz: int,
    max_attempts: int,
    initial_backoff_seconds: float,
) -> pd.DataFrame:
    """Fetch one batch's real interest-over-time data, retrying on Google's rate limit.

    A fresh `TrendReq` is created per attempt (it fetches a new session cookie on
    construction), and failures back off exponentially -- the unofficial Trends endpoint
    returns HTTP 429 aggressively, and a fixed short retry delay tends to just repeat the
    same failure.
    """
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            client = TrendReq(hl=hl, tz=tz)
            client.build_payload(list(keywords), timeframe=timeframe, geo=geo)
            frame = client.interest_over_time()
            if frame.empty:
                raise GoogleTrendsDataError(
                    f"Google Trends returned no data for {list(keywords)!r} "
                    f"(geo={geo!r}, timeframe={timeframe!r})."
                )
            return frame
        except (ResponseError, RequestException) as exc:
            last_error = exc
            if attempt < max_attempts:
                wait_seconds = initial_backoff_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Google Trends request failed (attempt %s/%s) for %s: %s; retrying in %.0fs",
                    attempt,
                    max_attempts,
                    keywords,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
    raise GoogleTrendsAccessError(
        f"Google Trends request repeatedly failed for {list(keywords)!r}: {last_error}"
    ) from last_error


def download_trends_batch(
    keywords: Sequence[str],
    dest_dir: str | Path,
    *,
    geo: str = TRENDS_DEFAULT_GEO,
    timeframe: str = TRENDS_DEFAULT_TIMEFRAME,
    hl: str = TRENDS_DEFAULT_HL,
    tz: int = TRENDS_DEFAULT_TZ,
    max_attempts: int = 4,
    initial_backoff_seconds: float = 30.0,
) -> Path:
    """Download and cache one real Google Trends batch (weekly interest, one column per keyword)."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = batch_raw_path(dest_dir, keywords, geo, timeframe)
    frame = _fetch_interest_over_time(
        keywords,
        geo=geo,
        timeframe=timeframe,
        hl=hl,
        tz=tz,
        max_attempts=max_attempts,
        initial_backoff_seconds=initial_backoff_seconds,
    )
    frame.to_csv(destination, encoding="utf-8")
    query = "%2C".join(keywords)
    write_cache_metadata(
        destination, source_url=f"{GOOGLE_TRENDS_WEBSITE}?q={query}&geo={geo}&date={timeframe}"
    )
    return destination


def tidy_trends_batch(path: str | Path, keywords: Sequence[str]) -> pd.DataFrame:
    """Tidy one cached raw batch CSV into long date/brand/raw_index rows.

    Weeks Google marks `isPartial` (the most recent, not-yet-complete week) are dropped
    rather than treated as a full week's value.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    if "isPartial" in frame.columns:
        frame = frame[~frame["isPartial"].astype(bool).fillna(False)]
        frame = frame.drop(columns=["isPartial"])

    missing = set(keywords) - set(frame.columns)
    if missing:
        raise GoogleTrendsDataError(
            f"Cached batch at {path} is missing expected keyword columns: {sorted(missing)}"
        )
    long_frame = frame[["date", *keywords]].melt(
        id_vars="date", var_name="brand", value_name="raw_index"
    )
    return long_frame


def tidy_manual_trends_export(path: str | Path) -> pd.DataFrame:
    """Tidy a CSV manually exported from the Google Trends website's own UI.

    Unlike `tidy_trends_batch()` (which expects the pytrends JSON response shape, with
    columns matching an exact keyword list this project itself requested), a manual
    export's column headers are whatever label Google's UI assigned the chosen search
    term/topic -- e.g. "BYD Auto" rather than "BYD" -- so every non-date column is melted
    as-is here and left to `normalize_trends_brands()`/`configs/brands.yaml`'s alias map to
    resolve to a canonical brand, exactly as KBA/Switzerland already resolve source-specific
    raw brand spellings. There is no `isPartial` column in a manual export (that is an
    artifact of the pytrends JSON API response only), but one is still dropped if present
    for consistency with `tidy_trends_batch()`.
    """
    frame = pd.read_csv(path, encoding="utf-8")
    date_column = frame.columns[0]
    frame = frame.rename(columns={date_column: "date"})
    frame["date"] = pd.to_datetime(frame["date"])
    if "isPartial" in frame.columns:
        frame = frame[~frame["isPartial"].astype(bool).fillna(False)]
        frame = frame.drop(columns=["isPartial"])

    value_columns = [column for column in frame.columns if column != "date"]
    if not value_columns:
        raise GoogleTrendsDataError(f"Manual Google Trends export at {path} has no brand columns.")
    return frame[["date", *value_columns]].melt(
        id_vars="date", var_name="brand", value_name="raw_index"
    )


def chain_trends_batches(batch_frames: Sequence[pd.DataFrame], anchor_brand: str) -> pd.DataFrame:
    """Rescale independently-normalized Google Trends batches onto one common index.

    Google Trends normalizes each request's values to 0-100 relative to that request's
    own peak, and accepts at most 5 keywords per request. With more than 5 tracked
    brands, every batch was fetched together with the same `anchor_brand` (see
    `build_keyword_batches`); each batch's raw values are rescaled here by the ratio
    between the anchor's mean value in that batch and its mean value in the first
    ("reference") batch. Without this, summing/comparing raw values across batches would
    be meaningless: a batch with no large brand in it shows inflated values purely from
    having less competition for the 0-100 scale, not genuinely higher interest.
    """
    if not batch_frames:
        raise GoogleTrendsDataError("No Google Trends batches to chain.")

    reference = batch_frames[0]
    reference_anchor_mean = reference.loc[reference["brand"] == anchor_brand, "raw_index"].mean()
    if pd.isna(reference_anchor_mean) or reference_anchor_mean == 0:
        raise GoogleTrendsDataError(
            f"Anchor brand {anchor_brand!r} has no usable signal in the reference batch; "
            "cannot chain other batches onto it."
        )

    chained_frames: list[pd.DataFrame] = []
    anchor_already_kept = False
    for batch in batch_frames:
        batch_anchor_mean = batch.loc[batch["brand"] == anchor_brand, "raw_index"].mean()
        if pd.isna(batch_anchor_mean) or batch_anchor_mean == 0:
            raise GoogleTrendsDataError(
                f"Anchor brand {anchor_brand!r} has no usable signal in one of the batches; "
                "cannot chain it onto the reference scale."
            )
        ratio = reference_anchor_mean / batch_anchor_mean
        scaled = batch.copy()
        scaled["search_interest_index"] = scaled["raw_index"] * ratio
        # The anchor appears in every batch by construction; keep only its first
        # (reference-batch) occurrence so it isn't duplicated/double-counted downstream.
        if anchor_already_kept:
            scaled = scaled[scaled["brand"] != anchor_brand]
        else:
            anchor_already_kept = True
        chained_frames.append(scaled[["date", "brand", "search_interest_index"]])

    return pd.concat(chained_frames, ignore_index=True)


def annualize_trends_series(
    long_frame: pd.DataFrame, *, end_year: int | None = TRENDS_DEFAULT_END_YEAR
) -> pd.DataFrame:
    """Aggregate chained weekly/monthly interest into an annual mean per brand.

    Matches the annual granularity of the KBA/Switzerland registration series so the
    three sources can be compared side by side over the same "last 5 years" window.
    `end_year` (default `TRENDS_DEFAULT_END_YEAR`, 2025) drops any year after it --
    "today 5-y" is a relative window resolved at request/export time, so it always
    includes whatever partial current year has elapsed so far; averaging that partial
    year in as if it were complete would understate/overstate it relative to the full
    calendar years around it. Pass `end_year=None` to keep every year present (e.g. for
    ad-hoc inspection of the raw chained series).
    """
    data = long_frame.copy()
    if end_year is not None:
        data = data[data["date"].dt.year <= end_year]
    data["reporting_period"] = data["date"].dt.year.astype(str)
    annual = (
        data.groupby(["reporting_period", "brand"])["search_interest_index"].mean().reset_index()
    )
    annual["value_type"] = VALUE_TYPE
    return annual[["reporting_period", "brand", "value_type", "search_interest_index"]]


def normalize_trends_brands(
    frame: pd.DataFrame,
    alias_map: dict[str, BrandAlias],
    *,
    brand_column: str = "brand",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map Google Trends keyword strings to canonical project brands.

    Unlike KBA/Switzerland (which resolve raw, source-supplied brand strings that may not
    match anything), every Trends keyword was chosen directly from `configs/brands.yaml`
    in `get_trend_brands`, so a keyword failing to resolve indicates a real bug (a stale
    keyword vs. a since-edited brands.yaml) rather than an expected "Other/Miscellaneous"
    case -- `fallback_canonical_name=None` returns such rows as `unresolved` instead of
    silently bucketing them.
    """
    return resolve_brands(
        frame,
        alias_map,
        brand_column=brand_column,
        fallback_canonical_name=None,
        fallback_origin_group=None,
    )


def validate_trends_data(frame: pd.DataFrame) -> None:
    """Validate a tidied, brand-resolved Trends frame before building the annual series."""
    required = {"reporting_period", "brand", "value_type", "search_interest_index"}
    missing = required.difference(frame.columns)
    if missing:
        raise GoogleTrendsDataError(
            f"Google Trends frame is missing required columns: {sorted(missing)}"
        )
    unknown_value_types = set(frame["value_type"].unique()) - {VALUE_TYPE}
    if unknown_value_types:
        raise GoogleTrendsDataError(
            f"Google Trends frame has unknown value_type entries: {sorted(unknown_value_types)}"
        )
    if (frame["search_interest_index"] < 0).any():
        raise GoogleTrendsDataError("Google Trends frame contains negative interest values.")


def build_trends_annual_series(
    frame: pd.DataFrame, *, geo: str = TRENDS_DEFAULT_GEO
) -> pd.DataFrame:
    """Build the Google Trends annual search-interest series from a resolved, tidy frame.

    `google_trends_interest_share` is a brand's share of the *tracked German+Chinese
    brands'* total chained interest that year -- not a share of all Google searches (no
    such total exists/is measurable). This is a materially different denominator than
    `kba_registration_share`/`ch_registration_share`, which are shares of every real
    registered car including "Other/Miscellaneous"; the two must not be read as the same
    kind of percentage.
    """
    validate_trends_data(frame)
    data = frame[frame["value_type"] == VALUE_TYPE].copy()
    data = data.rename(columns={"origin_group": "brand_group"})
    totals = (
        data.groupby("reporting_period")["search_interest_index"]
        .sum()
        .rename("all_tracked_brand_interest")
    )
    grouped = (
        data.groupby(["reporting_period", "canonical_brand", "brand_group"])[
            "search_interest_index"
        ]
        .sum()
        .reset_index()
    )
    grouped = grouped.merge(totals, on="reporting_period", how="left")
    grouped["google_trends_interest_share"] = (
        grouped["search_interest_index"] / grouped["all_tracked_brand_interest"]
    )
    grouped = attach_provenance(
        grouped,
        source_type="google_trends",
        source_name="Google Trends",
        source_url=f"{GOOGLE_TRENDS_WEBSITE}?geo={geo}",
        parser_version=PARSER_VERSION,
        collection_method="web_search_api",
        license_note="Google Trends relative search-interest index (trends.google.com); "
        "not an official registration or sales statistic.",
        reporting_period_column="reporting_period",
    )
    return grouped
