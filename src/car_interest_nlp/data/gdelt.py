from __future__ import annotations

import json
import random
import time
from datetime import date
from email.utils import parsedate_to_datetime
from pathlib import Path

import httpx
import pandas as pd

from ..logging_utils import configure_logging
from ..utils import stable_hash
from .collectors import write_cache_metadata
from .provenance import attach_provenance

logger = configure_logging()

PARSER_VERSION = "gdelt-doc2-1.0"
GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_WEBSITE = "https://www.gdeltproject.org/"

GDELT_DEFAULT_SOURCE_LANG = "eng"
GDELT_DEFAULT_RATE_LIMIT_SECONDS = 6.0
GDELT_DEFAULT_TOP_N_BRANDS = 5
# 360 minutes (6 hours) by default -- GDELT's real rate-limit tolerance is far stricter in
# practice than its documented "one request per 5 seconds" (confirmed directly), so a
# generous default budget is needed to make meaningful progress across ~1200 chunks even
# with the retry strategy below already spending real time waiting out 429s.
GDELT_DEFAULT_TIME_LIMIT_MINUTES = 360.0

# GDELT's DOC 2.0 API modes used here. "artlist" returns per-article metadata (url, title,
# seendate, domain, language, sourcecountry) -- no full article body text, hence the separate
# opt-in scraping module (article_text.py) for that. "timelinevol" is GDELT's own aggregate
# volume-over-time endpoint, used for the media-attention-over-time analysis so that doesn't
# require fetching (or scraping) individual articles at all. (A third mode, "timelinesourcecountry",
# was used earlier for a by-world-region breakdown; removed to cut required requests by a
# third -- see README's "GDELT News Analysis" section.)
GDELT_MODE_ARTICLES = "artlist"
GDELT_MODE_TIMELINE_VOLUME = "timelinevol"

GDELT_RAW_FILENAME_PREFIX = "gdelt_"

# Retry policy for HTTP 429 (and other transient failures): retry the *same* request rather
# than moving on to a different one, honoring a real `Retry-After` response header when GDELT
# sends one, otherwise a jittered exponential backoff matching this exact schedule (attempt N's
# range is the wait *before* attempt N+1): 1->60-120s, 2->120-240s, 3-14->150-300s (flat).
# `_BACKOFF_MAX_WAIT_SECONDS` is a hard ceiling on a single wait (never more than 5 minutes)
# rather than a fixed attempt index to freeze growth at -- an absolute-seconds cap stays
# correct regardless of `_BACKOFF_BASE_SECONDS`, unlike freezing at "attempt N" (which only
# bounds the max wait correctly for the specific base value it was tuned against).
# `GDELT_DEFAULT_MAX_ATTEMPTS = 15` gives 14 waits before giving up on that one chunk (worst
# case ~66 minutes total); requests are always sequential, never parallel.
GDELT_DEFAULT_MAX_ATTEMPTS = 15
_BACKOFF_BASE_SECONDS = 60.0
_BACKOFF_MAX_WAIT_SECONDS = 300.0

# Confirmed directly against the real API on 2026-07-15: a quoted phrase search of just
# `"BMW"` (3 characters) returned HTTP 200 with the plain-text body "The specified phrase is
# too short." rather than a JSON result -- GDELT's phrase-search parser has a minimum length.
# A second, real GDELT error ("Your search contained a keyword that was too short") was later
# observed for short *bare/unquoted* brand tokens too (e.g. "MG", 2 characters) -- so a
# length-based conditional-quoting scheme (quote only if long enough) doesn't reliably avoid
# either error for every short brand name. The query is therefore always built as the brand
# name concatenated with "car" (e.g. `"MG car"`) and always quoted as one exact phrase: this
# keeps the brand name from ever appearing as its own short standalone token (quoted or not),
# and the combined phrase is comfortably long enough to clear both length rules for every real
# brand in this project's lists.
GDELT_QUOTED_PHRASE_SUFFIX = "car"


class GdeltAccessError(RuntimeError):
    """Raised when GDELT repeatedly refuses a request (rate limit) or is unreachable."""


class GdeltDataError(ValueError):
    """Raised when a GDELT response cannot be parsed into the shape a mode is expected to have."""


def build_query(
    brand: str,
    *,
    source_country: str | None = None,
    domain: str | None = None,
    near: tuple[str, str, int] | None = None,
) -> str:
    """Build a GDELT DOC 2.0 query string for `brand` that avoids GDELT's short-keyword
    rejections.

    The query is just `"<brand> car"` -- a single quoted exact phrase, never a bare token and
    never an unquoted multi-word query (which GDELT would otherwise split into separate ANDed
    single-word terms) -- see `GDELT_QUOTED_PHRASE_SUFFIX`'s comment for the two real
    short-keyword errors this avoids. An earlier version also ANDed a "german"/"chinese"
    context word onto the phrase, but that was dropped after a real, direct check found it
    made matches far too sparse (`"Volkswagen car" german` matched only ~13% of days, at
    roughly 1/100th the coverage-share values of a broader query) -- the brand+"car" phrase
    alone is narrow enough to avoid the short-keyword errors without also cutting real
    coverage this drastically.

    `source_country` is a GDELT `sourcecountry:` filter value, `domain` a `domain:` filter,
    and `near` an optional `(word_a, word_b, max_distance)` proximity filter rendered as
    GDELT's `nearN:"word_a word_b"` operator -- all appended as additional ANDed clauses.
    """
    clauses = [f'"{brand} {GDELT_QUOTED_PHRASE_SUFFIX}"', f"sourcelang:{GDELT_DEFAULT_SOURCE_LANG}"]
    if source_country:
        clauses.append(f"sourcecountry:{source_country}")
    if domain:
        clauses.append(f"domain:{domain}")
    if near:
        word_a, word_b, distance = near
        clauses.append(f'near{distance}:"{word_a} {word_b}"')
    return " ".join(clauses)


def month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into whole calendar months as (first_day, first_day_of_next) pairs."""
    windows: list[tuple[date, date]] = []
    cursor = date(start.year, start.month, 1)
    while cursor <= end:
        next_month = date(cursor.year + (cursor.month // 12), (cursor.month % 12) + 1, 1)
        windows.append((cursor, next_month))
        cursor = next_month
    return windows


def year_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into whole calendar years as (first_day, first_day_of_next) pairs.

    Safe for GDELT's `timelinevol` mode specifically: **confirmed directly against the real
    API** that a full calendar-year window returns one real daily datapoint per day (365 for
    a non-leap year) with no truncation -- unlike `artlist`'s hard `maxrecords=250` cap,
    `timelinevol` has no per-request record limit, so widening the window purely cuts request
    count with zero data loss. Do not use this for `artlist`: real cached monthly chunks for
    high-volume brands (Volkswagen, Mercedes-Benz) already return exactly 250/250 articles
    every month, meaning a year-wide `artlist` window would silently drop most of the year's
    real coverage (only the most recent ~250 articles, sorted `datedesc`, would come back).
    """
    windows: list[tuple[date, date]] = []
    cursor = date(start.year, 1, 1)
    while cursor <= end:
        next_year = date(cursor.year + 1, 1, 1)
        windows.append((cursor, next_year))
        cursor = next_year
    return windows


def full_range_window(start: date, end: date) -> list[tuple[date, date]]:
    """A single (start, end) window spanning the whole requested range -- one request per
    brand total, rather than one per year.

    Only safe for GDELT modes with no per-request record cap. **Confirmed directly, but only
    up to a one-year window**: `timelinevol` returned the full 365 real daily datapoints for
    a full calendar year, no truncation (see `year_windows`'s docstring). Extending that same
    single-request-per-brand approach across the *entire* multi-year range (e.g. 2021-2025)
    is architecturally consistent with that evidence -- `timelinevol` is a server-side
    computed statistic, not a capped document sample like `artlist` -- but has **not been
    directly verified for a multi-year window**. Recommended before relying on this: run one
    live request for a single brand across the full configured range and confirm the
    response still contains one real datapoint per real day with no gaps/truncation, the same
    way the one-year case was confirmed. If that check ever fails, `year_windows` remains
    available as an already-confirmed fallback granularity for this mode.
    """
    return [(start, end)]


def _gdelt_datetime(value: date) -> str:
    return value.strftime("%Y%m%d000000")


def _chunk_cache_key(query: str, mode: str, start: date, end: date) -> str:
    return stable_hash(f"{query}|{mode}|{start.isoformat()}|{end.isoformat()}")[:16]


def chunk_raw_path(dest_dir: str | Path, query: str, mode: str, start: date, end: date) -> Path:
    key = _chunk_cache_key(query, mode, start, end)
    return Path(dest_dir) / f"{GDELT_RAW_FILENAME_PREFIX}{mode}_{key}.json"


def _jittered_backoff_seconds(attempt: int) -> float:
    """Exponential backoff with random jitter for the wait *after* `attempt` fails.

    Doubles each attempt (1->60-120s, 2->120-240s) until doing so would exceed
    `_BACKOFF_MAX_WAIT_SECONDS`, then flattens at a 150-300s tier for every further attempt --
    both the low and high end of the jitter range are clamped against the hard ceiling, so the
    flattened tier still has a real jitter width instead of collapsing to one fixed wait, and
    no single wait ever exceeds 5 minutes regardless of how many attempts are configured.
    """
    raw_low = _BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
    low = min(raw_low, _BACKOFF_MAX_WAIT_SECONDS / 2)
    high = min(raw_low * 2, _BACKOFF_MAX_WAIT_SECONDS)
    return random.uniform(low, high)


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a real `Retry-After` response header (delay-seconds or HTTP-date form)."""
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        target = parsedate_to_datetime(header)
    except (TypeError, ValueError):
        return None
    if target.tzinfo is None:
        return None
    from datetime import datetime as _datetime

    return max(0.0, (target - _datetime.now(target.tzinfo)).total_seconds())


def _fetch_gdelt(
    query: str,
    mode: str,
    start: date,
    end: date,
    *,
    timeout: float,
    max_attempts: int = GDELT_DEFAULT_MAX_ATTEMPTS,
    max_records: int = 250,
) -> dict:
    """Fetch one real GDELT DOC 2.0 response, retrying the *same* request on rate limiting.

    On HTTP 429, a real `Retry-After` response header is honored when GDELT sends one;
    otherwise `_jittered_backoff_seconds` is used. GDELT sometimes returns HTTP 200 with a
    plain-text (non-JSON) error body instead of an HTTP error code (confirmed directly, see
    `GDELT_QUOTED_PHRASE_SUFFIX`'s comment) -- a failed `response.json()` parse containing that
    same rate-limit message is therefore retried the same way, not just non-2xx status
    codes. Requests are always sequential (one at a time), never parallel.
    """
    params = {
        "query": query,
        "mode": mode,
        "format": "json",
        "startdatetime": _gdelt_datetime(start),
        "enddatetime": _gdelt_datetime(end),
    }
    if mode == GDELT_MODE_ARTICLES:
        params["maxrecords"] = str(max_records)
        params["sort"] = "datedesc"

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = httpx.get(
                GDELT_DOC_ENDPOINT,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "car-interest-nlp-research-bot/1.0"},
            )
            rate_limited_body = None
            if response.status_code != 429:
                try:
                    response.raise_for_status()
                    return response.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    body = response.text.strip()
                    if "limit requests" not in body.lower():
                        raise GdeltDataError(
                            f"GDELT returned a non-JSON response for query {query!r} "
                            f"(mode={mode!r}): {body[:200]}"
                        ) from exc
                    rate_limited_body = body

            retry_after = _retry_after_seconds(response)
            wait_seconds = (
                retry_after if retry_after is not None else _jittered_backoff_seconds(attempt)
            )
            reason = rate_limited_body[:200] if rate_limited_body else "HTTP 429"
            last_error = GdeltAccessError(f"GDELT rate-limited the request ({reason}): {query!r}")
            if attempt < max_attempts:
                logger.warning(
                    "GDELT rate-limited (attempt %s/%s) for %r (mode=%s); retrying in %.0fs (%s)",
                    attempt,
                    max_attempts,
                    query,
                    mode,
                    wait_seconds,
                    "Retry-After" if retry_after is not None else "backoff",
                )
                time.sleep(wait_seconds)
        except (httpx.HTTPStatusError, httpx.TransportError) as exc:
            last_error = exc
            if attempt < max_attempts:
                wait_seconds = _jittered_backoff_seconds(attempt)
                logger.warning(
                    "GDELT request failed (attempt %s/%s) for %r (mode=%s): %s; retrying in %.0fs",
                    attempt,
                    max_attempts,
                    query,
                    mode,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)
    raise GdeltAccessError(
        f"GDELT request repeatedly failed for {query!r} (mode={mode!r}): {last_error}"
    ) from last_error


def download_gdelt_chunk(
    query: str,
    mode: str,
    start: date,
    end: date,
    dest_dir: str | Path,
    *,
    timeout: float = 30.0,
    max_attempts: int = GDELT_DEFAULT_MAX_ATTEMPTS,
) -> Path:
    """Download and cache one real GDELT response for a (query, mode, month) chunk."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = chunk_raw_path(dest_dir, query, mode, start, end)
    payload = _fetch_gdelt(
        query,
        mode,
        start,
        end,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    query_url = f"{GDELT_DOC_ENDPOINT}?query={query}&mode={mode}"
    write_cache_metadata(destination, source_url=query_url)
    return destination


def tidy_article_chunk(path: str | Path, *, brand: str) -> pd.DataFrame:
    """Tidy one cached `artlist` chunk into one row per real article."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    articles = payload.get("articles")
    if articles is None:
        raise GdeltDataError(
            f"Cached GDELT artlist chunk at {path} has no 'articles' key "
            f"(actual keys: {sorted(payload.keys())})"
        )
    frame = pd.DataFrame(articles)
    frame["brand"] = brand
    return frame


def tidy_timeline_chunk(path: str | Path, *, brand: str, mode: str) -> pd.DataFrame:
    """Tidy one cached GDELT timeline-mode (e.g. `timelinevol`) chunk into long rows.

    GDELT's timeline modes return `{"timeline": [{"series": ..., "data": [{"date": ...,
    "value": ...}, ...]}, ...]}` -- one series per matched query term/breakdown category.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    timeline = payload.get("timeline")
    if timeline is None:
        raise GdeltDataError(
            f"Cached GDELT {mode} chunk at {path} has no 'timeline' key "
            f"(actual keys: {sorted(payload.keys())})"
        )
    rows: list[dict[str, object]] = []
    for series in timeline:
        series_name = series.get("series", mode)
        for point in series.get("data", []):
            rows.append(
                {
                    "brand": brand,
                    "series": series_name,
                    "date": point.get("date"),
                    "value": point.get("value"),
                }
            )
    return pd.DataFrame(rows)


def build_gdelt_annual_series(
    article_frame: pd.DataFrame, *, brand_group_map: dict[str, str]
) -> pd.DataFrame:
    """Build a per-brand, per-year article-count series with provenance from tidied articles.

    `brand_group_map` maps each canonical brand name to its `brand_group` ("german"/
    "chinese"), matching the shape used by the other three chapters -- there is no
    "Other/Miscellaneous" row here since, like Google Trends, every brand queried was
    chosen directly rather than parsed from a raw report.
    """
    if article_frame.empty:
        return article_frame
    data = article_frame.copy()
    data["seendate"] = pd.to_datetime(data["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    data["reporting_period"] = data["seendate"].dt.year.astype("Int64").astype(str)
    data["brand_group"] = data["brand"].map(brand_group_map)
    data["value_type"] = "news_article_count"

    grouped = (
        data.groupby(["reporting_period", "brand", "brand_group"])
        .size()
        .rename("article_count")
        .reset_index()
    )
    grouped = grouped.rename(columns={"brand": "canonical_brand"})
    grouped["value_type"] = "news_article_count"
    grouped = attach_provenance(
        grouped,
        source_type="gdelt",
        source_name="GDELT Project (DOC 2.0 API)",
        source_url=GDELT_DOC_ENDPOINT,
        parser_version=PARSER_VERSION,
        collection_method="structured_download",
        license_note="GDELT Project data; article metadata (title/url/date/source) is "
        "publicly published under GDELT's open access terms.",
        reporting_period_column="reporting_period",
    )
    return grouped
