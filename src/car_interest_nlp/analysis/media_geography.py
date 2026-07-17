from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from ..data.gdelt_dataset_builder import load_cached_article_frame, resolve_scope
from ..logging_utils import configure_logging

logger = configure_logging()

DEFAULT_DOMINANCE_BRANDS: tuple[str, str] = ("Volkswagen", "BYD")
DOMINANCE_COLUMNS = [
    "year",
    "sourcecountry",
    "brand_a",
    "brand_b",
    "count_a",
    "count_b",
    "total_count",
    "dominance_score",
]


def build_country_dominance(
    *,
    brands: Sequence[str] = DEFAULT_DOMINANCE_BRANDS,
    years: Sequence[int] = (2021, 2025),
    min_total_count: int = 2,
    raw_directory: str | Path | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
) -> pd.DataFrame:
    """Real per-country, per-year GDELT article-count dominance between exactly two brands.

    Built from the same real cached `artlist` chunks as the article-count/wordcloud/NER
    chapters -- specifically `sourcecountry`, GDELT's real per-article source-country field (a
    full English country name, e.g. "Germany", "United States", confirmed directly against
    real cached data). This requires no separate GDELT request: it is a client-side
    aggregation over whatever `artlist` coverage `ensure_gdelt_artlist_dataset(brands=brands)`
    has already fetched for `brands`.

    Exactly two brands are compared (`brands` must have length 2): for each (year, country)
    with at least `min_total_count` combined real articles, `dominance_score` is
    `(count_a - count_b) / (count_a + count_b)`, ranging over [-1, 1] -- +1 means every real
    article GDELT attributed to that country's media mentioned only the first brand, -1 only
    the second, 0 an even split. Countries below `min_total_count` are dropped rather than
    shown as a noisy +-1 score derived from a single article. This measures which brand a
    country's *English-language news media* wrote about more, not brand popularity or market
    share within that country's population.
    """
    if len(brands) != 2:
        raise ValueError(f"build_country_dominance compares exactly 2 brands, got {list(brands)!r}")
    brand_a, brand_b = brands

    raw_dir, start, end, top_n = resolve_scope(raw_directory, start_date, end_date, top_n_brands)
    combined, _brand_group_map, found, total = load_cached_article_frame(
        raw_dir, start, end, top_n, brands=brands
    )
    logger.info(
        "Built country-dominance data from %s/%s expected (brand, month) chunks for %s",
        found,
        total,
        list(brands),
    )
    if combined.empty or "sourcecountry" not in combined.columns:
        return pd.DataFrame(columns=DOMINANCE_COLUMNS)

    data = combined.copy()
    data["seendate"] = pd.to_datetime(data["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    data["year"] = data["seendate"].dt.year
    # GDELT sometimes returns "" (not NaN) for `sourcecountry` when it couldn't determine the
    # publishing outlet's country -- confirmed directly in real cached data (19 such rows) --
    # `.notna()` alone lets these through as a bogus phantom "country".
    has_country = data["sourcecountry"].notna() & (data["sourcecountry"].str.strip() != "")
    data = data[data["year"].isin(years) & has_country]
    if data.empty:
        return pd.DataFrame(columns=DOMINANCE_COLUMNS)

    counts = data.groupby(["year", "sourcecountry", "brand"]).size().unstack("brand", fill_value=0)
    for brand in (brand_a, brand_b):
        if brand not in counts.columns:
            counts[brand] = 0
    counts = counts.rename(columns={brand_a: "count_a", brand_b: "count_b"})[["count_a", "count_b"]]
    counts["total_count"] = counts["count_a"] + counts["count_b"]
    counts = counts[counts["total_count"] >= min_total_count]
    counts["dominance_score"] = (counts["count_a"] - counts["count_b"]) / counts["total_count"]
    counts["brand_a"] = brand_a
    counts["brand_b"] = brand_b

    return (
        counts.reset_index()[DOMINANCE_COLUMNS]
        .sort_values(["year", "sourcecountry"])
        .reset_index(drop=True)
    )
