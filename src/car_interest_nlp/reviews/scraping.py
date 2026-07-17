from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import httpx
import pandas as pd
from bs4 import BeautifulSoup

from ..config import PROJECT_ROOT, load_project_config
from ..data.article_text import is_allowed_by_robots
from ..data.collectors import write_cache_metadata
from ..data.validators import validate_local_cache
from ..logging_utils import configure_logging
from ..progress import TimeBudget, iter_with_progress
from ..utils import stable_hash

logger = configure_logging()

CARWALE_BASE_URL = "https://www.carwale.com"
REVIEW_CHUNK_PREFIX = "reviews_"
DEFAULT_RAW_DIRECTORY = PROJECT_ROOT / "data" / "raw" / "reviews"

# CarWale's real review-card markup, confirmed directly while porting this scraper from the
# original project: a listing page with fewer cards than this is the last page for that model.
DEFAULT_MIN_CARDS_PER_PAGE = 10
DEFAULT_MODELS_PER_BRAND = 4
DEFAULT_MAX_PAGES_PER_MODEL = 20

# CarWale's brand-slug labels map directly onto canonical brand names already present in
# configs/brands.yaml -- reused via preprocessing.brand_matching instead of a separate
# hardcoded country map.
CARWALE_BRAND_SLUG_TO_CANONICAL: dict[str, str] = {
    "byd-cars": "BYD",
    "mg-cars": "MG",
    "volkswagen-cars": "Volkswagen",
    "bmw-cars": "BMW",
    "mercedes-benz-cars": "Mercedes-Benz",
}


class ReviewScrapeError(RuntimeError):
    """Raised when a CarWale page cannot be fetched at all (not just empty of reviews)."""


def reviews_config_defaults() -> dict:
    return load_project_config()["sources"].get("reviews", {})


def _model_slug_from_url(model_review_url: str) -> str:
    """Extract the model slug from a CarWale review URL, e.g.
    'https://www.carwale.com/byd-cars/seal/reviews/' -> 'seal'.
    """
    parts = [part for part in model_review_url.split("/") if part]
    return parts[-2] if len(parts) >= 2 and parts[-1] == "reviews" else parts[-1]


def discover_model_urls(
    brand_slugs: Sequence[str],
    *,
    models_per_brand: int = DEFAULT_MODELS_PER_BRAND,
    timeout: float = 15.0,
) -> list[dict[str, str]]:
    """Discover real CarWale model-review URLs for each brand slug.

    Opens each brand's overview page (e.g. 'https://www.carwale.com/byd-cars') and extracts
    the first `models_per_brand` model links, turning each into its review-listing URL. No
    User-Agent header is sent for this page, matching the original working scraper -- only
    the per-page review fetch below needs the browser-like UA (see `scrape_model_reviews`).
    Every URL is checked against carwale.com's real robots.txt first (cached per domain, see
    `data.article_text.is_allowed_by_robots`) -- confirmed directly that its `Disallow` rules
    do not cover brand overview or `/reviews/` pages, only account/search/API paths.
    """
    entries: list[dict[str, str]] = []
    for brand_slug in brand_slugs:
        brand_url = f"{CARWALE_BASE_URL}/{brand_slug}"
        if not is_allowed_by_robots(brand_url, timeout=timeout):
            logger.warning("Skipping %s: disallowed by robots.txt", brand_url)
            continue
        logger.info("Discovering models for %s", brand_url)
        response = httpx.get(brand_url, timeout=timeout, follow_redirects=True)
        if response.status_code != 200:
            logger.warning(
                "Failed to fetch brand page %s (HTTP %s)", brand_url, response.status_code
            )
            continue
        soup = BeautifulSoup(response.content, "html.parser")
        cars = soup.find_all("a", class_="o-C o-os", href=True, limit=models_per_brand)
        for car in cars:
            model_url = f"{CARWALE_BASE_URL}{car['href']}reviews/"
            entries.append(
                {
                    "brand_slug": brand_slug,
                    "model_slug": _model_slug_from_url(model_url),
                    "model_url": model_url,
                }
            )
    return entries


def scrape_model_reviews(
    model_review_url: str,
    *,
    brand_slug: str,
    max_pages: int = DEFAULT_MAX_PAGES_PER_MODEL,
    min_cards_per_page: int = DEFAULT_MIN_CARDS_PER_PAGE,
    user_agent: str = "Mozilla/5.0",
    rate_limit_seconds: float = 1.0,
    timeout: float = 15.0,
) -> list[dict[str, object]]:
    """Scrape every real review card for one CarWale model, paging until a short page is hit.

    Mirrors the original notebook's extraction rules exactly: review title, comment text,
    and star rating are read from each card, and a page with fewer than
    `min_cards_per_page` cards is treated as the last page (CarWale fills pages to exactly
    10 cards otherwise). A real per-request delay (`rate_limit_seconds`) is enforced between
    page fetches, unlike the original notebook, out of politeness toward the real site.
    Checked against carwale.com's real robots.txt first, same as `discover_model_urls`.
    """
    if not is_allowed_by_robots(model_review_url, user_agent, timeout=timeout):
        logger.warning("Skipping %s: disallowed by robots.txt", model_review_url)
        return []

    headers = {"User-Agent": user_agent}
    reviews: list[dict[str, object]] = []
    for page_number in range(1, max_pages + 1):
        page_url = f"{model_review_url}page/{page_number}"
        response = httpx.get(page_url, headers=headers, timeout=timeout, follow_redirects=True)
        if response.status_code != 200:
            logger.warning(
                "Failed to fetch review page %s (HTTP %s)", page_url, response.status_code
            )
            break

        soup = BeautifulSoup(response.content, "html.parser")
        cards = soup.find_all(
            "li", class_=lambda css_class: css_class and "oxygen-card-wrapper" in css_class
        )
        number_of_cards = len(cards)

        for card in cards:
            title = card.find("a", href=lambda href: href and "/reviews/" in href)
            comment = card.find("div", attrs={"color": "dimGray"})
            stars = card.find_all("svg", attrs={"aria-label": "rating icon"})
            filled_stars = [
                star for star in stars if "o-k3" in cast(list[str], star.get("class") or [])
            ]
            reviews.append(
                {
                    "Title": title.get_text(" ", strip=True) if title else None,
                    "Comment": comment.get_text(" ", strip=True) if comment else None,
                    "Rating": len(filled_stars),
                    "brand_slug": brand_slug,
                }
            )

        if number_of_cards < min_cards_per_page:
            break
        time.sleep(rate_limit_seconds)
    return reviews


def _review_chunk_key(brand_slug: str, model_slug: str) -> str:
    return stable_hash(f"{brand_slug}|{model_slug}")[:16]


def review_chunk_path(raw_dir: str | Path, brand_slug: str, model_slug: str) -> Path:
    key = _review_chunk_key(brand_slug, model_slug)
    return Path(raw_dir) / f"{REVIEW_CHUNK_PREFIX}{brand_slug}_{model_slug}_{key}.json"


def _iter_review_chunk_files(raw_dir: str | Path) -> list[Path]:
    """List every real review chunk file under `raw_dir`, excluding `.metadata.json` sidecars.

    `Path.glob("reviews_*.json")` also matches each chunk's own `*.json.metadata.json`
    sidecar (its name still ends in `.json`) -- confirmed directly: without this filter, a
    metadata file's flat string values get iterated as bare characters when later combined
    into a DataFrame, raising a confusing pandas `ValueError`.
    """
    return sorted(
        path
        for path in Path(raw_dir).glob(f"{REVIEW_CHUNK_PREFIX}*.json")
        if not path.name.endswith(".metadata.json")
    )


def ensure_reviews_dataset(
    *,
    brand_slugs: Sequence[str],
    models_per_brand: int = DEFAULT_MODELS_PER_BRAND,
    max_pages: int = DEFAULT_MAX_PAGES_PER_MODEL,
    min_cards_per_page: int = DEFAULT_MIN_CARDS_PER_PAGE,
    raw_directory: str | Path | None = None,
    user_agent: str = "Mozilla/5.0",
    rate_limit_seconds: float = 1.0,
    mode: str = "live",
    time_budget: TimeBudget | None = None,
) -> dict[str, int]:
    """Fetch (or, in `mode="cached"`, just report) real CarWale reviews, cached per model.

    Each (brand, model) pair is cached as its own JSON chunk under `raw_directory`, exactly
    like GDELT's per-chunk caching (`data/gdelt_dataset_builder.py`) -- already-cached models
    are always skipped, so a re-run resumes rather than re-scraping. `mode="cached"` never
    sends a single request: it only counts whatever chunk files already exist on disk against
    the expected `len(brand_slugs) * models_per_brand` total, for re-running the notebook
    (e.g. to redo training/plots) without touching the real site again. `mode="live"`
    discovers the real current model list for each brand (this discovery step always makes a
    real request, even for already-fully-cached brands, since CarWale's model lineup can
    change) and then scrapes whatever model chunks are still missing.
    """
    raw_dir = Path(raw_directory) if raw_directory is not None else DEFAULT_RAW_DIRECTORY
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_total = len(brand_slugs) * models_per_brand
    summary = {"cached": 0, "fetched": 0, "failed": 0, "total": expected_total}

    if mode == "cached":
        existing = _iter_review_chunk_files(raw_dir)
        summary["cached"] = len(existing)
        summary["failed"] = max(0, expected_total - len(existing))
        logger.info(
            "CarWale reviews (cached mode): %s/%s expected model chunks already cached under %s",
            summary["cached"],
            expected_total,
            raw_dir,
        )
        return summary

    if mode != "live":
        raise ValueError(f"Unknown execution mode: {mode!r}")

    model_entries = discover_model_urls(brand_slugs, models_per_brand=models_per_brand)
    summary["total"] = len(model_entries)

    for entry in iter_with_progress(
        model_entries, total=len(model_entries), desc="CarWale reviews", time_budget=time_budget
    ):
        destination = review_chunk_path(raw_dir, entry["brand_slug"], entry["model_slug"])
        metadata_path = destination.with_suffix(destination.suffix + ".metadata.json")
        if validate_local_cache(destination, metadata_path):
            summary["cached"] += 1
            continue
        try:
            reviews = scrape_model_reviews(
                entry["model_url"],
                brand_slug=entry["brand_slug"],
                max_pages=max_pages,
                min_cards_per_page=min_cards_per_page,
                user_agent=user_agent,
                rate_limit_seconds=rate_limit_seconds,
            )
            destination.write_text(
                json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            write_cache_metadata(destination, source_url=entry["model_url"])
            summary["fetched"] += 1
        except (httpx.HTTPError, OSError) as exc:
            logger.warning("Failed to scrape %s: %s", entry["model_url"], exc)
            summary["failed"] += 1
        time.sleep(rate_limit_seconds)

    logger.info(
        "CarWale reviews fetch summary: %s cached, %s newly fetched, %s failed, %s total",
        summary["cached"],
        summary["fetched"],
        summary["failed"],
        summary["total"],
    )
    return summary


def load_cached_reviews(*, raw_directory: str | Path | None = None) -> pd.DataFrame:
    """Combine every cached CarWale review chunk under `raw_directory` into one DataFrame."""
    raw_dir = Path(raw_directory) if raw_directory is not None else DEFAULT_RAW_DIRECTORY
    records: list[dict[str, object]] = []
    for chunk_path in _iter_review_chunk_files(raw_dir):
        try:
            records.extend(json.loads(chunk_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            logger.warning("Skipping unparsable cached review chunk %s", chunk_path)
    return pd.DataFrame(records, columns=["Title", "Comment", "Rating", "brand_slug"])
