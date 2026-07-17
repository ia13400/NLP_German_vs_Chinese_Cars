from __future__ import annotations

import argparse

from car_interest_nlp.config import load_project_config
from car_interest_nlp.progress import TimeBudget
from car_interest_nlp.reviews.scraping import DEFAULT_MAX_PAGES_PER_MODEL, ensure_reviews_dataset


def main() -> int:
    config = load_project_config()
    reviews_config = config["sources"]["reviews"]

    parser = argparse.ArgumentParser(
        description="Scrape (resumably) real CarWale vehicle-model reviews, cached one JSON "
        "chunk per (brand, model). Already-cached models are never re-scraped."
    )
    parser.add_argument(
        "--brands",
        nargs="+",
        default=reviews_config["tracked_brands"],
        help="CarWale brand slugs, e.g. byd-cars volkswagen-cars.",
    )
    parser.add_argument("--models-per-brand", type=int, default=reviews_config["models_per_brand"])
    parser.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES_PER_MODEL)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument(
        "--mode",
        choices=["live", "cached"],
        default="live",
        help="'live' scrapes whatever is still missing; 'cached' only reports current "
        "coverage without sending any request.",
    )
    parser.add_argument("--time-limit-minutes", type=float, default=60.0)
    args = parser.parse_args()

    summary = ensure_reviews_dataset(
        brand_slugs=args.brands,
        models_per_brand=args.models_per_brand,
        max_pages=args.max_pages,
        raw_directory=args.raw_dir,
        mode=args.mode,
        time_budget=TimeBudget(minutes=args.time_limit_minutes),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
