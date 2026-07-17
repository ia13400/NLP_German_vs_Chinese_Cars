from __future__ import annotations

import argparse

from car_interest_nlp.config import load_project_config
from car_interest_nlp.data.article_text import fetch_article_texts
from car_interest_nlp.data.gdelt_dataset_builder import collect_article_urls
from car_interest_nlp.progress import TimeBudget


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scrape (resumably, robots.txt-respecting) real full article text for "
        "every URL found in cached GDELT article chunks. Best-effort only -- many fetches "
        "will legitimately fail (paywalls, blocks, JS-only rendering) and are logged and "
        "skipped, not treated as errors. Not run automatically by any other script; this "
        "is the project's only entry point that fetches content from arbitrary third-party "
        "news domains rather than GDELT's own API."
    )
    parser.add_argument("--time-limit-minutes", type=float, default=15.0)
    parser.add_argument("--dest-dir", default=None)
    args = parser.parse_args()

    gdelt_config = load_project_config()["sources"].get("gdelt", {})
    scraping_config = gdelt_config.get("scraping", {})
    dest_dir = args.dest_dir or scraping_config.get("raw_directory", "data/raw/gdelt_articles")

    urls = collect_article_urls()
    if not urls:
        print("No cached GDELT article URLs found yet -- run scripts/download_gdelt_news.py first.")
        return 0

    summary = fetch_article_texts(
        urls,
        dest_dir,
        user_agent=scraping_config.get("user_agent", "car-interest-nlp-research-bot/1.0"),
        per_domain_delay_seconds=float(scraping_config.get("per_domain_delay_seconds", 3.0)),
        respect_robots_txt=bool(scraping_config.get("respect_robots_txt", True)),
        time_budget=TimeBudget(minutes=args.time_limit_minutes),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
