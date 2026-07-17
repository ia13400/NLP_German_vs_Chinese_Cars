from __future__ import annotations

import argparse

from car_interest_nlp.data.trends_dataset_builder import ensure_trends_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch (if not already cached) and tidy real Google Trends "
        "search-interest data for every tracked German/Chinese brand (configs/brands.yaml) "
        "into one reporting_period/brand/value_type/search_interest_index interim CSV. "
        "Google's unofficial Trends endpoint allows at most 5 keywords per request and "
        "rate-limits/blocks automated requests, so this can take several minutes and may "
        "need to be re-run -- already-downloaded keyword batches under data/raw/trends/ "
        "are cached and never re-fetched."
    )
    parser.add_argument("--raw-dir", default=None, help="Defaults to data/raw/trends")
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to data/interim/trends/google_trends_brand_interest.csv",
    )
    parser.add_argument("--geo", default=None, help="Defaults to configs/sources.yaml's geo (DE)")
    parser.add_argument(
        "--timeframe", default=None, help="Defaults to configs/sources.yaml's timeframe (today 5-y)"
    )
    args = parser.parse_args()

    interim_path = ensure_trends_dataset(
        raw_directory=args.raw_dir,
        interim_file=args.output,
        geo=args.geo,
        timeframe=args.timeframe,
    )
    print(f"Google Trends interim file ready at {interim_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
