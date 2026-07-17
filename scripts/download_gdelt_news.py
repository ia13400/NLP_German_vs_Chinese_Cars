from __future__ import annotations

import argparse

from car_interest_nlp.data.gdelt import GDELT_DEFAULT_TIME_LIMIT_MINUTES
from car_interest_nlp.data.gdelt_dataset_builder import ensure_gdelt_dataset
from car_interest_nlp.progress import TimeBudget


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch (resumably) real GDELT DOC 2.0 article/timeline data for the "
        "top-5 German and top-5 Chinese car brands, 2021-2025, English sources only. "
        "timelinevol is ~10 requests (one per brand for the whole range, falling back to "
        "5 yearly requests per brand if that fails), artlist is ~600 (monthly, round-robin "
        "across brands), and GDELT's real rate-limit tolerance is far stricter in practice "
        "than its documented limit -- already-cached chunks are never re-fetched, so this "
        "is designed to be run repeatedly until coverage is complete."
    )
    parser.add_argument(
        "--time-limit-minutes",
        type=float,
        default=GDELT_DEFAULT_TIME_LIMIT_MINUTES,
        help="Stop after this many minutes.",
    )
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument(
        "--fetch-mode",
        choices=["live", "cached"],
        default="live",
        help="'live' fetches whatever is still missing; 'cached' only reports current "
        "coverage without sending any GDELT requests.",
    )
    args = parser.parse_args()

    summary = ensure_gdelt_dataset(
        raw_directory=args.raw_dir,
        fetch_mode=args.fetch_mode,
        time_budget=TimeBudget(minutes=args.time_limit_minutes),
    )
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
