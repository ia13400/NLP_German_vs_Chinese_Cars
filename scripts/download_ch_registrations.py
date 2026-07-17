from __future__ import annotations

import argparse

from car_interest_nlp.data.switzerland_dataset_builder import ensure_ch_dataset


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download (if not already present) and tidy the real Swiss "
        "new-registrations-by-make data (via the SDMX 2.1 REST API at "
        "disseminate.stats.swiss) into one reporting_period/brand/value_type/registrations "
        "interim CSV."
    )
    parser.add_argument(
        "--raw-dir", default=None, help="Defaults to data/raw/registrations/switzerland"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Defaults to data/interim/switzerland/ch_annual_brand_totals.csv",
    )
    args = parser.parse_args()

    interim_path = ensure_ch_dataset(raw_directory=args.raw_dir, interim_file=args.output)
    print(f"Swiss interim file ready at {interim_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
