from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from car_interest_nlp.config import PROJECT_ROOT
from car_interest_nlp.data.kba import (
    FZ10_RAW_FILENAME_PATTERN,
    FZ10_SHEET_NAME_CANDIDATES,
    tidy_fz10_annual_totals,
)
from car_interest_nlp.logging_utils import configure_logging

logger = configure_logging()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tidy downloaded real KBA FZ10 files into one "
        "reporting_period/brand/value_type/registrations CSV for --mode manual_import."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "registrations" / "kba"),
        help="Directory containing the downloaded fz10_<year>_<month>.xlsx files.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "interim" / "kba" / "kba_annual_brand_totals.csv"),
    )
    parser.add_argument(
        "--sheet-name",
        nargs="+",
        default=list(FZ10_SHEET_NAME_CANDIDATES),
        help="Candidate FZ10 sheet name(s) to try, in order (KBA's naming has changed "
        "across releases, e.g. 'FZ 10.1' vs 'FZ10.1').",
    )
    parser.add_argument(
        "--value-column", default="E", help="Column with the annual total, e.g. 'E'"
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    frames: list[pd.DataFrame] = []
    for file_path in sorted(raw_dir.glob("fz10_*.xlsx")):
        match = FZ10_RAW_FILENAME_PATTERN.match(file_path.name)
        if not match:
            logger.warning(
                "Skipping %s: filename doesn't match fz10_<year>_<month>.xlsx", file_path
            )
            continue
        year = int(match.group(1))
        try:
            frame = tidy_fz10_annual_totals(
                file_path, year, sheet_name=args.sheet_name, value_column=args.value_column
            )
        except Exception:
            logger.exception("Failed to tidy %s", file_path)
            continue
        logger.info("Extracted %s brand totals for %s from %s", len(frame), year, file_path.name)
        frames.append(frame)

    if not frames:
        print(f"No fz10_*.xlsx files found under {raw_dir}")
        return 1

    combined = pd.concat(frames, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False, encoding="utf-8")
    print(
        f"Wrote {len(combined)} rows ({combined['reporting_period'].nunique()} years) to {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
