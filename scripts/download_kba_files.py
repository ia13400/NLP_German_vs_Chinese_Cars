from __future__ import annotations

import argparse
from pathlib import Path

from car_interest_nlp.config import PROJECT_ROOT
from car_interest_nlp.data.kba import (
    FZ10_DEFAULT_YEARS,
    FZ10_LANDING_URL_TEMPLATE,
    discover_kba_files,
    download_kba_file,
)
from car_interest_nlp.logging_utils import configure_logging

logger = configure_logging()


def download_fz10_for_month(year: int, month: int, dest_dir: str | Path) -> Path | None:
    """Discover and download the real FZ10 file for one year/month. Returns None if unavailable."""
    landing_url = FZ10_LANDING_URL_TEMPLATE.format(year=year, month=month)
    links = discover_kba_files(landing_url)
    if not links:
        logger.warning(
            "No FZ10 download link found at %s (page may not exist for %04d-%02d yet)",
            landing_url,
            year,
            month,
        )
        return None
    destination = download_kba_file(links[0]["url"], dest_dir)
    logger.info("Downloaded FZ10 file for %04d-%02d -> %s", year, month, destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download the real KBA FZ10 (new registrations by brand/model series) "
        "file for a given month across multiple years."
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=list(FZ10_DEFAULT_YEARS),
        help="e.g. --years 2021 2022 2023",
    )
    parser.add_argument("--month", type=int, default=12, help="1-12, default 12 (December)")
    parser.add_argument(
        "--dest-dir",
        default=str(PROJECT_ROOT / "data" / "raw" / "registrations" / "kba"),
    )
    args = parser.parse_args()

    downloaded: list[Path] = []
    failed: list[int] = []
    for year in args.years:
        try:
            result = download_fz10_for_month(year, args.month, args.dest_dir)
            if result is not None:
                downloaded.append(result)
            else:
                failed.append(year)
        except Exception:
            logger.exception("Failed to download FZ10 for %04d-%02d", year, args.month)
            failed.append(year)

    print(f"Downloaded {len(downloaded)} file(s) to {args.dest_dir}")
    if failed:
        print(f"Failed or unavailable for years: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
