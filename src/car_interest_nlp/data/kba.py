from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse, urlsplit

import httpx
import pandas as pd
from bs4 import BeautifulSoup
from openpyxl.utils import column_index_from_string

from ..logging_utils import configure_logging
from ..preprocessing.brand_matching import BrandAlias, resolve_brands
from .collectors import write_cache_metadata
from .provenance import attach_provenance

logger = configure_logging()

PARSER_VERSION = "kba-1.0"
KBA_OFFICIAL_HOSTS = {"kba.de", "www.kba.de"}
KBA_WEBSITE = "https://www.kba.de/"
VALUE_TYPES = ("new_registrations", "stock")

# Real FZ10 sheets list each brand's individual model series followed by a summary row
# whose label is the brand name plus " ZUSAMMEN" (German for "combined"/"total"), e.g.
# "AUDI ZUSAMMEN", "BYD ZUSAMMEN". Confirmed directly against the real file structure.
FZ10_BRAND_TOTAL_ROW_PATTERN = re.compile(r"^(.+?)\s+ZUSAMMEN$", re.IGNORECASE)

# KBA has renamed the FZ10 detail sheet across releases -- confirmed directly: "FZ10.1"
# (no space) in the 2021/2022 files, "FZ 10.1" (with a space) from 2023 onward. Both are
# tried in order; an unrecognized future naming still raises a clear error listing the
# file's actual sheet names rather than guessing.
FZ10_SHEET_NAME_CANDIDATES: tuple[str, ...] = ("FZ 10.1", "FZ10.1")

# Real KBA landing-page pattern for the monthly "FZ 10" table (new passenger-car
# registrations by brand/model series), confirmed against kba.de for 2021-2025. The
# actual download link is only found by discovering it from this specific landing page
# per month -- it embeds an unpredictable CMS version parameter
# (`?__blob=publicationFile&v=N`) that cannot be guessed directly.
FZ10_LANDING_URL_TEMPLATE = "https://www.kba.de/SharedDocs/Downloads/DE/Statistik/Fahrzeuge/FZ10/fz10_{year}_{month:02d}.html"

# Matches the filenames produced by download_kba_file() (the CMS query string is
# stripped from the local filename).
FZ10_RAW_FILENAME_PATTERN = re.compile(r"^fz10_(\d{4})_(\d{2})\.xlsx$", re.IGNORECASE)

FZ10_DEFAULT_YEARS: tuple[int, ...] = tuple(range(2021, 2026))
FZ10_DEFAULT_MONTH = 12

# KBA publishes some brand groupings that don't line up 1:1 with a single alias in
# configs/brands.yaml (e.g. "MG ROEWE" bundles what the project tracks separately as MG).
# Only well-known, unambiguous KBA spellings are remapped here; anything else is left for
# the unresolved-brands review table rather than guessed.
KBA_BRAND_SPELLING_OVERRIDES: dict[str, str] = {
    "mg roewe": "mg motor",
    "great wall": "gwm",
}


class KbaAccessError(ValueError):
    """Raised when a KBA URL does not point at the official kba.de domain."""


class KbaDataError(ValueError):
    """Raised when KBA data cannot be validated (mixed stock/new-registration values, etc.)."""


def _assert_official_domain(url: str) -> None:
    host = urlparse(url).netloc.lower()
    if host not in KBA_OFFICIAL_HOSTS:
        raise KbaAccessError(f"Refusing to use non-official KBA URL (host={host!r}): {url}")


def discover_kba_files(
    listing_url: str, *, client: httpx.Client | None = None
) -> list[dict[str, str]]:
    """Discover downloadable XLSX/CSV links on an official KBA listing page.

    KBA's registration overview pages link to further subpages rather than files
    directly, so `listing_url` must point at the specific subpage where the desired
    table (e.g. monthly new registrations by brand) is actually linked. Real KBA
    download links are served through a CMS blob endpoint with a query string (e.g.
    `fz10_2023_12.xlsx?__blob=publicationFile&v=2`), so the file extension is matched
    on the URL's path component, not a bare suffix check on the full href.
    """
    _assert_official_domain(listing_url)
    owns_client = client is None
    client = client or httpx.Client(timeout=30.0, follow_redirects=True)
    try:
        response = client.get(
            listing_url, headers={"User-Agent": "car-interest-nlp-research-bot/1.0"}
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
    finally:
        if owns_client:
            client.close()
    links: list[dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = str(anchor["href"])
        href_path = urlsplit(href).path
        if href_path.lower().endswith((".xlsx", ".xls", ".csv")):
            absolute_url = str(httpx.URL(listing_url).join(href))
            links.append({"url": absolute_url, "filename": href_path.rsplit("/", 1)[-1]})
    if not links:
        logger.warning(
            "No XLSX/CSV links found at %s; KBA page structure may have changed", listing_url
        )
    return links


def download_kba_file(
    url: str, dest_dir: str | Path, *, client: httpx.Client | None = None
) -> Path:
    """Download a single file from the official KBA domain and stamp cache metadata.

    The filename is derived from the URL's path component only (ignoring any CMS query
    string like `?__blob=publicationFile&v=2`), since a raw query string would otherwise
    end up embedded in the local filename (invalid on Windows).
    """
    _assert_official_domain(url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    filename = urlsplit(url).path.rsplit("/", 1)[-1] or "kba_download"
    destination = dest_dir / filename
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        response = client.get(url, headers={"User-Agent": "car-interest-nlp-research-bot/1.0"})
        response.raise_for_status()
        destination.write_bytes(response.content)
    finally:
        if owns_client:
            client.close()
    write_cache_metadata(destination, source_url=url)
    return destination


def load_kba_xlsx(path: str | Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load a KBA XLSX/CSV download and record which sheet(s) it came from.

    Returns the raw sheet largely as-is: KBA table layouts vary by report type (new
    registrations vs. stock, by-brand vs. by-model-series), so this deliberately does
    not guess a universal tidy schema. Use it to inspect the real file, then feed the
    relevant brand/registration columns into `normalize_kba_brands`/`build_kba_monthly_series`.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path, encoding="utf-8")
        frame.attrs["sheet_names"] = [path.name]
        return frame
    excel_file = pd.ExcelFile(path)
    sheet = sheet_name if sheet_name is not None else excel_file.sheet_names[0]
    frame = excel_file.parse(sheet)
    frame.attrs["sheet_names"] = excel_file.sheet_names
    frame.attrs["selected_sheet"] = sheet
    return frame


def tidy_fz10_annual_totals(
    path: str | Path,
    year: int,
    *,
    sheet_name: str | Sequence[str] = FZ10_SHEET_NAME_CANDIDATES,
    value_column: str = "E",
) -> pd.DataFrame:
    """Extract per-brand annual registration totals from a real KBA FZ10 sheet.

    Real FZ10 sheets list each brand's individual model series followed by a
    "<BRAND> ZUSAMMEN" row summing that brand's total (e.g. "AUDI ZUSAMMEN", "BYD
    ZUSAMMEN"). This scans every row for a cell matching that pattern -- checked across
    all columns, since the exact column holding the brand label isn't fixed -- and reads
    the corresponding value from `value_column` (column E by default, whose header is
    "Jan.-Dezember <year>" for a December report, i.e. the cumulative annual total).

    `sheet_name` accepts a single name or a sequence of candidate names tried in order
    (default `FZ10_SHEET_NAME_CANDIDATES`, both confirmed real spellings KBA has used
    across releases); the first one present in the file is used. If none match, the
    error lists the file's actual sheet names rather than guessing.

    `year` is taken as an explicit argument rather than parsed from the sheet's header
    text, since the caller (iterating over one downloaded file per year) already knows
    it unambiguously. Rows whose value in `value_column` can't be parsed as a number are
    skipped and logged -- never guessed.
    """
    candidates = [sheet_name] if isinstance(sheet_name, str) else list(sheet_name)
    with pd.ExcelFile(path) as excel_file:
        matched_sheet = next((name for name in candidates if name in excel_file.sheet_names), None)
        if matched_sheet is None:
            raise KbaDataError(
                f"None of the candidate sheet names {candidates} found in {path}. "
                f"Available sheets: {excel_file.sheet_names}. Pass sheet_name=... explicitly "
                "for this file."
            )
        raw = excel_file.parse(matched_sheet, header=None)
    value_col_index = column_index_from_string(value_column.upper()) - 1

    rows: list[dict[str, object]] = []
    for _, row in raw.iterrows():
        brand = None
        for cell in row:
            if isinstance(cell, str):
                match = FZ10_BRAND_TOTAL_ROW_PATTERN.match(cell.strip())
                if match:
                    brand = match.group(1).strip()
                    break
        if brand is None:
            continue
        raw_value = row.iloc[value_col_index] if value_col_index < len(row) else None
        value = pd.to_numeric(pd.Series([raw_value]), errors="coerce").iloc[0]
        if pd.isna(value):
            logger.warning(
                "Skipping unparsable registration value for brand %r in %s (column %s)",
                brand,
                path,
                value_column,
            )
            continue
        rows.append({"brand": brand, "registrations": int(value)})

    result = pd.DataFrame(rows, columns=["brand", "registrations"])
    result["reporting_period"] = str(year)
    result["value_type"] = "new_registrations"
    return result[["reporting_period", "brand", "value_type", "registrations"]]


def normalize_kba_brands(
    frame: pd.DataFrame,
    alias_map: dict[str, BrandAlias],
    *,
    brand_column: str = "brand",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw KBA brand strings to canonical project brands.

    Returns `(resolved, unresolved)`. Rows whose raw brand string cannot be confidently
    mapped are never guessed -- they are returned separately so they can be written to
    artifacts/tables/unresolved_kba_brands.csv for manual review.
    """
    return resolve_brands(
        frame,
        alias_map,
        brand_column=brand_column,
        spelling_overrides=KBA_BRAND_SPELLING_OVERRIDES,
    )


def validate_kba_data(frame: pd.DataFrame) -> None:
    """Validate a loaded/tidied KBA frame before building a monthly series."""
    required = {"reporting_period", "brand", "value_type", "registrations"}
    missing = required.difference(frame.columns)
    if missing:
        raise KbaDataError(f"KBA frame is missing required columns: {sorted(missing)}")
    unknown_value_types = set(frame["value_type"].unique()) - set(VALUE_TYPES)
    if unknown_value_types:
        raise KbaDataError(
            f"KBA frame has unknown value_type entries: {sorted(unknown_value_types)}"
        )
    if frame["value_type"].nunique() > 1:
        raise KbaDataError(
            "KBA frame mixes new_registrations and stock in the same table; split them "
            "before building a monthly series (annual stock and monthly new registrations "
            "must never be summed together)."
        )
    if (frame["registrations"] < 0).any():
        raise KbaDataError("KBA frame contains negative registration counts.")


def build_kba_monthly_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the kba_registration_share time series from a validated, brand-resolved frame.

    Requires `frame` to already carry `canonical_brand`/`origin_group` (from
    `normalize_kba_brands`); `origin_group` is renamed to `brand_group` here to match the
    naming used everywhere else in the project, and is carried through the groupby so
    German-vs-Chinese comparisons remain possible on the returned series.
    """
    validate_kba_data(frame)
    data = frame[frame["value_type"] == "new_registrations"].copy()
    data = data.rename(columns={"origin_group": "brand_group"})
    totals = (
        data.groupby("reporting_period")["registrations"]
        .sum()
        .rename("all_passenger_car_registrations")
    )
    grouped = (
        data.groupby(["reporting_period", "canonical_brand", "brand_group"])["registrations"]
        .sum()
        .rename("brand_registrations")
        .reset_index()
    )
    grouped = grouped.merge(totals, on="reporting_period", how="left")
    grouped["kba_registration_share"] = (
        grouped["brand_registrations"] / grouped["all_passenger_car_registrations"]
    )
    grouped = attach_provenance(
        grouped,
        source_type="kba",
        source_name="Kraftfahrt-Bundesamt (KBA)",
        source_url=KBA_WEBSITE,
        parser_version=PARSER_VERSION,
        collection_method="structured_download",
        license_note="Kraftfahrt-Bundesamt open data, official German government statistics.",
        reporting_period_column="reporting_period",
    )
    return grouped
