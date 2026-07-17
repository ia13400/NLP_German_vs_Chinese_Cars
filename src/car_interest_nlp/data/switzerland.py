from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import httpx
import pandas as pd

from ..logging_utils import configure_logging
from ..preprocessing.brand_matching import BrandAlias, resolve_brands
from .collectors import write_cache_metadata
from .kba import validate_kba_data as validate_tidy_series
from .provenance import attach_provenance

logger = configure_logging()

PARSER_VERSION = "ch-new-registrations-1.0"
CH_OFFICIAL_HOSTS = {"disseminate.stats.swiss"}
CH_WEBSITE = "https://www.stats.swiss/"

# Federal Roads Office (FEDRO) new-passenger-car-registrations-by-make dataflow, disseminated
# through the Swiss Federal Statistical Office's public SDMX 2.1 REST API. Confirmed directly
# against the real API (the dataflow's `disseminate.stats.swiss/rest` base URL is embedded in
# the stats.swiss Data Explorer page's client config).
CH_FLOW_REF = "CH1.MFZ_IVS,DF_IVS_1_MAKE,1.0.0"
CH_DATA_BASE_URL = f"https://disseminate.stats.swiss/rest/data/{CH_FLOW_REF}"

# SDMX key selecting: MAKE=all (wildcard), HGDE_KT (canton)=_T (national total, no
# canton breakdown), REGISTRATION_TYPE=N (first registrations of *new* vehicles only, not
# "U" used/second-hand or "_T" total), FUEL=_T (all fuel types combined), FREQ=A (annual).
# Confirmed against the dataflow's actual dimension order and codelists.
CH_DATA_KEY = "._T.N._T.A"

CH_CODELIST_AGENCY = "CH1.MFZ_IVS"
CH_CODELIST_ID = "CL_RV_MAKE"
CH_CODELIST_URL = (
    f"https://disseminate.stats.swiss/rest/codelist/{CH_CODELIST_AGENCY}/{CH_CODELIST_ID}"
)

CH_DEFAULT_START_YEAR = 2021
CH_DEFAULT_END_YEAR = 2025

CH_DATA_RAW_FILENAME = "ch_new_registrations_by_make.csv"
CH_CODELIST_RAW_FILENAME = "ch_make_codelist.json"

# The MAKE codelist's "_T" code is the grand total across all makes, not a real brand --
# excluded before brand normalization so it's never mistaken for one.
_CH_TOTAL_MAKE_CODE = "_T"


class SwitzerlandAccessError(ValueError):
    """Raised when a Switzerland (stats.swiss) URL does not point at the official host."""


class SwitzerlandDataError(ValueError):
    """Raised when the Swiss new-registrations data or codelist cannot be validated."""


def _assert_official_domain(url: str) -> None:
    host = urlparse(url).netloc.lower()
    if host not in CH_OFFICIAL_HOSTS:
        raise SwitzerlandAccessError(
            f"Refusing to use non-official Switzerland URL (host={host!r}): {url}"
        )


def download_ch_data_file(
    dest_dir: str | Path,
    *,
    start_year: int = CH_DEFAULT_START_YEAR,
    end_year: int = CH_DEFAULT_END_YEAR,
    client: httpx.Client | None = None,
) -> Path:
    """Download the real Swiss new-registrations-by-make data via the SDMX 2.1 REST API.

    Queries `CH_DATA_KEY` (national total, new registrations only, all fuel types,
    annual) directly rather than downloading the full multi-dimensional dataset (which
    also breaks out by canton/registration-type/fuel and is >60 MB).
    """
    url = (
        f"{CH_DATA_BASE_URL}/{CH_DATA_KEY}?startPeriod={start_year}&endPeriod={end_year}&format=csv"
    )
    _assert_official_domain(url)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / CH_DATA_RAW_FILENAME
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


def download_ch_codelist_file(dest_dir: str | Path, *, client: httpx.Client | None = None) -> Path:
    """Download the real MAKE codelist (numeric code -> brand name) via the SDMX REST API.

    The data API only returns numeric MAKE codes, not brand names -- this codelist is
    required to resolve them. Requested as SDMX-JSON (via the Accept header) rather than
    the default SDMX-ML/XML, which is ~6x larger for the same content.
    """
    _assert_official_domain(CH_CODELIST_URL)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    destination = dest_dir / CH_CODELIST_RAW_FILENAME
    owns_client = client is None
    client = client or httpx.Client(timeout=60.0, follow_redirects=True)
    try:
        response = client.get(
            CH_CODELIST_URL,
            headers={
                "User-Agent": "car-interest-nlp-research-bot/1.0",
                "Accept": "application/vnd.sdmx.structure+json;version=1.0",
            },
        )
        response.raise_for_status()
        destination.write_bytes(response.content)
    finally:
        if owns_client:
            client.close()
    write_cache_metadata(destination, source_url=CH_CODELIST_URL)
    return destination


def _load_make_code_to_name(codelist_path: str | Path) -> dict[str, str]:
    with Path(codelist_path).open(encoding="utf-8") as f:
        payload = json.load(f)
    codelists = payload.get("data", {}).get("codelists", [])
    codelist = next((cl for cl in codelists if cl.get("id") == CH_CODELIST_ID), None)
    if codelist is None:
        raise SwitzerlandDataError(f"Codelist {CH_CODELIST_ID!r} not found in {codelist_path}.")
    return {code["id"]: code.get("name", code["id"]) for code in codelist["codes"]}


def tidy_ch_new_registrations(data_path: str | Path, codelist_path: str | Path) -> pd.DataFrame:
    """Tidy the real Swiss SDMX data CSV into reporting_period/brand/value_type/registrations.

    Maps each `UV_RV_MAKE` numeric code to its brand name via the codelist, drops the "_T"
    (grand total across all makes) row so it's never mistaken for a real brand, and skips
    rows with a missing/unparsable `OBS_VALUE` (some make/year combinations have no
    reported figure) rather than guessing a value.
    """
    frame = pd.read_csv(data_path, encoding="utf-8")
    required = {"UV_RV_MAKE", "TIME_PERIOD", "OBS_VALUE"}
    missing = required.difference(frame.columns)
    if missing:
        raise SwitzerlandDataError(
            f"Swiss data file is missing expected columns: {sorted(missing)}"
        )

    code_to_name = _load_make_code_to_name(codelist_path)
    data = frame[frame["UV_RV_MAKE"] != _CH_TOTAL_MAKE_CODE].copy()
    data["registrations"] = pd.to_numeric(data["OBS_VALUE"], errors="coerce")
    unparsable = data["registrations"].isna()
    if unparsable.any():
        logger.warning(
            "Skipping %s Swiss rows with missing/unparsable OBS_VALUE", int(unparsable.sum())
        )
    data = data[~unparsable]

    data["brand"] = data["UV_RV_MAKE"].map(code_to_name)
    unmapped = data["brand"].isna()
    if unmapped.any():
        logger.warning(
            "Skipping %s Swiss rows whose MAKE code has no entry in the codelist",
            int(unmapped.sum()),
        )
    data = data[~unmapped]

    data["reporting_period"] = data["TIME_PERIOD"].astype(str)
    data["value_type"] = "new_registrations"
    data["registrations"] = data["registrations"].astype(int)
    return data[["reporting_period", "brand", "value_type", "registrations"]].reset_index(drop=True)


def normalize_ch_brands(
    frame: pd.DataFrame,
    alias_map: dict[str, BrandAlias],
    *,
    brand_column: str = "brand",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map raw Swiss brand strings to canonical project brands.

    Returns `(resolved, unresolved)`; unresolved rows are never guessed -- they're written
    to artifacts/tables/unresolved_switzerland_brands.csv for manual review.
    """
    return resolve_brands(frame, alias_map, brand_column=brand_column)


def build_ch_annual_series(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the ch_registration_share time series from a validated, brand-resolved frame.

    Structurally identical to `build_kba_monthly_series` (this is genuinely the same kind
    of measurement -- an annual national new-registration flow -- just for Switzerland
    instead of Germany), but kept as its own function with its own column name
    (`ch_registration_share`, not `kba_registration_share`) and provenance so the two
    countries' series are never merged or confused, and can still be compared side by
    side as two independently computed national flows.
    """
    validate_tidy_series(frame)
    data = frame[frame["value_type"] == "new_registrations"].copy()
    data = data.rename(columns={"origin_group": "brand_group"})
    totals = (
        data.groupby("reporting_period")["registrations"]
        .sum()
        .rename("all_new_registrations_switzerland")
    )
    grouped = (
        data.groupby(["reporting_period", "canonical_brand", "brand_group"])["registrations"]
        .sum()
        .rename("brand_registrations")
        .reset_index()
    )
    grouped = grouped.merge(totals, on="reporting_period", how="left")
    grouped["ch_registration_share"] = (
        grouped["brand_registrations"] / grouped["all_new_registrations_switzerland"]
    )
    grouped = attach_provenance(
        grouped,
        source_type="ch_new_registrations",
        source_name="Bundesamt für Statistik (BFS) / Bundesamt für Strassen (ASTRA)",
        source_url=CH_DATA_BASE_URL,
        parser_version=PARSER_VERSION,
        collection_method="structured_download",
        license_note="Swiss Federal Statistical Office open data (stats.swiss).",
        reporting_period_column="reporting_period",
    )
    return grouped
