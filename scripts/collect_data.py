from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from car_interest_nlp.config import PROJECT_ROOT, load_project_config
from car_interest_nlp.data.errors import SourceUnavailableError
from car_interest_nlp.data.provenance import build_data_source_registry, write_data_source_registry
from car_interest_nlp.logging_utils import configure_logging
from car_interest_nlp.paths import ARTIFACTS_DIR

logger = configure_logging()

MODES = ("cached", "live", "manual_import")


def _resolve_local_dir(source_config: dict[str, Any]) -> Path | None:
    raw_value = source_config.get("raw_directory") or source_config.get("path")
    if not raw_value:
        return None
    path = PROJECT_ROOT / raw_value
    return path.parent if path.suffix else path


def _collect_cached(name: str, source_config: dict[str, Any]) -> None:
    """Cached mode: validate previously downloaded real files. Performs no network calls."""
    local_dir = _resolve_local_dir(source_config)
    if local_dir is None or not local_dir.exists() or not any(local_dir.iterdir()):
        raise SourceUnavailableError(
            source=name,
            reason=f"No cached data found under {local_dir}.",
            required_action="Run --mode live once, or --mode manual_import.",
            accepted_fallback="There is no synthetic fallback; real data is required.",
        )
    logger.info("Cached source %s validated at %s (no network call performed).", name, local_dir)


# Each real data source punts live-mode discovery/download to its own dedicated entry
# point rather than performing it from this generic script -- KBA needs a source-specific
# listing_url (a specific kba.de subpage) that can't be supplied generically here;
# Switzerland is a fixed SDMX query but still has its own ensure_ch_dataset() helper/script.
_LIVE_MESSAGES: dict[str, tuple[str, str]] = {
    "kba": (
        "KBA discovery/download needs an explicit listing_url (a specific kba.de subpage).",
        "Call build_analysis_dataset(mode='live', listing_url=...) directly, e.g. from "
        "the notebook (see src/car_interest_nlp/data/kba.py for the underlying adapter).",
    ),
    "switzerland": (
        "Switzerland discovery/download has its own dedicated entry point.",
        "Call ensure_ch_dataset() directly, or run scripts/download_ch_registrations.py "
        "(see src/car_interest_nlp/data/switzerland.py for the underlying adapter).",
    ),
    "google_trends": (
        "Google Trends discovery/download has its own dedicated entry point.",
        "Call ensure_trends_dataset() directly, or run scripts/download_google_trends.py "
        "(see src/car_interest_nlp/data/google_trends.py for the underlying adapter).",
    ),
    "gdelt": (
        "GDELT discovery/download has its own dedicated, resumable entry point (real full "
        "coverage takes many runs -- see README's 'GDELT News Analysis' section).",
        "Call ensure_gdelt_dataset() directly, or run scripts/download_gdelt_news.py "
        "(see src/car_interest_nlp/data/gdelt.py for the underlying adapter).",
    ),
}


def _collect_live(name: str, source_config: dict[str, Any]) -> None:
    """Live mode: each source needs its own real entry point, so this points at it."""
    reason, required_action = _LIVE_MESSAGES.get(
        name,
        (
            f"{name!r} has no generic live-mode handler in this script.",
            f"Use {name}'s dedicated adapter/entry point directly.",
        ),
    )
    raise SourceUnavailableError(
        source=name,
        reason=reason,
        required_action=required_action,
        accepted_fallback=f"--mode manual_import, if you already have a downloaded {name} file.",
    )


def _collect_manual_import(name: str, source_config: dict[str, Any]) -> None:
    """Manual-import mode: validate a user-supplied file exists before using it."""
    local_dir = _resolve_local_dir(source_config)
    if local_dir is None or not local_dir.exists() or not any(local_dir.iterdir()):
        raise SourceUnavailableError(
            source=name,
            reason=f"No manually supplied file found under {local_dir}.",
            required_action=f"Place a manually downloaded file for {name} under its raw_directory.",
            accepted_fallback="There is no synthetic fallback; real data is required.",
        )
    logger.info("Manual import found files for %s under %s.", name, local_dir)


_DISPATCH = {
    "cached": _collect_cached,
    "live": _collect_live,
    "manual_import": _collect_manual_import,
}


def run(mode: str) -> None:
    config = load_project_config()
    sources = config["sources"]
    handler = _DISPATCH[mode]
    for name, source_config in sources.items():
        if not source_config.get("enabled", False):
            logger.info("Skipping disabled source: %s", name)
            continue
        handler(name, source_config)

    rows = build_data_source_registry(sources, PROJECT_ROOT)
    registry_path = write_data_source_registry(
        rows, ARTIFACTS_DIR / "tables" / "data_source_registry.csv"
    )
    logger.info("Wrote data source registry to %s", registry_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate/prepare configured real data sources.")
    parser.add_argument("--mode", choices=MODES, required=True)
    args = parser.parse_args()
    run(args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
