from __future__ import annotations

from car_interest_nlp.config import PROJECT_ROOT, load_project_config
from car_interest_nlp.data.provenance import build_data_source_registry, write_data_source_registry
from car_interest_nlp.logging_utils import configure_logging
from car_interest_nlp.paths import ARTIFACTS_DIR


def main() -> int:
    """Rebuild artifacts/tables/data_source_registry.csv from configs/sources.yaml and disk state."""
    logger = configure_logging()
    config = load_project_config()
    rows = build_data_source_registry(config["sources"], PROJECT_ROOT)
    path = write_data_source_registry(rows, ARTIFACTS_DIR / "tables" / "data_source_registry.csv")
    logger.info("Wrote data source registry to %s (%s sources)", path, len(rows))
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
