from __future__ import annotations

from car_interest_nlp.data.dataset_builder import build_analysis_dataset


def main() -> int:
    """Build the real KBA registration-share series and cache it for reuse."""
    frame = build_analysis_dataset()
    frame.to_csv("data/processed/processed_dataset.csv", index=False, encoding="utf-8")
    frame.to_parquet("data/processed/analysis_dataset.parquet", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
