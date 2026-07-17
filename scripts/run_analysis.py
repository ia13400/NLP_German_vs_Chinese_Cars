from __future__ import annotations

from car_interest_nlp.analysis.descriptive import create_dataset_summary
from car_interest_nlp.data.dataset_builder import build_analysis_dataset


def main() -> int:
    """Run the KBA registration market-share analysis and print a summary."""
    frame = build_analysis_dataset()
    summary = create_dataset_summary(frame)
    print(summary)
    print(frame.sort_values("reporting_period").tail(10))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
