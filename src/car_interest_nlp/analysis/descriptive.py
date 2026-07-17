from __future__ import annotations

import pandas as pd


def create_dataset_summary(
    frame: pd.DataFrame, *, value_column: str = "brand_registrations"
) -> dict[str, object]:
    """Create a compact descriptive summary of a brand/period share series.

    `value_column` is the per-row count column to total by brand (`brand_registrations`
    by default, used by both KBA and Switzerland).
    """
    if frame.empty:
        return {"rows": 0, "columns": list(frame.columns)}
    return {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "reporting_period_start": str(frame["reporting_period"].min()),
        "reporting_period_end": str(frame["reporting_period"].max()),
        "brand_group_counts": frame["brand_group"].value_counts().to_dict(),
        "total_by_brand": frame.groupby("canonical_brand")[value_column]
        .sum()
        .sort_values(ascending=False)
        .to_dict(),
    }
