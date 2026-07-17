from __future__ import annotations

from pathlib import Path

from car_interest_nlp.data.dataset_builder import build_analysis_dataset
from car_interest_nlp.visualization.report_plots import create_report_placeholder
from car_interest_nlp.visualization.style import BRAND_GROUP_COLORS
from car_interest_nlp.visualization.trends import (
    plot_kba_registration_trend,
    plot_kba_share_pie_charts,
)


def _format_german_thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def main() -> int:
    """Generate the KBA registration-share trend, pie-chart, and summary figures/tables."""
    frame = build_analysis_dataset()

    Path("artifacts/tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv("artifacts/tables/kba_registration_share.csv", index=False, encoding="utf-8")

    plot_kba_registration_trend(frame, "artifacts/figures/trends/kba_registration_trend.png")
    plot_kba_share_pie_charts(frame, "artifacts/figures/trends/kba_share_pie_first_last.png")

    latest_period = frame["reporting_period"].max()
    latest = frame[frame["reporting_period"] == latest_period]
    market_share = dict(
        zip(latest["canonical_brand"], latest["kba_registration_share"], strict=False)
    )
    brand_to_group = dict(zip(latest["canonical_brand"], latest["brand_group"], strict=False))
    bar_colors = [
        BRAND_GROUP_COLORS.get(brand_to_group.get(brand), "#219ebc") for brand in market_share
    ]
    group_totals = latest.groupby("brand_group")["brand_registrations"].sum()
    legend_notes = [
        f"Deutsche Marken gesamt: {_format_german_thousands(group_totals.get('german', 0))}",
        f"Chinesische Marken gesamt: {_format_german_thousands(group_totals.get('chinese', 0))}",
    ]
    create_report_placeholder(
        "artifacts/figures/frequency/kba_market_share_latest.png",
        metrics=market_share,
        title=f"Marktanteile der Marken in Deutschland ({latest_period})",
        bar_colors=bar_colors,
        legend_notes=legend_notes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
