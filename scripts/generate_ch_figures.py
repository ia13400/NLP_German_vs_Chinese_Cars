from __future__ import annotations

from pathlib import Path

from car_interest_nlp.data.switzerland_dataset_builder import build_ch_analysis_dataset
from car_interest_nlp.visualization.report_plots import create_report_placeholder
from car_interest_nlp.visualization.style import BRAND_GROUP_COLORS, CH_SOURCE_NOTE
from car_interest_nlp.visualization.trends import (
    plot_kba_registration_trend,
    plot_kba_share_pie_charts,
)


def _format_german_thousands(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def main() -> int:
    """Generate the Swiss new-registration-share trend, pie-chart, and summary figures.

    Same measurement type as the KBA figures (national new-registration flow, annual), just
    for Switzerland instead of Germany -- meaningfully comparable in kind, but absolute
    registration counts should not be compared directly given the very different market
    sizes; only market-share percentages are comparable.
    """
    frame = build_ch_analysis_dataset()

    Path("artifacts/tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv("artifacts/tables/ch_registration_share.csv", index=False, encoding="utf-8")

    plot_kba_registration_trend(
        frame,
        "artifacts/figures/trends/ch_registration_trend.png",
        share_column="ch_registration_share",
        y_label="Anteil an den erfassten Neuzulassungen (Schweiz)",
        suptitle="Marktanteilsentwicklung in der Schweiz nach Markengruppe",
        source_note=CH_SOURCE_NOTE,
    )

    first_period = frame["reporting_period"].min()
    last_period = frame["reporting_period"].max()
    plot_kba_share_pie_charts(
        frame,
        "artifacts/figures/trends/ch_share_pie_first_last.png",
        share_column="ch_registration_share",
        suptitle=(
            "Anteil der Markengruppen an den erfassten Neuzulassungen in der Schweiz: "
            f"{first_period} vs. {last_period}"
        ),
        source_note=CH_SOURCE_NOTE,
    )

    latest_period = last_period
    latest = frame[frame["reporting_period"] == latest_period]
    market_share = dict(
        zip(latest["canonical_brand"], latest["ch_registration_share"], strict=False)
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
        "artifacts/figures/frequency/ch_market_share_latest.png",
        metrics=market_share,
        title=f"Marktanteile der Marken in der Schweiz ({latest_period})",
        bar_colors=bar_colors,
        legend_notes=legend_notes,
        source_note=CH_SOURCE_NOTE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
