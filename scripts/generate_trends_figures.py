from __future__ import annotations

from pathlib import Path

from car_interest_nlp.config import load_project_config
from car_interest_nlp.data.trends_dataset_builder import build_trends_analysis_dataset
from car_interest_nlp.visualization.report_plots import create_report_placeholder
from car_interest_nlp.visualization.style import BRAND_GROUP_COLORS, GOOGLE_TRENDS_SOURCE_NOTE
from car_interest_nlp.visualization.trends import (
    plot_kba_registration_trend,
    plot_kba_share_pie_charts,
)


def main() -> int:
    """Generate the Google Trends search-interest trend, pie-chart, and summary figures.

    A structurally different measurement than the KBA/Switzerland figures (search
    interest, not a market outcome), reusing the same generic plotting functions with
    Trends-specific columns/labels -- `google_trends_interest_share` is a share of
    *tracked* brand interest, not of all searches, and must not be read the same way as
    the registration sources' market-share percentages.
    """
    frame = build_trends_analysis_dataset()

    Path("artifacts/tables").mkdir(parents=True, exist_ok=True)
    frame.to_csv("artifacts/tables/google_trends_brand_interest.csv", index=False, encoding="utf-8")

    # Read from config rather than hardcoding "Deutschland" -- geo is configurable
    # (configs/sources.yaml's google_trends.geo) and the checked-in dataset is currently
    # Worldwide, not Germany-specific (see README's "Google Trends" section).
    geo = load_project_config()["sources"].get("google_trends", {}).get("geo", "")
    geo_label = geo if geo else "weltweit"

    # This chapter tracks exactly one brand per group (Volkswagen, BYD by default), not a
    # many-brand aggregate like KBA/Switzerland -- the generic "Deutsche Marken"/"Deutsch"
    # group labels the shared plotting functions default to would misleadingly imply an
    # aggregate, so the brand names are used as the actual chart labels instead.
    group_labels = dict(zip(frame["brand_group"], frame["canonical_brand"], strict=False))

    plot_kba_registration_trend(
        frame,
        "artifacts/figures/trends/google_trends_interest_trend.png",
        share_column="google_trends_interest_share",
        count_column="search_interest_index",
        y_label=f"Anteil am Suchinteresse ({geo_label})",
        suptitle="Entwicklung des Suchinteresses: Volkswagen vs. BYD (Google Trends)",
        source_note=GOOGLE_TRENDS_SOURCE_NOTE,
        group_labels=group_labels,
    )

    first_period = frame["reporting_period"].min()
    last_period = frame["reporting_period"].max()
    plot_kba_share_pie_charts(
        frame,
        "artifacts/figures/trends/google_trends_share_pie_first_last.png",
        share_column="google_trends_interest_share",
        suptitle=(
            "Anteil am erfassten Suchinteresse, Volkswagen vs. BYD (Google Trends): "
            f"{first_period} vs. {last_period}"
        ),
        source_note=GOOGLE_TRENDS_SOURCE_NOTE,
        group_labels=group_labels,
        legend_title="Marke",
    )

    latest = frame[frame["reporting_period"] == last_period]
    interest_share = dict(
        zip(latest["canonical_brand"], latest["google_trends_interest_share"], strict=False)
    )
    brand_to_group = dict(zip(latest["canonical_brand"], latest["brand_group"], strict=False))
    bar_colors = [
        BRAND_GROUP_COLORS.get(brand_to_group.get(brand), "#219ebc") for brand in interest_share
    ]
    group_totals = latest.groupby("brand_group")["search_interest_index"].sum()
    legend_notes = [
        f"{group_labels.get('german', 'Deutsche Marke')} (Such-Index gesamt): "
        f"{group_totals.get('german', 0):.0f}",
        f"{group_labels.get('chinese', 'Chinesische Marke')} (Such-Index gesamt): "
        f"{group_totals.get('chinese', 0):.0f}",
    ]
    create_report_placeholder(
        "artifacts/figures/frequency/google_trends_share_latest.png",
        metrics=interest_share,
        title=f"Anteil am erfassten Suchinteresse je Marke ({last_period})",
        bar_colors=bar_colors,
        legend_notes=legend_notes,
        source_note=GOOGLE_TRENDS_SOURCE_NOTE,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
