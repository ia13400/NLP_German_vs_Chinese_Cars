from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from .style import (
    BRAND_GROUP_COLORS,
    BRAND_GROUP_LABELS_DE,
    GDELT_SOURCE_NOTE,
    GDELT_TIMELINEVOL_FETCHING_NOTE,
    KBA_SOURCE_NOTE,
    add_source_caption,
    configure_plot_style,
)


def plot_kba_share_pie_charts(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    share_column: str = "kba_registration_share",
    suptitle: str | None = None,
    source_note: str = KBA_SOURCE_NOTE,
    group_labels: dict[str, str] | None = None,
    legend_title: str = "Markengruppe",
) -> Path:
    """Plot two pie charts side by side: brand_group share in the first vs. last reporting period.

    Colors are assigned per brand_group from a fixed mapping so the same group keeps the
    same color in both pies -- matplotlib would otherwise reassign colors independently
    per subplot, which is misleading when the set of groups present differs between
    periods (e.g. "other" only appearing once SONSTIGE-mapped rows exist). `share_column`
    and `suptitle`/`source_note` let this be reused for a different series (e.g.
    Switzerland's `ch_registration_share`) without duplicating the whole function.
    `group_labels` overrides the default `BRAND_GROUP_LABELS_DE` group names (e.g. for
    Google Trends' Volkswagen-vs-BYD chapter, where each "group" is really a single named
    brand, not a many-brand aggregate like KBA/Switzerland -- labeling it "Deutsch" there
    would misleadingly imply an aggregate).
    """
    configure_plot_style()
    labels_map = {**BRAND_GROUP_LABELS_DE, **(group_labels or {})}
    periods = sorted(frame["reporting_period"].unique())
    if len(periods) < 2:
        raise ValueError("Need at least two distinct reporting periods to compare first vs. last.")
    first_period, last_period = periods[0], periods[-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.5))
    legend_handles: dict[str, object] = {}
    for ax, period in zip(axes, [first_period, last_period], strict=True):
        subset = frame[frame["reporting_period"] == period]
        shares = subset.groupby("brand_group")[share_column].sum().sort_index()
        colors = [BRAND_GROUP_COLORS.get(group, "#999999") for group in shares.index]
        # Small neighboring slices (e.g. "chinese"/"other") produce overlapping inline
        # percentage text, so their percentage moves into the exterior label instead and
        # the inline autopct is suppressed for them.
        small_slice = shares.values < 0.05
        labels = [
            f"{labels_map.get(group, group)}\n({share * 100:.1f}%)"
            if is_small
            else labels_map.get(group, group)
            for group, share, is_small in zip(shares.index, shares.values, small_slice, strict=True)
        ]
        explode = [0.07 if is_small else 0 for is_small in small_slice]

        def _autopct(pct: float, _small_cutoff: float = 5.0) -> str:
            return "" if pct < _small_cutoff else f"{pct:.1f}%"

        wedges, _texts, _autotexts = ax.pie(
            shares.values,
            labels=labels,
            autopct=_autopct,
            startangle=90,
            colors=colors,
            explode=explode,
            labeldistance=1.18,
        )
        ax.grid(False)
        for group, wedge in zip(shares.index, wedges, strict=True):
            legend_handles.setdefault(labels_map.get(group, group), wedge)
        ax.set_title(str(period), pad=28)

    fig.legend(
        legend_handles.values(),
        legend_handles.keys(),
        title=legend_title,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=len(legend_handles),
    )
    if suptitle is None:
        suptitle = (
            "Anteil der Markengruppen an den erfassten Neuzulassungen in Deutschland: "
            f"{first_period} vs. {last_period} (KBA)"
        )
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.16, 1, 0.9))
    add_source_caption(fig, note=source_note)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


_TREND_SUBPLOT_GROUPS = (("german", "Deutsche Marken"), ("chinese", "Chinesische Marken"))
# matplotlib's standard "tab10" qualitative palette, spelled out as plain hex strings rather
# than read off `plt.cm.tab10.colors` (untyped in matplotlib's stubs -- `Colormap` has no
# `.colors` attribute in general, only `ListedColormap` does at runtime).
_BRAND_LINE_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


def _format_count(value: float) -> str:
    return f"{int(round(value)):,}".replace(",", ".")


def plot_kba_registration_trend(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    share_column: str = "kba_registration_share",
    count_column: str = "brand_registrations",
    y_label: str = "Anteil an den erfassten Neuzulassungen (Deutschland)",
    suptitle: str = "Marktanteilsentwicklung in Deutschland nach Markengruppe (KBA)",
    source_note: str = KBA_SOURCE_NOTE,
    group_labels: dict[str, str] | None = None,
) -> Path:
    """Plot a share column over the years as two subplots, one for German and one for
    Chinese brands, and save as PNG.

    German and Chinese shares differ by roughly two orders of magnitude, so each group gets
    its own independently scaled y-axis -- a single shared axis would flatten the Chinese
    brands' growth into an almost invisible line near zero. Each brand_group has multiple
    brands per period, so shares/counts are summed per (period, brand_group) first --
    plotting the unaggregated per-brand rows directly would connect different brands'
    points into one chaotic line. `share_column`/`count_column`/`y_label`/`suptitle`/
    `source_note` let this be reused for a different series (e.g. Switzerland's
    `ch_registration_share`/`brand_registrations`) without duplicating the function.
    `group_labels` overrides the default subplot titles (`_TREND_SUBPLOT_GROUPS`, "Deutsche
    Marken"/"Chinesische Marken") -- e.g. for Google Trends' Volkswagen-vs-BYD chapter,
    where each "group" is really a single named brand, not a many-brand aggregate, so the
    generic group titles would misleadingly imply an aggregate of multiple brands.
    """
    configure_plot_style()
    subplot_groups = [
        (group, (group_labels or {}).get(group, default_title))
        for group, default_title in _TREND_SUBPLOT_GROUPS
    ]
    grouped = (
        frame.groupby(["reporting_period", "brand_group"])[[share_column, count_column]]
        .sum()
        .reset_index()
    )
    periods = sorted(grouped["reporting_period"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (group, title) in zip(axes, subplot_groups, strict=True):
        subset = grouped[grouped["brand_group"] == group].sort_values("reporting_period")
        ax.plot(
            subset["reporting_period"],
            subset[share_column],
            marker="o",
            color=BRAND_GROUP_COLORS.get(group),
        )
        ax.set_title(title)
        ax.set_xlabel("Jahr")
        # With many reporting periods, showing every tick unrotated crowds them into an
        # unreadable smear -- thin to ~10 ticks and rotate once there are more than a
        # handful.
        tick_step = max(1, len(periods) // 10)
        ax.set_xticks(periods[::tick_step])
        if len(periods) > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        # Extra headroom (taller now that annotations carry a second line) so they never
        # get clipped by the axes frame or crowd the x-axis tick labels.
        ax.margins(x=0.08, y=0.3)

        max_idx = subset[share_column].idxmax()
        min_idx = subset[share_column].idxmin()
        # The max is annotated above its point and the min below: since these are the
        # series' global extrema, the curve never rises above the max or dips below the
        # min, so annotating in these directions can never overlap the line itself.
        placements = {max_idx: (14, "bottom"), min_idx: (-14, "top")}
        for idx, (offset_y, vertical_alignment) in placements.items():
            row = subset.loc[idx]
            ax.annotate(
                f"{row[share_column] * 100:.1f}%\n({_format_count(row[count_column])})",
                xy=(row["reporting_period"], row[share_column]),
                xytext=(0, offset_y),
                textcoords="offset points",
                ha="center",
                va=vertical_alignment,
                fontsize=8,
                linespacing=1.3,
            )

    axes[0].set_ylabel(y_label)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    add_source_caption(fig, note=source_note)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def _plot_per_brand_group_lines(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    value_column: str,
    y_label: str,
    suptitle: str,
    source_note: str,
    percent_format: bool,
) -> Path:
    """Shared rendering for a "one line per brand, one subplot per group" chart -- used by
    both `plot_gdelt_article_count_trend` (absolute counts) and `plot_gdelt_attention_trend`
    (percentage shares), which differ only in whether the y-axis is percent-formatted.

    Brands keep their own line rather than being summed into one group total, since brand
    identity (e.g. BYD specifically, not just "chinesische Marken" in aggregate) is the point
    of this GDELT chapter -- mirrors Google Trends' Volkswagen-vs-BYD chapter, not KBA's
    group-aggregate view. `frame` must already be aggregated to one row per
    (reporting_period, brand_group, canonical_brand) -- or be completely empty (e.g. zero
    `timelinevol`/`artlist` chunks cached at all yet), in which case both subplots render the
    same "no data yet" placeholder rather than erroring on a missing column.
    """
    configure_plot_style()
    has_data = not frame.empty and "reporting_period" in frame.columns
    # Plot on a numeric (int year) x-axis rather than the raw string reporting_period --
    # matplotlib assigns string-category positions in first-plotted order, so if one brand's
    # data doesn't start in the earliest year (real, partial GDELT coverage makes this
    # common), a later brand introducing that earlier year appends it out of order at the
    # end of the axis instead of at the start. Confirmed directly: with BYD (plotted first,
    # alphabetically) missing 2021 data, the x-axis rendered as 2022-2023-2024-2025-2021.
    periods = sorted(frame["reporting_period"].astype(int).unique()) if has_data else []

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, (group, title) in zip(axes, _TREND_SUBPLOT_GROUPS, strict=True):
        subset = frame[frame["brand_group"] == group] if has_data else frame
        brands = sorted(subset["canonical_brand"].unique()) if has_data else []
        for i, brand in enumerate(brands):
            brand_rows = subset[subset["canonical_brand"] == brand].sort_values("reporting_period")
            ax.plot(
                brand_rows["reporting_period"].astype(int),
                brand_rows[value_column],
                marker="o",
                label=brand,
                color=_BRAND_LINE_COLORS[i % len(_BRAND_LINE_COLORS)],
            )
        ax.set_title(title)
        ax.set_xlabel("Jahr")
        ax.margins(y=0.15)
        if percent_format:
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
        if brands:
            # Only set explicit xticks/xlim when something was actually plotted -- calling
            # set_xticks() on an axis that never received a plot() call raises a matplotlib
            # unit-conversion error, confirmed directly when the Chinese-brand subplot had
            # zero cached chunks yet. Both subplots share the same explicit xlim (spanning
            # the full period range across *both* groups, not just this one) rather than
            # relying on autoscale -- confirmed directly that a subplot with real data for
            # only a single year (Polestar's one cached chunk) otherwise autoscales its axis
            # around just that one point, squeezing all the year tick labels into an
            # unreadable overlapping cluster.
            ax.set_xticks(periods)
            ax.set_xlim(periods[0] - 0.5, periods[-1] + 0.5)
            ax.legend(fontsize=8, loc="upper left")
        else:
            ax.text(
                0.5,
                0.5,
                "Noch keine echten Daten gecacht",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
                color="#777777",
            )

    axes[0].set_ylabel(y_label)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    add_source_caption(fig, note=source_note)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_gdelt_article_count_trend(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    count_column: str = "article_count",
    start_year: int = 2021,
    end_year: int = 2025,
    suptitle: str = "Medienberichterstattung je Marke: reale Artikelanzahl 2021-2025 (GDELT)",
    source_note: str = GDELT_SOURCE_NOTE,
) -> Path:
    """Plot each brand's real yearly GDELT article count as its own line, one subplot per
    group (German/Chinese), and save as PNG.

    Plots absolute counts, not a percentage share -- unlike KBA/Switzerland, GDELT's top-5+5
    brand selection has no natural "Other" denominator to compute a share against, since
    every brand queried was chosen deliberately, not parsed from a raw report (see
    `gdelt.build_gdelt_annual_series`'s docstring).

    Rows outside `[start_year, end_year]` are dropped before plotting: real `seendate` values
    occasionally fall just past a fetched month-window's edge (confirmed directly: a handful
    of cached chunks near the end of 2025 contained articles GDELT itself dated into January
    2026), which would otherwise show up as a stray, misleadingly tiny partial year on a
    chart titled "2021-2025" -- the same reasoning Google Trends' `end_year` filtering uses
    for its own incomplete trailing year.

    Real GDELT `artlist` coverage is frequently partial (rate-limited, resumable fetch -- see
    README's "GDELT News Analysis") and `article_count` is capped at 250/month for
    high-volume brands, so a given brand's line reflects fetched-so-far coverage, not
    necessarily true total media volume; brands with no cached chunks yet simply have no
    line, rather than a fabricated zero.
    """
    scoped = frame[
        (frame["reporting_period"].astype(int) >= start_year)
        & (frame["reporting_period"].astype(int) <= end_year)
    ]
    grouped = (
        scoped.groupby(["reporting_period", "brand_group", "canonical_brand"])[count_column]
        .sum()
        .reset_index()
    )
    return _plot_per_brand_group_lines(
        grouped,
        output_path,
        value_column=count_column,
        y_label="Anzahl real erfasster Artikel",
        suptitle=suptitle,
        source_note=source_note,
        percent_format=False,
    )


def _plot_overlaid_series(
    output_path: str | Path,
    series: Sequence[tuple[str, pd.DataFrame, str | None]],
    *,
    value_column: str,
    y_label: str,
    suptitle: str,
    source_note: str,
    note_box: str | None = None,
) -> Path:
    """Shared rendering for a "few named lines overlaid on one shared axes" chart -- used by
    both `plot_gdelt_attention_trend` (German-vs-Chinese group totals) and
    `plot_gdelt_brand_attention_trend` (specific named brands).

    `series` is a list of (label, subset_frame, color) tuples, each subset_frame already
    filtered to just that line's rows (with `reporting_period`/`value_column` columns) --
    an empty subset_frame is skipped and reported as "no data yet" rather than silently
    omitted, matching every other GDELT chart's honest-partial-coverage handling.

    `note_box`, if given, is rendered as a bordered text box in the lower-right corner of the
    axes (e.g. listing exactly which real brands were summed into each group line) -- kept
    separate from the "no data yet" placeholder text, which anchors lower-left instead, so the
    two never overlap.
    """
    configure_plot_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted_labels: list[str] = []
    all_periods: set[int] = set()

    for label, subset, color in series:
        if subset.empty:
            continue
        subset = subset.sort_values("reporting_period")
        periods_int = subset["reporting_period"].astype(int)
        all_periods.update(periods_int.tolist())
        plotted_labels.append(label)
        ax.plot(periods_int, subset[value_column], marker="o", label=label, color=color)

    if plotted_labels:
        periods = sorted(all_periods)
        ax.set_xticks(periods)
        ax.set_xlim(periods[0] - 0.5, periods[-1] + 0.5)
        ax.legend(fontsize=9)

    missing_labels = [label for label, subset, _color in series if subset.empty]
    if missing_labels:
        note = (
            "Noch keine echten Daten gecacht"
            if not plotted_labels
            else f"Noch keine echten Daten: {', '.join(missing_labels)}"
        )
        anchor = (0.5, 0.5) if not plotted_labels else (0.02, 0.02)
        ax.text(
            *anchor,
            note,
            ha="center" if not plotted_labels else "left",
            va="center" if not plotted_labels else "bottom",
            transform=ax.transAxes,
            fontsize=9 if not plotted_labels else 8,
            color="#777777",
        )

    if note_box:
        ax.text(
            0.98,
            0.02,
            note_box,
            ha="right",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7.5,
            linespacing=1.4,
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#999999"},
        )

    ax.set_xlabel("Jahr")
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.margins(y=0.15)
    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0.02, 0.05, 1, 0.95))
    add_source_caption(fig, note=source_note)

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


def plot_gdelt_attention_trend(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    value_column: str = "mean_daily_share",
    suptitle: str = "Medienaufmerksamkeit: Top 5 deutsche vs. Top 5 chinesische Marken 2021-2025 (GDELT)",
    source_note: str = GDELT_SOURCE_NOTE,
) -> Path:
    """Plot the combined German-vs-Chinese GDELT media-attention trend as two overlaid lines
    on one shared axes, and save as PNG.

    `frame` is `media_attention.summarize_attention_by_year()`'s per-brand output --
    aggregated here to one line per *group*, not per brand, per explicit preference: this
    chapter shows only the German-vs-Chinese comparison, unlike `plot_gdelt_article_count_trend`
    (which keeps individual brand lines). Each group's brands are summed per year (not
    averaged): summing each real per-brand yearly-mean daily share is the natural "combined
    group share" statistic, valid under linearity of expectation -- though it is a real
    methodological approximation, not an exact deduplicated total, since a single real
    article mentioning two brands from the same group (e.g. both BMW and Mercedes-Benz)
    contributes to both brands' individual GDELT query results and is therefore counted
    twice in the group sum. Both lines share one axes (rather than KBA/Switzerland's
    separate-y-axis-per-group subplots) since German and Chinese media-attention shares are
    within the same order of magnitude here (real cached data: roughly a 5-10x gap, not the
    ~50-100x gap KBA/Switzerland's market shares have), so a direct overlay stays readable
    and gives the clearest side-by-side comparison. Since each line is a sum over 5 brands,
    the title says so explicitly ("Top 5 ... vs. Top 5 ...") and a note box on the chart spells
    out exactly which real brands (`get_top_brands()`'s actual output, not a fixed guess) were
    summed into each line -- so the chart is self-explanatory without the surrounding notebook
    markdown.
    """
    has_data = not frame.empty and "reporting_period" in frame.columns
    grouped = (
        frame.groupby(["reporting_period", "brand_group"])[value_column].sum().reset_index()
        if has_data
        else frame
    )
    empty = grouped.iloc[0:0]
    series = [
        (
            label,
            grouped[grouped["brand_group"] == group] if has_data else empty,
            BRAND_GROUP_COLORS.get(group),
        )
        for group, label in _TREND_SUBPLOT_GROUPS
    ]

    note_box = None
    if has_data:
        lines = []
        for group, label in _TREND_SUBPLOT_GROUPS:
            brands = sorted(frame.loc[frame["brand_group"] == group, "canonical_brand"].unique())
            if brands:
                lines.append(f"{label}: {', '.join(brands)}")
        note_box = "\n".join(lines) or None

    return _plot_overlaid_series(
        output_path,
        series,
        value_column=value_column,
        y_label="Mittlerer Tagesanteil an globaler Berichterstattung\n(Gruppensumme)",
        suptitle=suptitle,
        source_note=source_note,
        note_box=note_box,
    )


def plot_gdelt_brand_attention_trend(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    brands: Sequence[str] = ("Volkswagen", "BYD"),
    value_column: str = "mean_daily_share",
    suptitle: str = "Medienaufmerksamkeit: Volkswagen vs. BYD 2021-2025 (GDELT)",
    source_note: str = GDELT_SOURCE_NOTE,
    fetching_note: str = GDELT_TIMELINEVOL_FETCHING_NOTE,
) -> Path:
    """Plot specific named brands' (default Volkswagen vs. BYD) real yearly-mean GDELT
    `timelinevol` coverage share as overlaid lines on one shared axes, with a caption under
    the chart explaining exactly how the data was fetched, and save as PNG.

    `frame` is `media_attention.summarize_attention_by_year()`'s per-brand output, filtered
    directly to `brands` -- no group aggregation, unlike `plot_gdelt_attention_trend` --
    mirroring the Google Trends chapter's Volkswagen-vs-BYD flagship-brand comparison in the
    other notebook, so this GDELT chapter offers the same two-brand view for media coverage.
    Each brand's line uses its own `brand_group`'s color (`BRAND_GROUP_COLORS`) for visual
    consistency with every other German-vs-Chinese chart in this project. The caption
    combines the usual source attribution with `fetching_note`, which spells out the real
    fetch methodology (one request per brand for the whole range, exact quoted query phrase,
    yearly fallback) directly on the figure, not just in the notebook's surrounding markdown.
    """
    has_data = not frame.empty and "canonical_brand" in frame.columns
    empty = frame.iloc[0:0]
    series: list[tuple[str, pd.DataFrame, str | None]] = []
    for brand in brands:
        subset = frame[frame["canonical_brand"] == brand] if has_data else empty
        color = BRAND_GROUP_COLORS.get(subset["brand_group"].iloc[0]) if not subset.empty else None
        series.append((brand, subset, color))

    combined_note = f"{source_note}\n{fetching_note}"
    return _plot_overlaid_series(
        output_path,
        series,
        value_column=value_column,
        y_label="Mittlerer Tagesanteil an globaler Berichterstattung",
        suptitle=suptitle,
        source_note=combined_note,
    )
