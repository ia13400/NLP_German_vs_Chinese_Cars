from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .style import BRAND_GROUP_COLORS, GDELT_SOURCE_NOTE

# Plotly (not matplotlib, unlike every other chart in this project) is used only here, since a
# real geographic choropleth needs actual country boundary geometry that matplotlib does not
# ship -- plotly bundles a world atlas and resolves real country names directly, avoiding a
# heavier geopandas/GDAL dependency. Static PNG export (matching every other chart's artifact
# format) goes through `kaleido`, confirmed directly to work in this project's environment.
#
# `locationmode="country names"` matches GDELT's real `sourcecountry` field values directly
# (e.g. "Germany", "United States", "Pakistan", confirmed against real cached data) with no
# manual ISO-code mapping table needed -- plotly currently warns this lookup's underlying
# library is changing in a future version; if country resolution ever silently degrades after
# a plotly upgrade, switching to `locationmode="ISO-3"` with an explicit name->ISO3 mapping is
# the fallback.

_NEUTRAL_MIDPOINT_COLOR = "#F2F2F2"


def plot_media_dominance_choropleth(
    frame: pd.DataFrame,
    year: int,
    output_path: str | Path,
    *,
    brand_a: str = "Volkswagen",
    brand_b: str = "BYD",
    color_a: str = BRAND_GROUP_COLORS["german"],
    color_b: str = BRAND_GROUP_COLORS["chinese"],
    suptitle: str | None = None,
    source_note: str = GDELT_SOURCE_NOTE,
) -> Path:
    """Plot a world choropleth of real per-country GDELT media dominance for one year between
    two named brands, and save as a static PNG.

    `frame` is `analysis.media_geography.build_country_dominance()`'s output, already scoped
    to both brands -- this just selects `year` and colors each country by its real
    `dominance_score` (see that function's docstring) on a diverging scale anchored at `frame`'s
    own brand colors (`color_a`/`color_b`, defaulting to this project's German/Chinese brand-
    group colors so this map stays visually consistent with every other German-vs-Chinese
    chart here), with a neutral grey midpoint at an even split. Countries with no real cached
    coverage for this year are simply absent from the map (plotly leaves them uncolored),
    rather than shown as a fabricated zero -- an empty `frame`/year renders an honest "no data
    yet" placeholder instead of an empty map, matching every other GDELT chart's handling of
    partial real coverage.
    """
    suptitle = (
        suptitle or f"Mediendominanz nach Ursprungsland {year}: {brand_a} vs. {brand_b} (GDELT)"
    )
    subset = frame[frame["year"] == year] if not frame.empty else frame

    if subset.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Noch keine echten Daten gecacht",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 16, "color": "#777777"},
        )
        fig.update_layout(
            title=suptitle, xaxis={"visible": False}, yaxis={"visible": False}, height=500
        )
    else:
        fig = px.choropleth(
            subset,
            locations="sourcecountry",
            locationmode="country names",
            color="dominance_score",
            range_color=(-1, 1),
            color_continuous_scale=[
                [0.0, color_b],
                [0.5, _NEUTRAL_MIDPOINT_COLOR],
                [1.0, color_a],
            ],
            title=suptitle,
        )
        fig.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            bgcolor="rgba(0,0,0,0)",
        )
        fig.update_layout(
            coloraxis_colorbar={
                "title": "Dominanz",
                "tickvals": [-1, 0, 1],
                "ticktext": [brand_b, "Ausgeglichen", brand_a],
            }
        )

    fig.add_annotation(
        text=source_note,
        x=0.5,
        y=-0.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 10, "color": "#444444"},
        xanchor="center",
    )
    fig.update_layout(
        font={"family": "Georgia, Times New Roman, serif"},
        paper_bgcolor="white",
        margin={"t": 70, "b": 60, "l": 10, "r": 10},
        width=1100,
        height=650,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(target, scale=2)
    return target
