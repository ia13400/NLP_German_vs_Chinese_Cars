from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .style import KBA_SOURCE_NOTE, add_source_caption, configure_plot_style


def create_report_placeholder(
    output_path: str | Path,
    metrics: dict[str, float] | None = None,
    *,
    title: str = "Zusammenfassung der Analyse",
    bar_colors: list[str] | None = None,
    legend_notes: list[str] | None = None,
    source_note: str = KBA_SOURCE_NOTE,
) -> Path:
    """Create a compact KPI/summary bar chart PNG.

    Figure width scales with the number of bars, and labels are rotated once there are
    more than a handful -- without this, a chart with many categories (e.g. one bar per
    brand) renders with illegible, overlapping x-axis labels.

    `bar_colors` optionally colors each bar individually (e.g. by brand group), in the
    same order as `metrics`. `legend_notes` are optional extra text lines (e.g. group
    totals not directly visible from the bars) rendered as a text-only legend box via
    invisible proxy handles. `source_note` overrides the default KBA attribution caption
    (e.g. for a different data source's summary chart).
    """
    configure_plot_style()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = metrics or {"Dokumente": 10, "Markengruppen": 2}
    width = max(6.0, 0.4 * len(values))
    figure, axis = plt.subplots(figsize=(width, 4))
    axis.bar(values.keys(), values.values(), color=bar_colors or "#219ebc")
    axis.set_title(title)
    axis.set_ylabel("Wert")
    if len(values) > 6:
        plt.setp(axis.get_xticklabels(), rotation=60, ha="right")
    if legend_notes:
        handles = [Patch(facecolor="none", edgecolor="none", label=note) for note in legend_notes]
        axis.legend(
            handles=handles,
            loc="upper left",
            frameon=True,
            handlelength=0,
            handletextpad=0,
        )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    add_source_caption(figure, note=source_note)
    figure.savefig(target, dpi=300)
    plt.close(figure)
    return target
