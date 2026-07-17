from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from .style import BRAND_GROUP_COLORS, BRAND_GROUP_LABELS_DE, configure_plot_style

REVIEWS_SOURCE_NOTE = (
    "Quelle: CarWale (www.carwale.com) - reale Fahrzeugbewertungen; eigenes Aspekt-NER-Modell "
    "und Klassifikator, trainiert auf manuell annotierten Bewertungen"
)


def _finish_figure(figure: Figure, output_path: str | Path) -> Path:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=300)
    plt.close(figure)
    return target


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    *,
    title: str = "Konfusionsmatrix - Aspektklassifikation",
) -> Path:
    """Plot the out-of-fold confusion matrix for the aspect classifier."""
    configure_plot_style()
    figure, axis = plt.subplots(
        figsize=(max(6.0, 0.9 * len(class_names)), max(5.0, 0.7 * len(class_names)))
    )
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=class_names)
    display.plot(ax=axis, xticks_rotation=45, values_format="d", colorbar=False)
    axis.set_title(title)
    return _finish_figure(figure, output_path)


def plot_predicted_aspect_distribution(
    counts: dict[str, int],
    output_path: str | Path,
    *,
    kind: str = "pie",
    title: str = "Verteilung der vorhergesagten Aspekte",
) -> Path:
    """Plot the distribution of predicted aspects across newly scraped reviews, as a pie
    (`kind="pie"`) or bar (`kind="bar"`) chart.
    """
    configure_plot_style()
    labels = list(counts.keys())
    values = list(counts.values())
    if kind == "pie":
        figure, axis = plt.subplots(figsize=(8, 8))
        axis.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
    elif kind == "bar":
        figure, axis = plt.subplots(figsize=(max(8.0, 0.9 * len(labels)), 6))
        axis.bar(labels, values, color="#219ebc")
        axis.set_ylabel("Anzahl Bewertungen")
        plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    else:
        raise ValueError(f"Unknown kind: {kind!r}")
    axis.set_title(title)
    return _finish_figure(figure, output_path)


def plot_average_rating_by_aspect_and_group(
    frame: pd.DataFrame,
    output_path: str | Path,
    *,
    aspect_column: str = "predicted_aspect",
    group_column: str = "origin_group",
    rating_column: str = "Rating",
    title: str = "Durchschnittliche Bewertung je Aspekt: Deutsch vs. Chinesisch",
) -> Path:
    """Grouped bar chart of average star rating per predicted aspect, split by brand group
    (German vs. Chinese), reusing the same `BRAND_GROUP_COLORS`/`BRAND_GROUP_LABELS_DE`
    mapping the KBA/Switzerland/Trends/GDELT chapters use for the same two groups.
    """
    configure_plot_style()
    pivot = frame.groupby([aspect_column, group_column])[rating_column].mean().unstack(group_column)
    pivot = pivot.rename(columns=BRAND_GROUP_LABELS_DE)
    colors = [
        BRAND_GROUP_COLORS.get(group, "#7F7F7F")
        for group in frame[group_column].unique()
        if group in BRAND_GROUP_LABELS_DE
    ]
    figure, axis = plt.subplots(figsize=(max(8.0, 1.1 * len(pivot)), 6))
    pivot.plot(kind="bar", ax=axis, color=colors or None)
    axis.set_title(title)
    axis.set_xlabel("Aspekt")
    axis.set_ylabel("Durchschnittliche Bewertung")
    axis.set_ylim(0, 6)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    axis.legend(title="Markengruppe")
    return _finish_figure(figure, output_path)
