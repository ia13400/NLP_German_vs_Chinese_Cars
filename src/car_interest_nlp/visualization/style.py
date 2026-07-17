from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

KBA_SOURCE_NOTE = (
    "Quelle: Kraftfahrt-Bundesamt (KBA) - Offizielle PKW-Neuzulassungen pro Jahr in "
    "Deutschland (www.kba.de)"
)

CH_SOURCE_NOTE = (
    "Quelle: Bundesamt für Statistik (BFS) / ASTRA - Neuzulassungen von Personenwagen pro Jahr "
    "in der Schweiz (stats.swiss)"
)

GOOGLE_TRENDS_SOURCE_NOTE = (
    "Quelle: Google Trends - relativer Suchinteresse-Index (trends.google.com); misst "
    "Sucheinteresse, keine echte Marktkennzahl"
)

GDELT_SOURCE_NOTE = (
    "Quelle: GDELT Project DOC 2.0 API (api.gdeltproject.org) - reale, englischsprachige "
    "Artikel-Metadaten; misst Medienberichterstattung, keine echte Marktkennzahl"
)

# Explains the real timelinevol fetching methodology directly on the chart, not just the
# generic source attribution above -- confirmed directly during this project's development:
# one request per brand for the whole 2021-2025 range (not chunked monthly/yearly), query is
# the exact quoted phrase "<brand> car" with no additional context word (a "german"/"chinese"
# context word was tried and removed after live testing showed it cut real matches to ~13% of
# days instead of ~54%), and a five-yearly-request fallback if the single request fails.
GDELT_TIMELINEVOL_FETCHING_NOTE = (
    "Datenerhebung: GDELT-Modus 'timelinevol', eine Anfrage je Marke über den gesamten "
    'Zeitraum 2021-2025 (Suchbegriff: exakte Phrase "<Marke> car", ohne Kontextwort), mit '
    "automatischem Fallback auf fünf einzelne Jahres-Anfragen bei Fehlschlag. Wert = realer, "
    "über das jeweilige Jahr gemittelter Tagesanteil an der weltweiten GDELT-Berichterstattung."
)

BRAND_GROUP_COLORS = {
    "german": "#B8860B",
    "chinese": "#B22222",
    "other": "#7F7F7F",
}

BRAND_GROUP_LABELS_DE = {
    "german": "Deutsch",
    "chinese": "Chinesisch",
    "other": "Sonstige",
}


def configure_plot_style() -> None:
    """Configure a consistent, print-friendly academic style for all project figures."""
    mpl.rcParams["figure.dpi"] = 300
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.size"] = 10
    mpl.rcParams["axes.titlesize"] = 12
    mpl.rcParams["axes.labelsize"] = 10
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["grid.linestyle"] = "--"
    mpl.rcParams["legend.frameon"] = False


def add_source_caption(fig: plt.Figure, note: str = KBA_SOURCE_NOTE) -> None:
    """Add a small italic source-attribution caption at the bottom of a figure.

    `wrap=True` lets matplotlib break a long note onto multiple lines instead of running
    off the edge of a narrow figure (e.g. a bar chart with very few bars, where figure
    width scales down with the number of bars).
    """
    fig.text(
        0.5,
        0.01,
        note,
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="#444444",
        wrap=True,
    )
