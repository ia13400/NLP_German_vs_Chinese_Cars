from __future__ import annotations

import pandas as pd

from ..data.dataset_builder import build_analysis_dataset


def get_top_brands(n: int = 5, *, frame: pd.DataFrame | None = None) -> dict[str, list[str]]:
    """Return the top `n` German and Chinese brands by real cumulative KBA registrations.

    Ranking is derived from real data (not a fixed/guessed list) so it stays correct if the
    underlying KBA series changes -- `frame` defaults to `build_analysis_dataset(mode="cached")`.
    Only the "german"/"chinese" brand groups are ranked ("other" has no single named brand
    to rank; see `configs/brands.yaml`'s `Other/Miscellaneous` entry).
    """
    data = frame if frame is not None else build_analysis_dataset(mode="cached")
    totals = (
        data.groupby(["canonical_brand", "brand_group"])["brand_registrations"].sum().reset_index()
    )
    result: dict[str, list[str]] = {}
    for group in ("german", "chinese"):
        subset = totals[totals["brand_group"] == group].sort_values(
            "brand_registrations", ascending=False
        )
        result[group] = subset["canonical_brand"].head(n).tolist()
    return result
