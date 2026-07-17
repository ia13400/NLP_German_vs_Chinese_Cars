from __future__ import annotations

from typing import Any

import pandas as pd
from spacy.language import Language

from ..config import load_project_config
from ..preprocessing.brand_matching import build_brand_alias_map, resolve_brands
from ..progress import TimeBudget
from .features import build_entity_count_matrix
from .scraping import CARWALE_BRAND_SLUG_TO_CANONICAL


def add_brand_group_columns(
    frame: pd.DataFrame, *, brand_column: str = "brand_slug"
) -> pd.DataFrame:
    """Attach `canonical_brand`/`origin_group` columns to a reviews frame.

    Reuses the same `configs/brands.yaml` alias resolution the KBA/Switzerland chapters use
    (`preprocessing.brand_matching`) instead of a separate hardcoded brand-to-country map --
    every CarWale brand slug used in this project already has a matching canonical entry
    there (see `CARWALE_BRAND_SLUG_TO_CANONICAL`).
    """
    config = load_project_config()
    alias_map = build_brand_alias_map(config["brands"])
    mapped = frame.copy()
    mapped["brand"] = mapped[brand_column].map(CARWALE_BRAND_SLUG_TO_CANONICAL)
    resolved, unresolved = resolve_brands(mapped, alias_map, brand_column="brand")
    if not unresolved.empty:
        raise ValueError(
            f"{len(unresolved)} review row(s) had a brand slug with no canonical mapping in "
            "CARWALE_BRAND_SLUG_TO_CANONICAL/configs/brands.yaml: "
            f"{sorted(unresolved[brand_column].unique())}"
        )
    return resolved


def predict_aspects(
    nlp: Language,
    model: Any,
    entity_labels: list[str],
    aspect_classes: list[str],
    frame: pd.DataFrame,
    *,
    time_budget: TimeBudget | None = None,
) -> pd.DataFrame:
    """Apply the trained NER pipeline + aspect classifier to real (unlabeled) reviews.

    Adds a `predicted_aspect` column built from the same entity-count feature representation
    used during training (`features.build_entity_count_matrix`), so a classifier trained on
    the labeled corpus and a classifier applied to new reviews always see identical features.
    """
    records = frame.to_dict(orient="records")
    feature_matrix = build_entity_count_matrix(nlp, records, entity_labels, time_budget=time_budget)
    predicted_ids = model.predict(feature_matrix)
    result = frame.copy()
    result["predicted_aspect"] = [aspect_classes[int(class_id)] for class_id in predicted_ids]
    return result
