from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from spacy.language import Language

from ..progress import TimeBudget, iter_with_progress


def _review_document_text(title: object, comment: object) -> str:
    """Combine title and comment into one document, matching the original notebook's input
    shape for both NER training/inference and feature extraction.
    """
    return f"TITLE:\n{title}\n\nTEXT:\n{comment}"


def extract_review_entities(nlp: Language, title: object, comment: object) -> pd.DataFrame:
    """Run the trained review NER pipeline over one review, returning its entities as rows."""
    doc = nlp(_review_document_text(title, comment))
    return pd.DataFrame(
        [(ent.text, ent.label_) for ent in doc.ents], columns=["entity_text", "label"]
    )


def build_entity_count_matrix(
    nlp: Language,
    records: Sequence[dict[str, object]],
    labels: Sequence[str],
    *,
    time_budget: TimeBudget | None = None,
) -> np.ndarray:
    """Build a (n_reviews x n_labels) entity-count feature matrix for aspect classification.

    Each row counts, per real review, how many entities of each label the trained NER
    pipeline extracted -- the same feature representation the original notebook builds
    manually per aspect class. `records` are dicts with `"Title"`/`"Comment"` keys (a pandas
    row or a plain dict both work via `record[...]` access).
    """
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(records), len(labels)), dtype=int)
    for row_index, record in enumerate(
        iter_with_progress(
            records, total=len(records), desc="Entity extraction", time_budget=time_budget
        )
    ):
        doc = nlp(_review_document_text(record["Title"], record["Comment"]))
        for ent in doc.ents:
            column = label_index.get(ent.label_)
            if column is not None:
                matrix[row_index, column] += 1
    return matrix
