from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def compute_tfidf(
    documents: Sequence[str],
    *,
    max_features: int = 200,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 2,
) -> pd.Series:
    """Compute the mean TF-IDF weight per term across `documents`, sorted descending.

    `documents` should already be cleaned (see `text_cleaning.clean_tokens`, joined back
    into space-separated pseudo-documents) so brand names/boilerplate are excluded the same
    way for both frequency and TF-IDF word clouds. Terms frequent in a few documents but
    not universally common across the whole corpus score higher here than in a raw
    frequency count, surfacing genuinely distinctive vocabulary rather than just common
    words. `min_df=2` requires a term to appear in at least 2 documents, so a single
    article's idiosyncratic wording can't dominate.
    """
    if not documents:
        return pd.Series(dtype=float)
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=min_df)
    matrix = vectorizer.fit_transform(documents)
    mean_weights = matrix.mean(axis=0).A1
    return pd.Series(mean_weights, index=vectorizer.get_feature_names_out()).sort_values(
        ascending=False
    )
