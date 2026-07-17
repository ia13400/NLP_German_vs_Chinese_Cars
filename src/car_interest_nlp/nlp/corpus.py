from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..data.article_text import read_cached_article_text
from ..data.gdelt_dataset_builder import (
    DEFAULT_GDELT_RAW_DIRECTORY,
    load_cached_article_frame,
    resolve_scope,
)
from ..logging_utils import configure_logging

logger = configure_logging()


def assemble_text_corpus(article_frame: pd.DataFrame, article_text_dir: str | Path) -> pd.DataFrame:
    """Attach real cached full article text (or fall back to the real GDELT title) per row.

    `article_frame` is the tidied GDELT `artlist` output -- one row per real article, with
    `url`/`title`/`seendate`/`brand` columns (see
    `gdelt_dataset_builder.collect_article_urls`/`load_cached_article_frame`). Rows whose
    URL was never successfully scraped (see `article_text.fetch_article_texts`) fall back to
    just the real GDELT title -- still real text, just much shorter -- rather than being
    dropped, so word clouds/NER can run over whatever text is actually available, with
    `text_source` recording which case applied per row.
    """
    if article_frame.empty:
        return article_frame
    data = article_frame.copy()
    data["scraped_text"] = data["url"].map(
        lambda url: read_cached_article_text(article_text_dir, url)
    )
    data["text"] = data["scraped_text"].fillna(data.get("title", ""))
    data["text_source"] = data["scraped_text"].notna().map({True: "scraped", False: "title_only"})
    data["year"] = pd.to_datetime(data["seendate"], errors="coerce").dt.year
    return data.drop(columns=["scraped_text"])


def load_gdelt_corpus(
    *,
    raw_directory: str | Path | None = None,
    article_text_dir: str | Path = DEFAULT_GDELT_RAW_DIRECTORY.parent / "gdelt_articles",
    start_date: date | None = None,
    end_date: date | None = None,
    top_n_brands: int | None = None,
) -> pd.DataFrame:
    """Load the real per-article GDELT corpus (with `brand_group` and `text` attached) for
    every `wordclouds`/`ner` function that needs individual article rows, not the annual
    count series `gdelt_dataset_builder.build_gdelt_analysis_dataset()` returns.

    Works over whatever article chunks (and, separately, scraped article text) are cached
    so far -- see `gdelt_dataset_builder.load_cached_article_frame`'s docstring for the
    same partial-coverage reasoning every GDELT function in this project follows.
    """
    raw_dir, start, end, top_n = resolve_scope(raw_directory, start_date, end_date, top_n_brands)
    article_frame, _brand_group_map, found, total = load_cached_article_frame(
        raw_dir, start, end, top_n
    )
    logger.info(
        "Loaded GDELT corpus from %s/%s expected (brand, month) article chunks", found, total
    )
    return assemble_text_corpus(article_frame, article_text_dir)
