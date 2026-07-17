from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from ..config import load_yaml_config
from ..logging_utils import configure_logging
from ..visualization.style import configure_plot_style
from .text_cleaning import clean_tokens
from .tfidf import compute_tfidf

logger = configure_logging()

DEFAULT_WORDCLOUD_WIDTH = 1600
DEFAULT_WORDCLOUD_HEIGHT = 900


def _render_word_cloud(
    frequencies: dict[str, float], output_path: str | Path, *, title: str, colormap: str = "viridis"
) -> Path:
    """Render a single word cloud PNG from a term -> weight mapping."""
    if not frequencies:
        raise ValueError(f"No terms to render a word cloud for {title!r} -- corpus is empty.")
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    cloud = WordCloud(
        width=DEFAULT_WORDCLOUD_WIDTH,
        height=DEFAULT_WORDCLOUD_HEIGHT,
        background_color="white",
        colormap=colormap,
        prefer_horizontal=0.9,
    ).generate_from_frequencies(frequencies)

    configure_plot_style()
    fig, ax = plt.subplots(figsize=(12, 6.75))
    ax.imshow(cloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(target, dpi=200)
    plt.close(fig)
    return target


def word_cloud_from_texts(
    texts: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    colormap: str = "viridis",
    exclude_brands: bool = True,
    generic_terms: Sequence[str] = ("car", "cars", "vehicle", "vehicles"),
    extra_exclude: Sequence[str] = (),
) -> Path:
    """Real-frequency word cloud: a plain token count over the cleaned corpus."""
    tokens = [
        token
        for text in texts
        for token in clean_tokens(
            text,
            exclude_brands=exclude_brands,
            generic_terms=generic_terms,
            extra_exclude=extra_exclude,
        )
    ]
    return _render_word_cloud(dict(Counter(tokens)), output_path, title=title, colormap=colormap)


def tfidf_word_cloud_from_texts(
    texts: Sequence[str],
    output_path: str | Path,
    *,
    title: str,
    colormap: str = "plasma",
    exclude_brands: bool = True,
    generic_terms: Sequence[str] = ("car", "cars", "vehicle", "vehicles"),
    extra_exclude: Sequence[str] = (),
) -> Path:
    """TF-IDF-weighted word cloud: distinctive vocabulary, not just frequent vocabulary."""
    cleaned_docs = [
        " ".join(
            clean_tokens(
                text,
                exclude_brands=exclude_brands,
                generic_terms=generic_terms,
                extra_exclude=extra_exclude,
            )
        )
        for text in texts
    ]
    scores = compute_tfidf(cleaned_docs)
    return _render_word_cloud(scores.to_dict(), output_path, title=title, colormap=colormap)


_GROUP_LABEL_DE = {"german": "Deutsche Marken", "chinese": "Chinesische Marken"}


def generate_brand_group_word_cloud(
    corpus: pd.DataFrame, group: str, output_path: str | Path, *, tfidf: bool = False
) -> Path:
    """Word cloud (frequency or TF-IDF) over every real article about brands in `group`.

    `corpus` is `nlp.corpus.assemble_text_corpus()`'s output (must have `brand_group`/`text`
    columns, e.g. from `gdelt.build_gdelt_annual_series`'s row shape or an equivalent
    per-article frame with `brand_group` already attached).
    """
    texts = corpus.loc[corpus["brand_group"] == group, "text"].dropna().tolist()
    label = _GROUP_LABEL_DE.get(group, group)
    fn = tfidf_word_cloud_from_texts if tfidf else word_cloud_from_texts
    kind = "TF-IDF" if tfidf else "Häufigkeit"
    return fn(texts, output_path, title=f"{label} in den Medien ({kind})")


def generate_yearly_word_clouds(corpus: pd.DataFrame, output_dir: str | Path) -> dict[int, Path]:
    """One frequency word cloud per real reporting year present in `corpus`."""
    output_dir = Path(output_dir)
    paths: dict[int, Path] = {}
    for year, group in corpus.dropna(subset=["year"]).groupby("year"):
        texts = group["text"].dropna().tolist()
        if not texts:
            continue
        year_int = int(year)
        paths[year_int] = word_cloud_from_texts(
            texts,
            output_dir / f"gdelt_wordcloud_{year_int}.png",
            title=f"Medienberichterstattung {year_int}",
        )
    return paths


def _gazetteer_terms(labels: Sequence[str]) -> list[str]:
    """Flatten the real curated terms for the given NER labels from `configs/ner_gazetteer.yaml`."""
    gazetteer = load_yaml_config("ner_gazetteer").get("gazetteer", {})
    terms: list[str] = []
    for label in labels:
        entries = gazetteer.get(label, [])
        if isinstance(entries, dict):
            for models in entries.values():
                terms.extend(models)
        else:
            terms.extend(entries)
    return terms


def _filter_corpus_by_terms(corpus: pd.DataFrame, terms: Sequence[str]) -> pd.DataFrame:
    if not terms:
        return corpus.iloc[0:0]
    pattern = re.compile("|".join(re.escape(term) for term in terms), re.IGNORECASE)
    mask = corpus["text"].fillna("").str.contains(pattern, regex=True)
    return corpus[mask]


def generate_technology_word_cloud(
    corpus: pd.DataFrame, output_path: str | Path, *, tfidf: bool = False
) -> Path:
    """Word cloud over articles that mention a real technology/component term.

    Documents are selected by containing at least one `TECHNOLOGY`/`COMPONENT` gazetteer
    term (`configs/ner_gazetteer.yaml`) -- the word cloud itself shows the surrounding
    vocabulary of technology-related coverage, not just the gazetteer terms themselves
    (which would be circular).
    """
    subset = _filter_corpus_by_terms(corpus, _gazetteer_terms(["TECHNOLOGY", "COMPONENT"]))
    texts = subset["text"].dropna().tolist()
    fn = tfidf_word_cloud_from_texts if tfidf else word_cloud_from_texts
    return fn(texts, output_path, title="Technologie-bezogene Berichterstattung", colormap="cool")


def generate_regulation_word_cloud(
    corpus: pd.DataFrame, output_path: str | Path, *, tfidf: bool = False
) -> Path:
    """Word cloud over articles that mention a real regulation/tariff term (see `REGULATION`
    entries in `configs/ner_gazetteer.yaml`); same "surrounding vocabulary" reasoning as
    `generate_technology_word_cloud`.
    """
    subset = _filter_corpus_by_terms(corpus, _gazetteer_terms(["REGULATION"]))
    texts = subset["text"].dropna().tolist()
    fn = tfidf_word_cloud_from_texts if tfidf else word_cloud_from_texts
    return fn(
        texts, output_path, title="Regulierung & Zölle -- Berichterstattung", colormap="autumn"
    )


def generate_all_word_clouds(corpus: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    """Generate all 7 required word cloud categories, skipping any that have no input text yet.

    With real, partial GDELT coverage, it is entirely normal for *some* categories to have
    no matching articles yet even when others do (e.g. only German-brand chunks have been
    fetched so far, or no article yet mentions a regulation term) -- one empty category
    must not abort the rest. Each category is generated independently; a category with no
    text is logged and skipped (not a hidden failure -- the return dict simply omits it, so
    callers can tell exactly which categories actually rendered).
    """
    output_dir = Path(output_dir)
    results: dict[str, Path] = {}

    def _try(name: str, build: Callable[[], Path]) -> None:
        try:
            results[name] = build()
        except ValueError as exc:
            logger.info("Skipping %r word cloud: %s", name, exc)

    _try(
        "german",
        lambda: generate_brand_group_word_cloud(
            corpus, "german", output_dir / "wordcloud_german.png"
        ),
    )
    _try(
        "chinese",
        lambda: generate_brand_group_word_cloud(
            corpus, "chinese", output_dir / "wordcloud_chinese.png"
        ),
    )
    _try(
        "german_tfidf",
        lambda: generate_brand_group_word_cloud(
            corpus, "german", output_dir / "wordcloud_german_tfidf.png", tfidf=True
        ),
    )
    _try(
        "chinese_tfidf",
        lambda: generate_brand_group_word_cloud(
            corpus, "chinese", output_dir / "wordcloud_chinese_tfidf.png", tfidf=True
        ),
    )
    _try(
        "technology",
        lambda: generate_technology_word_cloud(corpus, output_dir / "wordcloud_technology.png"),
    )
    _try(
        "regulation",
        lambda: generate_regulation_word_cloud(corpus, output_dir / "wordcloud_regulation.png"),
    )

    try:
        yearly_paths = generate_yearly_word_clouds(corpus, output_dir / "yearly")
        results.update({f"year_{year}": path for year, path in yearly_paths.items()})
    except ValueError as exc:
        logger.info("Skipping yearly word clouds: %s", exc)

    return results
