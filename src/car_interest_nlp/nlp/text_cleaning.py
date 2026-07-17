from __future__ import annotations

import re
from collections.abc import Iterable

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from ..config import load_project_config

# Generic news-source boilerplate phrases that appear across many articles regardless of
# topic (bylines, copyright notices, cookie/subscription prompts) -- stripped before
# tokenizing so they don't drown out genuine content words in word clouds/TF-IDF.
BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"all rights reserved",
    r"click here to subscribe",
    r"sign up for our newsletter",
    r"share this article",
    r"terms of service",
    r"privacy policy",
    r"cookie policy",
    r"read more:?",
    r"follow us on",
)

_BOILERPLATE_REGEX = re.compile("|".join(BOILERPLATE_PATTERNS), re.IGNORECASE)
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z'-]+")


def get_brand_exclusion_terms() -> set[str]:
    """Real brand names + aliases from `configs/brands.yaml`, lowercased, for exclusion.

    Shared by every word cloud category (`nlp/wordclouds.py`) so a brand's own name never
    dominates its own "associated topics" word cloud -- the point of those clouds is what
    the brand is discussed *alongside*, not the brand name itself.
    """
    config = load_project_config()
    terms: set[str] = set()
    for group in config["brands"].values():
        for entry in group:
            terms.add(entry["canonical_name"].lower())
            for alias in entry.get("aliases", []):
                terms.add(alias.lower())
    return terms


def strip_boilerplate(text: str) -> str:
    return _BOILERPLATE_REGEX.sub(" ", text)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_PATTERN.finditer(text)]


def clean_tokens(
    text: str,
    *,
    extra_exclude: Iterable[str] = (),
    exclude_brands: bool = True,
    generic_terms: Iterable[str] = ("car", "cars", "vehicle", "vehicles"),
    min_length: int = 3,
) -> list[str]:
    """Tokenize `text` and filter out stopwords/boilerplate/brand names/generic terms.

    This is the shared filtering step behind every word cloud category. `exclude_brands`/
    `generic_terms`/`extra_exclude` are configurable per call so a given word cloud can
    decide what counts as noise for its own purpose -- e.g. a technology-focused word cloud
    still wants brand names excluded, but a yearly-comparison cloud might keep
    "car"/"vehicle" in scope if that comparison is specifically about them. Uses
    scikit-learn's standard `ENGLISH_STOP_WORDS` list, not a hand-rolled one.
    """
    exclude = (
        set(ENGLISH_STOP_WORDS)
        | {term.lower() for term in generic_terms}
        | {term.lower() for term in extra_exclude}
    )
    if exclude_brands:
        exclude |= get_brand_exclusion_terms()
    tokens = tokenize(strip_boilerplate(text))
    return [token for token in tokens if token not in exclude and len(token) >= min_length]


def clean_corpus(
    texts: Iterable[str],
    *,
    extra_exclude: Iterable[str] = (),
    exclude_brands: bool = True,
    generic_terms: Iterable[str] = ("car", "cars", "vehicle", "vehicles"),
    min_length: int = 3,
) -> list[list[str]]:
    return [
        clean_tokens(
            text,
            extra_exclude=extra_exclude,
            exclude_brands=exclude_brands,
            generic_terms=generic_terms,
            min_length=min_length,
        )
        for text in texts
    ]
