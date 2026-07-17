from __future__ import annotations

import json
import re
from pathlib import Path

# Real hand-annotated training file uses inline bracket markup "[span]_LABEL" (e.g.
# "[driving experience]_PERFORMANCE") to mark aspect-entity spans directly inside the
# review text -- ported as-is from the original manually annotated dataset.
_ARTICLE_SEPARATOR_PATTERN = re.compile(r"\n-{10,}\n")

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b")
_LONG_DIGIT_SEQUENCE_PATTERN = re.compile(r"(?:\+?\d[\d\s()/.-]{6,}\d)")
_MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")


def clean_review_text(text: str) -> str:
    """Strip URLs, e-mail addresses, phone-number-like digit runs, and repeated whitespace.

    Applied before inline-annotation parsing so the character offsets `parse_inline_annotations`
    produces line up with the cleaned text actually used for NER training.
    """
    text = _URL_PATTERN.sub("", text)
    text = _EMAIL_PATTERN.sub("", text)
    text = _LONG_DIGIT_SEQUENCE_PATTERN.sub("", text)
    text = _MULTI_SPACE_PATTERN.sub(" ", text)
    return text.strip()


def split_inline_annotated_articles(raw_text: str) -> list[str]:
    """Split a raw inline-annotated file into its individual per-review articles.

    Articles are separated by a line of 10+ dashes, matching the original manually
    annotated dataset's own separator convention.
    """
    return [
        article.strip() for article in _ARTICLE_SEPARATOR_PATTERN.split(raw_text) if article.strip()
    ]


def load_inline_annotated_articles(path: str | Path) -> list[str]:
    raw_text = Path(path).read_text(encoding="utf-8")
    return split_inline_annotated_articles(raw_text)


def parse_inline_annotations(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Convert inline bracket markup "[span]_LABEL" into spaCy-style character offsets.

    Returns `(plain_text, entities)` where `entities` is a list of `(start_char, end_char,
    label)` tuples into `plain_text` (the brackets/labels removed). Ported directly from the
    original notebook's `annotation()` function, since its exact bracket-scanning logic (not
    a regex) is what the real hand-annotated training file was written against.
    """
    entities: list[tuple[int, int, str]] = []
    while True:
        start_bracket = text.find("[")
        if start_bracket < 0:
            break
        end_bracket = text.find("]", start_bracket)
        label_start = end_bracket + 2  # skip "]_"
        label_end = label_start
        while label_end < len(text) and text[label_end : label_end + 1].isalpha():
            label_end += 1

        span_text = text[start_bracket + 1 : end_bracket]
        label = text[label_start:label_end]
        text = text[:start_bracket] + span_text + text[label_end:]

        entities.append((start_bracket, end_bracket - 1, label))
    return text, entities


def build_ner_training_records(articles: list[str]) -> list[dict[str, object]]:
    """Build spaCy-shaped training records from real inline-annotated review articles.

    Each article is cleaned (`clean_review_text`) before its bracket markup is parsed, so
    the resulting character offsets are valid for the exact text spaCy will train on.
    """
    records: list[dict[str, object]] = []
    for article in articles:
        cleaned = clean_review_text(article)
        plain_text, entities = parse_inline_annotations(cleaned)
        records.append(
            {
                "text": plain_text,
                "entities": [[start, end, label] for start, end, label in entities],
            }
        )
    return records


def load_labeled_reviews(path: str | Path) -> list[dict[str, object]]:
    """Load the real, hand-labeled aspect-classification dataset (Title/Comment/Rating/
    mark/aspect per review) unchanged.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover_entity_labels(records: list[dict[str, object]]) -> list[str]:
    """Collect every distinct entity label actually present in `records`, sorted.

    Used instead of a hand-maintained label list so the NER model's label set and the
    feature-extraction label set (`reviews.features`) are always built from the same real
    source of truth and can never silently drift apart.
    """
    labels: set[str] = set()
    for record in records:
        raw_entities = record["entities"]
        assert isinstance(raw_entities, list)
        for entity in raw_entities:
            labels.add(str(entity[2]))
    return sorted(labels)
