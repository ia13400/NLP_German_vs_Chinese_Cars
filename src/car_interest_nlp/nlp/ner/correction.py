from __future__ import annotations

import json
from pathlib import Path

from .gazetteer import CUSTOM_NER_LABELS

# spaCy's pretrained en_core_web_sm labels that may also appear in seed/corrected records
# alongside the custom labels -- both are valid, only something outside both sets indicates
# a real typo/corruption in a hand-edited correction file.
_PRETRAINED_LABELS = frozenset(
    {
        "PERSON",
        "ORG",
        "GPE",
        "LOC",
        "NORP",
        "DATE",
        "MONEY",
        "CARDINAL",
        "PERCENT",
        "TIME",
        "ORDINAL",
        "QUANTITY",
        "FAC",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
        "LANGUAGE",
        "PRODUCT",
    }
)
_KNOWN_LABELS = frozenset(CUSTOM_NER_LABELS) | _PRETRAINED_LABELS


class NerAnnotationError(ValueError):
    """Raised when a seed/correction JSONL record is malformed -- never silently skipped."""


def _validate_record(record: dict[str, object], *, line_number: int) -> None:
    text = record.get("text")
    entities = record.get("entities")
    if not isinstance(text, str):
        raise NerAnnotationError(f"Line {line_number}: 'text' must be a string.")
    if not isinstance(entities, list):
        raise NerAnnotationError(f"Line {line_number}: 'entities' must be a list.")
    for entity in entities:
        if not (isinstance(entity, list) and len(entity) == 3):
            raise NerAnnotationError(
                f"Line {line_number}: each entity must be [start_char, end_char, label], got {entity!r}."
            )
        start, end, label = entity
        if not (isinstance(start, int) and isinstance(end, int) and 0 <= start < end <= len(text)):
            raise NerAnnotationError(
                f"Line {line_number}: entity span [{start}, {end}) is out of bounds for text of "
                f"length {len(text)}."
            )
        if label not in _KNOWN_LABELS:
            raise NerAnnotationError(
                f"Line {line_number}: unknown label {label!r} (expected one of "
                f"{sorted(_KNOWN_LABELS)})."
            )


def load_corrected_annotations(
    path: str | Path, *, require_verified: bool = True
) -> list[dict[str, object]]:
    """Load a (possibly hand-corrected) seed JSONL file, validating every record.

    Raises `NerAnnotationError` on the first malformed record (bad span, unknown label,
    wrong shape) rather than silently dropping it -- a corrupted correction file should
    never quietly train on bad spans. `require_verified=True` (default) keeps only records
    a human has explicitly flagged `"verified": true`; pass `False` to also include
    not-yet-reviewed Stage A seed records (e.g. for a smoke test of the training code path).
    """
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            _validate_record(record, line_number=line_number)
            if require_verified and not record.get("verified", False):
                continue
            records.append(record)
    return records
