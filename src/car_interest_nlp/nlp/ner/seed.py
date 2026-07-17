from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from .pipeline import DEFAULT_SPACY_MODEL, build_stage_a_pipeline


def generate_seed_annotations(
    texts: Sequence[str], *, model_name: str = DEFAULT_SPACY_MODEL
) -> list[dict[str, object]]:
    """Run Stage A over real texts to produce weak-supervision seed annotations.

    Each record is `{"text": ..., "entities": [[start_char, end_char, label], ...],
    "verified": False}` -- spaCy's classic offset-tuple training shape, plus a `verified`
    flag a human reviewer flips to `True` after checking/correcting a record (see
    `correction.py`). This is a starting point for Stage B training, not ground truth --
    Stage A's rule-based gazetteer labels are precise by construction, but its pretrained-
    model labels (PERSON/ORG/GPE/...) can still be wrong, which is exactly what manual
    correction is for.
    """
    nlp = build_stage_a_pipeline(model_name)
    records: list[dict[str, object]] = []
    for doc in nlp.pipe(texts):
        entities = [[ent.start_char, ent.end_char, ent.label_] for ent in doc.ents]
        records.append({"text": doc.text, "entities": entities, "verified": False})
    return records


def export_seed_for_correction(
    records: Sequence[dict[str, object]], output_path: str | Path
) -> Path:
    """Write seed annotations as one JSON object per line (JSONL) for manual review/editing.

    A human reviewer opens this file, corrects `entities` spans/labels directly (or
    deletes/adds entries), sets `verified` to `true` on each row they've checked, and saves
    it back (typically to a different path -- see `correction.load_corrected_annotations`)
    for `train.train_ner_model` to consume.
    """
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return target
