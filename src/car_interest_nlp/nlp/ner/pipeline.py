from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import cast

import pandas as pd
import spacy
from spacy.language import Language
from spacy.pipeline import EntityRuler

from .gazetteer import build_entity_ruler_patterns

DEFAULT_SPACY_MODEL = "en_core_web_sm"


@lru_cache(maxsize=4)
def build_stage_a_pipeline(model_name: str = DEFAULT_SPACY_MODEL) -> Language:
    """Build the Stage A NER pipeline: a real pretrained spaCy pipeline plus a rule-based
    `EntityRuler` seeded from the real brand/model/supplier/technology/regulation gazetteers.

    Per "do not train from scratch when a suitable pretrained pipeline is available," this
    loads `model_name`'s real pretrained weights (English NER/tagging/parsing already
    learned from real text) rather than a blank language. The `EntityRuler` is inserted
    `before="ner"` -- spaCy's standard pattern for combining rule-based and statistical
    NER -- so exact real-gazetteer matches (CAR_BRAND/CAR_MODEL/SUPPLIER/TECHNOLOGY/
    COMPONENT/FACTORY/REGULATION) take priority over the pretrained component's generic
    guesses for the same span, while PERSON/ORG/GPE (used here as LOCATION) still come
    from the pretrained model for everything the gazetteer doesn't cover.
    `phrase_matcher_attr="LOWER"` makes gazetteer matching case-insensitive (e.g. "byd"
    inside a lowercased headline still matches the "BYD" pattern).
    """
    nlp = spacy.load(model_name)
    # nlp.add_pipe()'s return type is the generic pipe-callable protocol; the "entity_ruler"
    # factory always actually returns an EntityRuler, which mypy can't infer from the string
    # name alone.
    ruler = cast(
        EntityRuler,
        nlp.add_pipe("entity_ruler", before="ner", config={"phrase_matcher_attr": "LOWER"}),
    )
    # EntityRuler.add_patterns()'s declared type is broader (patterns may also carry
    # token-attribute lists, unused here) than plain dict[str, object] can express without
    # replicating spaCy's internal type alias.
    ruler.add_patterns(build_entity_ruler_patterns())  # type: ignore[arg-type]
    return nlp


def extract_entities(
    texts: Sequence[str], *, model_name: str = DEFAULT_SPACY_MODEL
) -> pd.DataFrame:
    """Run the Stage A pipeline over real texts, returning one row per extracted entity."""
    nlp = build_stage_a_pipeline(model_name)
    rows: list[dict[str, object]] = []
    for doc_index, doc in enumerate(nlp.pipe(texts)):
        for ent in doc.ents:
            rows.append(
                {
                    "doc_index": doc_index,
                    "text": ent.text,
                    "label": ent.label_,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                }
            )
    return pd.DataFrame(rows, columns=["doc_index", "text", "label", "start_char", "end_char"])
