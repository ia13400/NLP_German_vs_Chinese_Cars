from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRecognizer
from spacy.training import Example
from tqdm import tqdm

from ...logging_utils import configure_logging
from ...progress import TimeBudget
from .gazetteer import CUSTOM_NER_LABELS
from .pipeline import DEFAULT_SPACY_MODEL

logger = configure_logging()

DEFAULT_BATCH_SIZE = 8
DEFAULT_DROPOUT = 0.2
DEFAULT_DEV_FRACTION = 0.2


def _records_to_examples(nlp: Language, records: list[dict[str, object]]) -> list[Example]:
    examples = []
    for record in records:
        doc = nlp.make_doc(str(record["text"]))
        raw_entities = record["entities"]
        assert isinstance(raw_entities, list)
        entities = [tuple(entity) for entity in raw_entities]
        examples.append(Example.from_dict(doc, {"entities": entities}))
    return examples


def split_train_dev(
    records: list[dict[str, object]], *, dev_fraction: float = DEFAULT_DEV_FRACTION, seed: int = 42
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) <= 1:
        return shuffled, []
    cutoff = max(1, int(len(shuffled) * (1 - dev_fraction)))
    return shuffled[:cutoff], shuffled[cutoff:]


def train_ner_model(
    records: list[dict[str, object]],
    *,
    base_model: str = DEFAULT_SPACY_MODEL,
    output_dir: str | Path,
    n_epochs: int = 10,
    batch_size: int = DEFAULT_BATCH_SIZE,
    dropout: float = DEFAULT_DROPOUT,
    dev_fraction: float = DEFAULT_DEV_FRACTION,
    time_budget: TimeBudget | None = None,
) -> tuple[Language, list[dict[str, object]]]:
    """Fine-tune a real pretrained spaCy pipeline's NER component on corrected annotations.

    Per "do not train from scratch when a suitable pretrained pipeline is available," this
    loads `base_model`'s real pretrained weights fresh (not the cached Stage A pipeline
    with its `EntityRuler` still attached) and only updates the "ner" component -- every
    other pipe (tagger, parser, lemmatizer, ...) is disabled during training, standard
    spaCy fine-tuning practice, so they aren't disturbed by NER-specific gradient updates.
    `records` should come from `correction.load_corrected_annotations()` (human-verified)
    rather than raw Stage A seed output. Returns `(trained_pipeline, dev_records)` --
    `dev_records` are the raw held-out JSONL-shaped dicts (not in-memory `Example` objects),
    so a separate process/script can persist them and evaluate the saved model later
    without needing to keep this call's Python objects alive (see
    `evaluate.evaluate_ner_model`, which accepts this same raw record shape).
    """
    train_records, dev_records = split_train_dev(records, dev_fraction=dev_fraction)
    if not train_records:
        raise ValueError(
            "No training records after the train/dev split -- provide more annotations."
        )

    nlp = spacy.load(base_model)
    if "ner" not in nlp.pipe_names:
        nlp.add_pipe("ner")
    # nlp.get_pipe()'s return type is the generic pipe-callable protocol; the "ner" pipe is
    # always actually an EntityRecognizer, which mypy can't infer from the string name alone.
    ner = cast(EntityRecognizer, nlp.get_pipe("ner"))
    for label in CUSTOM_NER_LABELS:
        ner.add_label(label)

    train_examples = _records_to_examples(nlp, train_records)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.resume_training()
        epoch_bar = tqdm(range(n_epochs), desc="NER training epochs", unit="epoch")
        for epoch in epoch_bar:
            if time_budget is not None and time_budget.expired:
                logger.info(
                    "NER training: time budget of %.1f min reached after %s/%s epochs; stopping.",
                    time_budget.minutes,
                    epoch,
                    n_epochs,
                )
                break
            random.Random(42 + epoch).shuffle(train_examples)
            losses: dict[str, float] = {}
            for start in range(0, len(train_examples), batch_size):
                batch = train_examples[start : start + batch_size]
                nlp.update(batch, drop=dropout, losses=losses, sgd=optimizer)
            epoch_bar.set_postfix(loss=f"{losses.get('ner', 0.0):.4f}")
        epoch_bar.close()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    logger.info("Saved trained NER pipeline to %s", output_path)
    return nlp, dev_records
