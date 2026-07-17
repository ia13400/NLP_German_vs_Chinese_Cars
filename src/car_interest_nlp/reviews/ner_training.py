from __future__ import annotations

import random
from pathlib import Path
from typing import cast

import spacy
from spacy.language import Language
from spacy.pipeline import EntityRecognizer
from spacy.training import Example
from spacy.util import minibatch
from tqdm.auto import tqdm

from ..config import PROJECT_ROOT
from ..logging_utils import configure_logging
from ..progress import TimeBudget

logger = configure_logging()

DEFAULT_NER_MODEL_DIR = PROJECT_ROOT / "data" / "interim" / "reviews" / "ner_model"
DEFAULT_N_EPOCHS = 50
DEFAULT_BATCH_SIZE = 8


def build_blank_ner_pipeline(labels: list[str]) -> Language:
    """Build a blank English spaCy pipeline with an untrained `ner` component.

    Trained from scratch, unlike the GDELT chapter's NER (`nlp/ner/train.py`), which fine-
    tunes a real pretrained pipeline -- these aspect labels (ENGINE, COMFORT, SERVICE, ...)
    are review-specific jargon with no equivalent in a general-purpose pretrained model, so
    starting from scratch matches the original approach and is the correct choice here.
    """
    nlp = spacy.blank("en")
    ner = cast(EntityRecognizer, nlp.add_pipe("ner"))
    for label in labels:
        ner.add_label(label)
    return nlp


def _records_to_examples(nlp: Language, records: list[dict[str, object]]) -> list[Example]:
    examples = []
    for record in records:
        doc = nlp.make_doc(str(record["text"]))
        raw_entities = record["entities"]
        assert isinstance(raw_entities, list)
        entities = [tuple(entity) for entity in raw_entities]
        examples.append(Example.from_dict(doc, {"entities": entities}))
    return examples


def train_review_ner_model(
    records: list[dict[str, object]],
    *,
    labels: list[str],
    output_dir: str | Path = DEFAULT_NER_MODEL_DIR,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = 42,
    time_budget: TimeBudget | None = None,
) -> Language:
    """Train a blank spaCy NER pipeline on real, hand-annotated review aspect entities.

    Mirrors the original notebook's training loop (shuffle examples each epoch, minibatch
    updates, log the loss) behind a single live progress bar instead of one print per epoch.
    A `time_budget` stops training gracefully between epochs, matching `nlp/ner/train.py`'s
    convention for every long-running phase in this project.
    """
    nlp = build_blank_ner_pipeline(labels)
    examples = _records_to_examples(nlp, records)
    nlp.initialize(lambda: examples)

    rng = random.Random(seed)
    epoch_bar = tqdm(range(n_epochs), desc="Review NER training epochs", unit="epoch")
    for epoch in epoch_bar:
        if time_budget is not None and time_budget.expired:
            logger.info(
                "Review NER training: time budget of %.1f min reached after %s/%s epochs; stopping.",
                time_budget.minutes,
                epoch,
                n_epochs,
            )
            break
        rng.shuffle(examples)
        losses: dict[str, float] = {}
        for batch in minibatch(examples, size=batch_size):
            nlp.update(batch, losses=losses)
        epoch_bar.set_postfix(loss=f"{losses.get('ner', 0.0):.4f}")
    epoch_bar.close()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    nlp.to_disk(output_path)
    logger.info("Saved trained review NER pipeline to %s", output_path)
    return nlp


def ensure_review_ner_model(
    records: list[dict[str, object]],
    *,
    labels: list[str],
    output_dir: str | Path = DEFAULT_NER_MODEL_DIR,
    n_epochs: int = DEFAULT_N_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    mode: str = "cached",
    time_budget: TimeBudget | None = None,
) -> Language:
    """Reuse an already-trained review NER pipeline (`mode="cached"`, default) or retrain it
    (`mode="live"`).

    Training takes noticeable real wall-clock time (`DEFAULT_N_EPOCHS` epochs over the full
    hand-annotated corpus each run), so a re-run of the notebook should not silently retrain
    from scratch every time -- exactly the same reasoning as `ensure_reviews_dataset`'s
    live/cached toggle for scraping.
    """
    output_path = Path(output_dir)
    if mode == "cached" and (output_path / "meta.json").is_file():
        logger.info("Reusing cached review NER model at %s", output_path)
        return spacy.load(output_path)
    if mode not in ("cached", "live"):
        raise ValueError(f"Unknown execution mode: {mode!r}")
    return train_review_ner_model(
        records,
        labels=labels,
        output_dir=output_path,
        n_epochs=n_epochs,
        batch_size=batch_size,
        time_budget=time_budget,
    )
