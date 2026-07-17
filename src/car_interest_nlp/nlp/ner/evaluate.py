from __future__ import annotations

from spacy.language import Language
from spacy.scorer import Scorer
from spacy.training import Example


def evaluate_ner_model(nlp: Language, dev_records: list[dict[str, object]]) -> dict[str, object]:
    """Score a trained NER pipeline on held-out records: precision/recall/F1, overall and
    per label, using spaCy's own `Scorer` (the standard tool for this, not a hand-rolled
    metric).

    `dev_records` are the same raw JSONL-shaped dicts `correction.load_corrected_annotations`
    produces (and the second element `train.train_ner_model()` returns) -- real held-out
    annotations the model was never trained on. Taking raw records rather than pre-built
    spaCy `Example` objects means a trained model (loaded fresh via `spacy.load(model_dir)`)
    can be evaluated in a separate process/script run without needing to keep the training
    call's in-memory objects alive.
    """
    if not dev_records:
        raise ValueError(
            "No held-out dev records to evaluate on -- provide more annotations (the "
            "dev_fraction passed to train_ner_model must yield at least one dev record)."
        )
    scorer = Scorer()
    predicted_examples = []
    for record in dev_records:
        reference_doc = nlp.make_doc(str(record["text"]))
        raw_entities = record["entities"]
        assert isinstance(raw_entities, list)
        entities = [tuple(entity) for entity in raw_entities]
        reference_example = Example.from_dict(reference_doc, {"entities": entities})
        predicted_doc = nlp(str(record["text"]))
        predicted_examples.append(Example(predicted_doc, reference_example.reference))
    scores = scorer.score(predicted_examples)
    return {
        "precision": scores.get("ents_p"),
        "recall": scores.get("ents_r"),
        "f1": scores.get("ents_f"),
        "per_label": scores.get("ents_per_type"),
    }
