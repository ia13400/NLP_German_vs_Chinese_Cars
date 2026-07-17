from __future__ import annotations

import argparse

import spacy

from car_interest_nlp.nlp.ner.correction import load_corrected_annotations
from car_interest_nlp.nlp.ner.evaluate import evaluate_ner_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NER model (see scripts/train_ner_model.py) on its "
        "held-out dev split -- real precision/recall/F1 per label via spaCy's own Scorer."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--dev-split", required=True)
    args = parser.parse_args()

    nlp = spacy.load(args.model_dir)
    dev_records = load_corrected_annotations(args.dev_split, require_verified=False)
    scores = evaluate_ner_model(nlp, dev_records)
    print(scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
