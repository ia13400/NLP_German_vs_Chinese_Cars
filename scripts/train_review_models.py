from __future__ import annotations

import argparse

from car_interest_nlp.config import resolve_project_path
from car_interest_nlp.progress import TimeBudget
from car_interest_nlp.reviews.classification import (
    build_labels_array,
    discover_aspect_classes,
    ensure_aspect_classifier,
)
from car_interest_nlp.reviews.features import build_entity_count_matrix
from car_interest_nlp.reviews.ner_training import DEFAULT_N_EPOCHS, ensure_review_ner_model
from car_interest_nlp.reviews.text_prep import (
    build_ner_training_records,
    discover_entity_labels,
    load_inline_annotated_articles,
    load_labeled_reviews,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train (or reuse, mode=cached) the review aspect NER model and "
        "classifier from the real hand-annotated review datasets under data/raw/reviews/."
    )
    parser.add_argument("--ner-inline-file", default="data/raw/reviews/cars_reviews_ner_inline.txt")
    parser.add_argument("--labeled-file", default="data/raw/reviews/cars_reviews_labeled.json")
    parser.add_argument("--ner-model-dir", default="data/interim/reviews/ner_model")
    parser.add_argument("--classifier-dir", default="data/interim/reviews/classifier")
    parser.add_argument("--ner-epochs", type=int, default=DEFAULT_N_EPOCHS)
    parser.add_argument(
        "--mode",
        choices=["live", "cached"],
        default="live",
        help="'live' always retrains both models; 'cached' reuses whatever is already saved.",
    )
    parser.add_argument("--ner-time-limit-minutes", type=float, default=15.0)
    args = parser.parse_args()

    annotated_articles = load_inline_annotated_articles(resolve_project_path(args.ner_inline_file))
    ner_training_records = build_ner_training_records(annotated_articles)
    entity_labels = discover_entity_labels(ner_training_records)

    labeled_reviews = load_labeled_reviews(resolve_project_path(args.labeled_file))
    aspect_classes = discover_aspect_classes(labeled_reviews)

    ner_model = ensure_review_ner_model(
        ner_training_records,
        labels=entity_labels,
        output_dir=args.ner_model_dir,
        n_epochs=args.ner_epochs,
        mode=args.mode,
        time_budget=TimeBudget(minutes=args.ner_time_limit_minutes),
    )

    X = build_entity_count_matrix(ner_model, labeled_reviews, entity_labels)
    y = build_labels_array(labeled_reviews, aspect_classes)
    result = ensure_aspect_classifier(
        X,
        y,
        aspect_classes=aspect_classes,
        entity_labels=entity_labels,
        output_dir=args.classifier_dir,
        mode=args.mode,
    )

    print(f"NER model saved to {args.ner_model_dir} ({len(entity_labels)} labels).")
    print(f"Classifier saved to {args.classifier_dir} ({len(aspect_classes)} aspect classes).")
    if not result["from_cache"]:
        print(f"Best classifier: {result['best_model_name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
