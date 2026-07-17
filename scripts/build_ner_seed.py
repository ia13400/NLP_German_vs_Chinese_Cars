from __future__ import annotations

import argparse

from car_interest_nlp.nlp.corpus import load_gdelt_corpus
from car_interest_nlp.nlp.ner.seed import export_seed_for_correction, generate_seed_annotations


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Stage A (pretrained spaCy pipeline + real automotive gazetteer "
        "EntityRuler) over the real GDELT article corpus to produce weak-supervision seed "
        "annotations for manual correction (see configs/ner_gazetteer.yaml and "
        "src/car_interest_nlp/nlp/ner/correction.py). Open the output JSONL, correct any "
        "wrong spans/labels, and set 'verified': true per row you've checked before "
        "training on it."
    )
    parser.add_argument("--output", default="data/interim/ner/seed_annotations.jsonl")
    parser.add_argument(
        "--max-documents", type=int, default=None, help="Limit to the first N documents."
    )
    args = parser.parse_args()

    corpus = load_gdelt_corpus()
    if corpus.empty:
        print("No GDELT article corpus available yet -- run scripts/download_gdelt_news.py first.")
        return 0

    texts = corpus["text"].dropna().tolist()
    if args.max_documents:
        texts = texts[: args.max_documents]

    records = generate_seed_annotations(texts)
    path = export_seed_for_correction(records, args.output)
    print(
        f"Wrote {len(records)} seed records to {path}. Review/correct it, mark "
        f"'verified': true per row, then run scripts/train_ner_model.py --annotations "
        f"{path}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
