from __future__ import annotations

import argparse
from pathlib import Path

from car_interest_nlp.nlp.ner.correction import load_corrected_annotations
from car_interest_nlp.nlp.ner.seed import export_seed_for_correction
from car_interest_nlp.nlp.ner.train import train_ner_model
from car_interest_nlp.progress import TimeBudget


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fine-tune a real pretrained spaCy pipeline's NER component on "
        "human-corrected annotations (see scripts/build_ner_seed.py) -- never trains from "
        "scratch. The held-out dev split is also written to disk so "
        "scripts/evaluate_ner_model.py can score the saved model independently."
    )
    parser.add_argument("--annotations", required=True, help="Path to a corrected seed JSONL file.")
    parser.add_argument("--output-dir", default="data/interim/ner/model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--time-limit-minutes", type=float, default=15.0)
    args = parser.parse_args()

    records = load_corrected_annotations(args.annotations)
    if not records:
        print(
            f"No verified records found in {args.annotations!r} -- mark rows "
            f"'verified': true first."
        )
        return 0

    _nlp, dev_records = train_ner_model(
        records,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        time_budget=TimeBudget(minutes=args.time_limit_minutes),
    )
    dev_path = Path(args.output_dir) / "dev_split.jsonl"
    export_seed_for_correction(dev_records, dev_path)

    print(f"Trained on {len(records) - len(dev_records)} records, {len(dev_records)} held out.")
    print(f"Model saved to {args.output_dir}; dev split saved to {dev_path}.")
    print(
        f"Run: uv run python scripts/evaluate_ner_model.py --model-dir {args.output_dir} "
        f"--dev-split {dev_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
