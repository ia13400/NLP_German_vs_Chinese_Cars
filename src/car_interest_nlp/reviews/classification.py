from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from ..config import PROJECT_ROOT
from ..logging_utils import configure_logging

logger = configure_logging()

DEFAULT_CLASSIFIER_DIR = PROJECT_ROOT / "data" / "interim" / "reviews" / "classifier"
DEFAULT_CV_SPLITS = 4
DEFAULT_RANDOM_STATE = 42
# The original notebook's README documented "5-Fold CV, Macro F1-Score" but the code called
# `cross_val_score(clf, X, y, cv=cv)` without a `scoring=` argument, which for a classifier
# defaults to plain accuracy, not macro F1 -- confirmed directly by reading scikit-learn's
# `cross_val_score` docs. `f1_macro` is passed explicitly here so the metric actually
# computed matches what both the README and this project's model-selection intent describe.
DEFAULT_SCORING = "f1_macro"


def discover_aspect_classes(records: list[dict[str, object]]) -> list[str]:
    """Real aspect labels from the hand-labeled dataset, in first-seen order.

    Determined from the actual data rather than a hardcoded list, so the classifier's label
    order can never silently drift from what `cars_reviews_labeled.json` actually contains.
    """
    return list(dict.fromkeys(str(record["aspect"]) for record in records))


def build_labels_array(records: list[dict[str, object]], aspect_classes: list[str]) -> np.ndarray:
    class_index = {aspect: index for index, aspect in enumerate(aspect_classes)}
    return np.array([class_index[str(record["aspect"])] for record in records])


def build_candidate_models(*, random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Any]:
    """The three classifiers evaluated in the original notebook, unchanged in kind."""
    return {
        "SVC": Pipeline(
            [("scaler", StandardScaler()), ("classifier", SVC(class_weight="balanced"))]
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", n_estimators=500, random_state=random_state
        ),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(max_iter=1000, C=2.0)),
            ]
        ),
    }


def cross_validate_models(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, dict[str, object]]:
    """Cross-validate every candidate model and return per-model fold scores + their mean."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    models = build_candidate_models(random_state=random_state)
    results: dict[str, dict[str, object]] = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        results[name] = {"scores": scores, "mean_score": float(scores.mean())}
        logger.info("%s: %s scores=%s mean=%.4f", name, scoring, scores, scores.mean())
    return results


def train_aspect_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
    scoring: str = DEFAULT_SCORING,
) -> dict[str, object]:
    """Cross-validate all candidate models, fit the best one on the full data, and return a
    confusion matrix built from out-of-fold predictions (never predictions the model was
    trained on).
    """
    cv_results = cross_validate_models(
        X, y, cv_splits=cv_splits, random_state=random_state, scoring=scoring
    )
    best_model_name = max(cv_results, key=lambda name: float(cv_results[name]["mean_score"]))  # type: ignore[arg-type]
    models = build_candidate_models(random_state=random_state)
    best_model = models[best_model_name]
    best_model.fit(X, y)

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(best_model, X, y, cv=cv)
    logger.info(
        "Best review aspect classifier: %s (mean %s=%.4f)",
        best_model_name,
        scoring,
        cv_results[best_model_name]["mean_score"],
    )
    return {
        "cv_results": cv_results,
        "best_model_name": best_model_name,
        "best_model": best_model,
        "y_pred": y_pred,
        "confusion_matrix": confusion_matrix(y, y_pred),
    }


def save_classifier_artifacts(
    result: dict[str, object],
    aspect_classes: list[str],
    entity_labels: list[str],
    *,
    output_dir: str | Path = DEFAULT_CLASSIFIER_DIR,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(result["best_model"], output_path / "car_reviews_classifier.pkl")
    joblib.dump(aspect_classes, output_path / "aspect_classes.pkl")
    joblib.dump(entity_labels, output_path / "entity_labels.pkl")
    logger.info("Saved review aspect classifier artifacts to %s", output_path)
    return output_path


def load_classifier_artifacts(
    *, output_dir: str | Path = DEFAULT_CLASSIFIER_DIR
) -> tuple[Any, list[str], list[str]]:
    output_path = Path(output_dir)
    model = joblib.load(output_path / "car_reviews_classifier.pkl")
    aspect_classes = joblib.load(output_path / "aspect_classes.pkl")
    entity_labels = joblib.load(output_path / "entity_labels.pkl")
    return model, aspect_classes, entity_labels


def ensure_aspect_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    aspect_classes: list[str],
    entity_labels: list[str],
    output_dir: str | Path = DEFAULT_CLASSIFIER_DIR,
    mode: str = "cached",
    cv_splits: int = DEFAULT_CV_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Reuse an already-trained aspect classifier (`mode="cached"`, default) or retrain it
    (`mode="live"`), matching `ner_training.ensure_review_ner_model`'s live/cached convention.

    A cached load skips cross-validation entirely (there is nothing to re-score), so
    `"cv_results"`/`"y_pred"`/`"confusion_matrix"` are only present when this call actually
    trained a fresh model.
    """
    output_path = Path(output_dir)
    if mode == "cached" and (output_path / "car_reviews_classifier.pkl").is_file():
        model, cached_aspect_classes, cached_entity_labels = load_classifier_artifacts(
            output_dir=output_path
        )
        logger.info("Reusing cached review aspect classifier at %s", output_path)
        return {
            "best_model": model,
            "aspect_classes": cached_aspect_classes,
            "entity_labels": cached_entity_labels,
            "from_cache": True,
        }
    if mode not in ("cached", "live"):
        raise ValueError(f"Unknown execution mode: {mode!r}")

    result = train_aspect_classifier(X, y, cv_splits=cv_splits, random_state=random_state)
    save_classifier_artifacts(result, aspect_classes, entity_labels, output_dir=output_path)
    result["aspect_classes"] = aspect_classes
    result["entity_labels"] = entity_labels
    result["from_cache"] = False
    return result
