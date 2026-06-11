from __future__ import annotations

import logging
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from config import (
    EVALUATION_OUTPUT_DIR,
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TUNED_MODEL_PATH,
    TUNING_REPORT_PATH,
    TUNING_RESULTS_PATH,
)
from model.metrics_utils import compute_metrics
from model.model_utils import ModelArtifact, load_processed_dataset, save_model_artifact
from model.train import split_features_and_labels

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer()),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def create_param_grid() -> dict[str, list]:
    return {
        "tfidf__max_features": [5000, TFIDF_MAX_FEATURES],
        "tfidf__ngram_range": [(1, 1), (1, 2)],
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }


def tune_model(x_train: pd.Series, y_train: pd.Series) -> GridSearchCV:
    search = GridSearchCV(
        estimator=create_pipeline(),
        param_grid=create_param_grid(),
        scoring="f1",
        cv=3,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(x_train, y_train)
    return search


def save_tuning_results(search: GridSearchCV) -> None:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(search.cv_results_)
    columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_tfidf__max_features",
        "param_tfidf__ngram_range",
        "param_classifier__C",
        "param_classifier__class_weight",
    ]

    results[columns].sort_values("rank_test_score").to_csv(
        TUNING_RESULTS_PATH,
        index=False,
    )


def save_tuned_artifact(
    search: GridSearchCV,
    x_test: pd.Series,
    y_test: pd.Series,
) -> ModelArtifact:
    best_pipeline = search.best_estimator_

    artifact = ModelArtifact(
        pipeline=best_pipeline,
        vectorizer=best_pipeline.named_steps["tfidf"],
        classifier=best_pipeline.named_steps["classifier"],
        model_name="Tuned Logistic Regression",
        X_test=x_test,
        y_test=y_test,
    )

    save_model_artifact(artifact, TUNED_MODEL_PATH)
    return artifact


def build_tuning_report(
    search: GridSearchCV,
    metrics: dict[str, float],
) -> str:
    lines = [
        "Fake Job Offer Detector - raport strojenia hiperparametrów",
        "",
        "Model bazowy:",
        "Logistic Regression + TF-IDF",
        "",
        "Metoda:",
        "GridSearchCV, scoring=f1, cv=3",
        "",
        "Najlepsze parametry:",
        str(search.best_params_),
        "",
        "Najlepszy średni F1-score w walidacji krzyżowej:",
        f"{search.best_score_:.4f}",
        "",
        "Wyniki najlepszego modelu na zbiorze testowym:",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Precision: {metrics['precision']:.4f}",
        f"Recall: {metrics['recall']:.4f}",
        f"F1-score: {metrics['f1_score']:.4f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}",
        f"ROC-AUC: {metrics['roc_auc']:.4f}",
        "",
        "Wygenerowane pliki:",
        f"- {TUNING_RESULTS_PATH.name}",
        f"- {TUNED_MODEL_PATH.name}",
    ]

    return "\n".join(lines)


def run_hyperparameter_tuning() -> dict[str, float]:
    dataset = load_processed_dataset()
    x_train, x_test, y_train, y_test = split_features_and_labels(dataset)

    search = tune_model(x_train, y_train)
    save_tuning_results(search)
    save_tuned_artifact(search, x_test, y_test)

    predictions = search.best_estimator_.predict(x_test)
    scores = search.best_estimator_.predict_proba(x_test)[:, 1]
    metrics = compute_metrics(y_test, predictions, scores)

    report = build_tuning_report(search, metrics)
    TUNING_REPORT_PATH.write_text(report, encoding="utf-8")

    print("\nStrojenie hiperparametrów zakończone.")
    print(f"Najlepsze parametry: {search.best_params_}")
    print(f"F1-score na zbiorze testowym: {metrics['f1_score']:.4f}")
    print(f"Wyniki zapisano w: {EVALUATION_OUTPUT_DIR}")

    return metrics


def main() -> int:
    configure_logging()

    try:
        run_hyperparameter_tuning()
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Błąd strojenia hiperparametrów: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())