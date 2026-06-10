from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix

from config import (
    CONFUSION_MATRIX_PLOT_PATH,
    ERROR_ANALYSIS_PATH,
    EVALUATION_OUTPUT_DIR,
    EVALUATION_REPORT_PATH,
    LABEL_COLUMN,
    MODEL_COMPARISON_PATH,
    MODEL_METRICS_PLOT_PATH,
    ROC_CURVE_PLOT_PATH,
)
from model.metrics_utils import compute_metrics
from model.model_utils import load_processed_dataset
from model.train import get_candidate_models, split_features_and_labels

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def get_positive_class_scores(pipeline, x_test: pd.Series) -> list[float]:
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(x_test)
        return [float(value) for value in probabilities[:, 1]]

    if hasattr(pipeline, "decision_function"):
        scores = pipeline.decision_function(x_test)
        min_score = float(min(scores))
        max_score = float(max(scores))

        if max_score == min_score:
            return [0.5 for _ in scores]

        return [float((score - min_score) / (max_score - min_score)) for score in scores]

    predictions = pipeline.predict(x_test)
    return [float(value) for value in predictions]


def evaluate_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, str, object, list[int], list[float]]:
    rows = []
    best_name = ""
    best_pipeline = None
    best_predictions = []
    best_scores = []
    best_f1 = -1.0

    for model_name, pipeline in get_candidate_models().items():
        logger.info("Trenowanie i ewaluacja modelu: %s", model_name)

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        scores = get_positive_class_scores(pipeline, x_test)
        metrics = compute_metrics(y_test, predictions, scores)

        rows.append({"model_name": model_name, **metrics})

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_name = model_name
            best_pipeline = pipeline
            best_predictions = [int(value) for value in predictions]
            best_scores = scores

    if best_pipeline is None:
        raise RuntimeError("Nie udało się wybrać najlepszego modelu.")

    results = pd.DataFrame(rows).sort_values("f1_score", ascending=False)
    return results, best_name, best_pipeline, best_predictions, best_scores


def save_model_comparison(results: pd.DataFrame) -> None:
    EVALUATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(MODEL_COMPARISON_PATH, index=False)


def plot_model_metrics(results: pd.DataFrame) -> None:
    metrics = ["accuracy", "precision", "recall", "f1_score", "balanced_accuracy"]
    plot_data = results.set_index("model_name")[metrics]

    ax = plot_data.plot(kind="bar", figsize=(11, 6))
    ax.set_title("Porównanie jakości modeli")
    ax.set_xlabel("Model")
    ax.set_ylabel("Wartość metryki")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Metryka")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(MODEL_METRICS_PLOT_PATH)
    plt.close()


def plot_confusion_matrix(y_test: pd.Series, predictions: list[int], model_name: str) -> None:
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Real", "Fake"],
    )

    display.plot(values_format="d")
    plt.title(f"Confusion matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PLOT_PATH)
    plt.close()


def plot_roc_curve(y_test: pd.Series, scores: list[float], model_name: str) -> None:
    RocCurveDisplay.from_predictions(y_test, scores, name=model_name)
    plt.title(f"ROC curve - {model_name}")
    plt.tight_layout()
    plt.savefig(ROC_CURVE_PLOT_PATH)
    plt.close()


def save_error_analysis(
    x_test: pd.Series,
    y_test: pd.Series,
    predictions: list[int],
    scores: list[float],
) -> pd.DataFrame:
    rows = []

    for text, true_label, predicted_label, fake_score in zip(
        x_test,
        y_test,
        predictions,
        scores,
    ):
        if int(true_label) != int(predicted_label):
            confidence = fake_score if int(predicted_label) == 1 else 1 - fake_score

            rows.append(
                {
                    "text": text,
                    "true_label": int(true_label),
                    "predicted_label": int(predicted_label),
                    "confidence": round(float(confidence), 4),
                }
            )

    error_dataframe = pd.DataFrame(rows)
    error_dataframe.to_csv(ERROR_ANALYSIS_PATH, index=False)
    return error_dataframe


def build_report(
    results: pd.DataFrame,
    best_name: str,
    error_dataframe: pd.DataFrame,
    dataset: pd.DataFrame,
) -> str:
    best_row = results.iloc[0]
    class_distribution = dataset[LABEL_COLUMN].value_counts().sort_index()
    real_count = int(class_distribution.get(0, 0))
    fake_count = int(class_distribution.get(1, 0))

    lines = [
        "Fake Job Offer Detector - raport ewaluacji modelu",
        "",
        "Cel etapu:",
        "Porównanie jakości modeli, przygotowanie metryk, wykresów, analizy błędów oraz wskazanie najlepszego modelu.",
        "",
        "Dane:",
        f"Liczba rekordów: {len(dataset)}",
        f"Prawdziwe oferty: {real_count}",
        f"Fałszywe oferty: {fake_count}",
        "",
        "Porównane modele:",
    ]

    for model_name in results["model_name"]:
        lines.append(f"- {model_name}")

    lines.extend(
        [
            "",
            "Wyniki:",
            results.to_string(index=False),
            "",
            "Najlepszy model:",
            f"{best_name}",
            f"Accuracy: {best_row['accuracy']:.4f}",
            f"Precision: {best_row['precision']:.4f}",
            f"Recall: {best_row['recall']:.4f}",
            f"F1-score: {best_row['f1_score']:.4f}",
            f"Balanced accuracy: {best_row['balanced_accuracy']:.4f}",
            f"ROC-AUC: {best_row['roc_auc']:.4f}",
            "",
            "Analiza błędów:",
            f"Liczba błędnych klasyfikacji w zbiorze testowym: {len(error_dataframe)}",
            "Szczegóły błędów zapisano w pliku error_analysis.csv.",
            "",
            "Wygenerowane pliki:",
            f"- {MODEL_COMPARISON_PATH.name}",
            f"- {ERROR_ANALYSIS_PATH.name}",
            f"- {MODEL_METRICS_PLOT_PATH.name}",
            f"- {CONFUSION_MATRIX_PLOT_PATH.name}",
            f"- {ROC_CURVE_PLOT_PATH.name}",
            "",
            "Wniosek:",
            "Najważniejszą metryką w projekcie jest F1-score, ponieważ wykrywana klasa fałszywych ofert jest rzadsza niż klasa prawdziwych ofert. Accuracy może być mylące przy niezbalansowanych danych.",
        ]
    )

    return "\n".join(lines)


def save_report(report: str) -> None:
    EVALUATION_REPORT_PATH.write_text(report, encoding="utf-8")


def run_extended_evaluation() -> pd.DataFrame:
    dataset = load_processed_dataset()
    x_train, x_test, y_train, y_test = split_features_and_labels(dataset)

    results, best_name, best_pipeline, predictions, scores = evaluate_models(
        x_train,
        x_test,
        y_train,
        y_test,
    )

    save_model_comparison(results)
    plot_model_metrics(results)
    plot_confusion_matrix(y_test, predictions, best_name)
    plot_roc_curve(y_test, scores, best_name)

    error_dataframe = save_error_analysis(x_test, y_test, predictions, scores)
    report = build_report(results, best_name, error_dataframe, dataset)
    save_report(report)

    print("\nEwaluacja zakończona.")
    print(f"Najlepszy model: {best_name}")
    print(f"Wyniki zapisano w: {EVALUATION_OUTPUT_DIR}")

    return results


def main() -> int:
    configure_logging()

    try:
        run_extended_evaluation()
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Błąd rozszerzonej ewaluacji: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())