"""Skrypt ewaluacji wytrenowanego modelu klasyfikacji ofert pracy."""

from __future__ import annotations

import logging
import sys

from sklearn.metrics import classification_report, confusion_matrix

from model.metrics_utils import compute_metrics
from model.model_utils import ModelLoadError, load_model_artifact

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Konfiguruje logger skryptu ewaluacji."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def print_metrics(metrics: dict[str, float], model_name: str) -> None:
    """Wyświetla obliczone metryki w czytelnej formie.

    Args:
        metrics: Słownik metryk ewaluacji.
        model_name: Nazwa ewaluowanego modelu.
    """
    print(f"\nEwaluacja modelu: {model_name}")
    print("=" * 40)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1_score']:.4f}")


def print_confusion_matrix(y_true, y_pred) -> None:
    """Wyświetla macierz pomyłek dla klasyfikacji binarnej.

    Args:
        y_true: Prawdziwe etykiety.
        y_pred: Predykcje modelu.
    """
    matrix = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print("-" * 40)
    print("                 Predicted")
    print("                 Real  Fake")
    print(f"Actual Real      {matrix[0][0]:5d} {matrix[0][1]:5d}")
    print(f"       Fake      {matrix[1][0]:5d} {matrix[1][1]:5d}")
    print("-" * 40)


def run_evaluation() -> dict[str, float]:
    """Wczytuje model i przeprowadza ewaluację na zbiorze testowym.

    Returns:
        Słownik obliczonych metryk.

    Raises:
        ModelLoadError: Gdy nie można wczytać zapisanego modelu.
    """
    artifact = load_model_artifact()
    pipeline = artifact["pipeline"]
    x_test = artifact["X_test"]
    y_test = artifact["y_test"]

    predictions = pipeline.predict(x_test)
    metrics = compute_metrics(y_test, predictions)

    print_metrics(metrics, artifact["model_name"])
    print_confusion_matrix(y_test, predictions)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions, zero_division=0))

    logger.info("Ewaluacja zakończona pomyślnie.")
    return metrics


def main() -> int:
    """Punkt wejścia skryptu ewaluacji.

    Returns:
        Kod wyjścia: 0 przy sukcesie, 1 przy błędzie.
    """
    configure_logging()

    try:
        run_evaluation()
    except ModelLoadError as exc:
        logger.error("Błąd ewaluacji: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
