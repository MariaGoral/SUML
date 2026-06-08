"""Wspólne funkcje obliczania metryk klasyfikacji."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true, y_pred) -> dict[str, float]:
    """Oblicza metryki klasyfikacji binarnej dla klasy pozytywnej (fałszywa).

    Args:
        y_true: Prawdziwe etykiety.
        y_pred: Predykcje modelu.

    Returns:
        Słownik z accuracy, precision, recall i f1_score.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
    }
