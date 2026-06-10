"""Wspólne funkcje obliczania metryk klasyfikacji."""

from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, y_score=None) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0, pos_label=1)
        ),
        "recall": float(recall_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0, pos_label=1)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if y_score is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["roc_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0

    return metrics
