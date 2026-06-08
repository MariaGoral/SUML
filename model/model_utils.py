"""Wspólne funkcje wczytywania danych i obsługi artefaktów modelu."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypedDict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from config import LABEL_COLUMN, MODEL_PATH, PROCESSED_DATA_PATH, TEXT_COLUMN

logger = logging.getLogger(__name__)


class ModelArtifact(TypedDict):
    """Struktura zapisanego artefaktu modelu."""

    pipeline: Pipeline
    vectorizer: Any
    classifier: Any
    model_name: str
    X_test: pd.Series
    y_test: pd.Series


class PredictionResult(TypedDict):
    """Wynik predykcji dla pojedynczej oferty pracy."""

    prediction: int
    label: str
    confidence: float


class ModelLoadError(OSError):
    """Wyjątek zgłaszany, gdy nie można wczytać zapisanego modelu."""


class DataLoadError(OSError):
    """Wyjątek zgłaszany, gdy nie można wczytać danych treningowych."""


def load_processed_dataset(data_path: Path | None = None) -> pd.DataFrame:
    """Wczytuje przetworzony dataset z katalogu data/processed/.

    Args:
        data_path: Opcjonalna ścieżka do pliku CSV. Domyślnie cleaned_jobs.csv.

    Returns:
        DataFrame z kolumnami ``text`` i ``fraudulent``.

    Raises:
        DataLoadError: Gdy plik nie istnieje lub brakuje wymaganych kolumn.
    """
    path = data_path or PROCESSED_DATA_PATH

    if not path.exists():
        raise DataLoadError(f"Plik z danymi nie istnieje: {path}")

    try:
        dataframe = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        raise DataLoadError(f"Nie można wczytać pliku CSV: {path}") from exc

    if LABEL_COLUMN not in dataframe.columns:
        raise DataLoadError(
            f"Brak wymaganej kolumny etykiet: '{LABEL_COLUMN}'."
        )

    if TEXT_COLUMN not in dataframe.columns:
        raise DataLoadError(
            f"Brak wymaganej kolumny '{TEXT_COLUMN}'. "
            "Uruchom pipeline danych (data/main.py)."
        )

    logger.info("Wczytano %s rekordów z pliku %s.", len(dataframe), path)
    return dataframe


def save_model_artifact(artifact: ModelArtifact, model_path: Path | None = None) -> None:
    """Zapisuje wytrenowany model i vectorizer do pliku pickle.

    Args:
        artifact: Słownik z pipeline, vectorizerem, klasyfikatorem i danymi testowymi.
        model_path: Opcjonalna ścieżka docelowa pliku .pkl.
    """
    path = model_path or MODEL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path)
    logger.info("Zapisano model '%s' do pliku: %s", artifact["model_name"], path)


def load_model_artifact(model_path: Path | None = None) -> ModelArtifact:
    """Wczytuje zapisany artefakt modelu z pliku pickle.

    Args:
        model_path: Opcjonalna ścieżka do pliku .pkl.

    Returns:
        Słownik z pipeline, vectorizerem, klasyfikatorem i danymi testowymi.

    Raises:
        ModelLoadError: Gdy plik modelu nie istnieje lub jest uszkodzony.
    """
    path = model_path or MODEL_PATH

    if not path.exists():
        raise ModelLoadError(
            f"Plik modelu nie istnieje: {path}. Uruchom najpierw train.py."
        )

    try:
        artifact: ModelArtifact = joblib.load(path)
    except (OSError, ValueError, EOFError) as exc:
        raise ModelLoadError(f"Nie można wczytać modelu z pliku: {path}") from exc

    return artifact
