"""Moduł wczytywania i walidacji datasetu ofert pracy."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import LABEL_COLUMN, REQUIRED_COLUMNS, VALID_LABEL_VALUES

logger = logging.getLogger(__name__)


class DataLoadError(OSError):
    """Wyjątek zgłaszany, gdy nie można wczytać pliku z danymi."""


class DataValidationError(ValueError):
    """Wyjątek zgłaszany, gdy dane nie przechodzą walidacji."""


def load_dataset(file_path: str | Path) -> pd.DataFrame:
    """Wczytuje dataset ofert pracy z pliku CSV.

    Args:
        file_path: Ścieżka do pliku CSV z danymi.

    Returns:
        DataFrame zawierający wczytane dane.

    Raises:
        DataLoadError: Gdy plik nie istnieje lub nie można go odczytać.
        DataValidationError: Gdy dane nie spełniają wymagań walidacji.
    """
    path = Path(file_path)

    if not path.exists():
        raise DataLoadError(f"Plik datasetu nie istnieje: {path}")

    if not path.is_file():
        raise DataLoadError(f"Podana ścieżka nie wskazuje na plik: {path}")

    if path.suffix.lower() != ".csv":
        raise DataLoadError(
            f"Oczekiwano pliku CSV, otrzymano rozszerzenie: {path.suffix}"
        )

    try:
        dataframe = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadError(f"Plik CSV jest pusty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadError(f"Nie można sparsować pliku CSV: {path}") from exc
    except OSError as exc:
        raise DataLoadError(f"Błąd odczytu pliku: {path}") from exc

    logger.info("Wczytano %s rekordów z pliku %s", len(dataframe), path)
    validate_dataset(dataframe)
    return dataframe


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Wykonuje podstawową walidację wczytanego datasetu.

    Args:
        dataframe: DataFrame do walidacji.

    Raises:
        DataValidationError: Gdy dane nie spełniają wymagań jakościowych.
    """
    if dataframe.empty:
        raise DataValidationError("Dataset jest pusty — brak rekordów do analizy.")

    missing_columns = REQUIRED_COLUMNS - set(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise DataValidationError(
            f"Brak wymaganych kolumn w datasetcie: {missing}"
        )

    if dataframe[LABEL_COLUMN].isnull().any():
        null_count = int(dataframe[LABEL_COLUMN].isnull().sum())
        raise DataValidationError(
            f"Kolumna '{LABEL_COLUMN}' zawiera {null_count} brakujących wartości."
        )

    unique_labels = set(dataframe[LABEL_COLUMN].unique())
    invalid_labels = unique_labels - VALID_LABEL_VALUES
    if invalid_labels:
        raise DataValidationError(
            "Kolumna etykiet zawiera nieprawidłowe wartości: "
            f"{sorted(invalid_labels)}. Oczekiwano: 0 (prawdziwa) lub 1 (fałszywa)."
        )

    logger.info("Walidacja datasetu zakończona pomyślnie.")
