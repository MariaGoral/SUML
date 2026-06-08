"""Testy jednostkowe modułu data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import LABEL_COLUMN, TEXT_COLUMN
from data.data_loader import DataLoadError, DataValidationError, load_dataset
from data.preprocessing import (
    build_text_column,
    clean_text,
    preprocess_dataset,
)


def test_load_dataset_success(sample_raw_csv: Path) -> None:
    """Wczytuje poprawny plik CSV i zwraca oczekiwaną liczbę rekordów."""
    dataframe = load_dataset(sample_raw_csv)
    assert len(dataframe) == 2
    assert LABEL_COLUMN in dataframe.columns


def test_load_dataset_missing_file(tmp_path: Path) -> None:
    """Zgłasza DataLoadError, gdy plik nie istnieje."""
    with pytest.raises(DataLoadError):
        load_dataset(tmp_path / "missing.csv")


def test_load_dataset_invalid_labels(sample_raw_csv: Path) -> None:
    """Zgłasza DataValidationError przy nieprawidłowych etykietach."""
    dataframe = pd.read_csv(sample_raw_csv)
    dataframe[LABEL_COLUMN] = [0, 2]
    invalid_path = sample_raw_csv.parent / "invalid_labels.csv"
    dataframe.to_csv(invalid_path, index=False)

    with pytest.raises(DataValidationError):
        load_dataset(invalid_path)


def test_clean_text_normalization() -> None:
    """Czyści tekst: małe litery, bez znaków specjalnych, bez pustych stringów."""
    assert clean_text("  Marketing Intern!!! ") == "marketing intern"
    assert clean_text("@@@") == "missing"
    assert clean_text(None) == "missing"


def test_build_text_column(sample_dataframe: pd.DataFrame) -> None:
    """Tworzy kolumnę text z połączonych pól tekstowych."""
    from data.preprocessing import clean_text_columns

    cleaned = clean_text_columns(sample_dataframe)
    processed = build_text_column(cleaned)
    assert TEXT_COLUMN in processed.columns
    assert "hello world" in processed[TEXT_COLUMN].iloc[0]
    assert processed[TEXT_COLUMN].iloc[1] != ""


def test_preprocess_dataset(sample_dataframe: pd.DataFrame) -> None:
    """Preprocessing tworzy kolumnę text i eliminuje braki w pamięci."""
    processed = preprocess_dataset(sample_dataframe)
    assert TEXT_COLUMN in processed.columns
    assert processed.isnull().sum().sum() == 0
    assert processed[TEXT_COLUMN].str.len().min() > 0


def test_preprocess_csv_roundtrip_no_nan(
    sample_dataframe: pd.DataFrame,
    tmp_path: Path,
) -> None:
    """Po zapisie i ponownym wczytaniu CSV nie ma wartości NaN."""
    processed = preprocess_dataset(sample_dataframe)
    output_path = tmp_path / "cleaned.csv"
    processed.to_csv(output_path, index=False)

    reloaded = pd.read_csv(output_path)
    assert reloaded.isnull().sum().sum() == 0
    assert TEXT_COLUMN in reloaded.columns
