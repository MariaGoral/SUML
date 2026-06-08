"""Moduł czyszczenia i przetwarzania danych ofert pracy."""

from __future__ import annotations

import logging
import re

import pandas as pd

from config import (
    LABEL_COLUMN,
    MISSING_TEXT_PLACEHOLDER,
    TEXT_COLUMN,
    TEXT_COLUMNS,
)

logger = logging.getLogger(__name__)

WHITESPACE_PATTERN: re.Pattern[str] = re.compile(r"\s+")
SPECIAL_CHARS_PATTERN: re.Pattern[str] = re.compile(r"[^a-z0-9\s]")


def remove_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Usuwa zduplikowane rekordy z datasetu.

    Args:
        dataframe: DataFrame z danymi wejściowymi.

    Returns:
        DataFrame bez duplikatów.
    """
    initial_count = len(dataframe)
    cleaned = dataframe.drop_duplicates().reset_index(drop=True)
    removed_count = initial_count - len(cleaned)

    if removed_count > 0:
        logger.info("Usunięto %s zduplikowanych rekordów.", removed_count)

    return cleaned


def handle_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Obsługuje brakujące wartości w kolumnach tekstowych i numerycznych.

    Args:
        dataframe: DataFrame z danymi wejściowymi.

    Returns:
        DataFrame z uzupełnionymi brakującymi wartościami.
    """
    processed = dataframe.copy()
    text_columns = get_text_columns(processed)

    for column in text_columns:
        processed[column] = processed[column].fillna(MISSING_TEXT_PLACEHOLDER)

    numeric_columns = processed.select_dtypes(include="number").columns.tolist()
    for column in numeric_columns:
        if column == LABEL_COLUMN:
            continue
        median_value = processed[column].median()
        processed[column] = processed[column].fillna(median_value)

    logger.info("Uzupełniono brakujące wartości w %s kolumnach.", len(text_columns))
    return processed


def normalize_whitespace(text: str) -> str:
    """Usuwa nadmiarowe spacje i białe znaki z tekstu.

    Args:
        text: Tekst do normalizacji.

    Returns:
        Tekst z pojedynczymi spacjami między słowami.
    """
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def remove_special_characters(text: str) -> str:
    """Usuwa znaki specjalne, pozostawiając litery, cyfry i spacje.

    Args:
        text: Tekst do oczyszczenia.

    Returns:
        Tekst bez znaków specjalnych.
    """
    return SPECIAL_CHARS_PATTERN.sub("", text)


def clean_text(text: str) -> str:
    """Wykonuje pełne czyszczenie pojedynczego pola tekstowego.

    Args:
        text: Surowy tekst do oczyszczenia.

    Returns:
        Oczyszczony tekst lub placeholder ``missing`` dla pustych wartości.
    """
    if pd.isna(text):
        return MISSING_TEXT_PLACEHOLDER

    normalized = str(text).lower()
    normalized = remove_special_characters(normalized)
    normalized = normalize_whitespace(normalized)

    return normalized if normalized else MISSING_TEXT_PLACEHOLDER


def get_text_columns(dataframe: pd.DataFrame) -> list[str]:
    """Zwraca listę kolumn tekstowych obecnych w datasetcie.

    Args:
        dataframe: DataFrame, z którego wybierane są kolumny.

    Returns:
        Lista nazw kolumn tekstowych dostępnych w DataFrame.
    """
    known_columns = [column for column in TEXT_COLUMNS if column in dataframe.columns]
    object_columns = dataframe.select_dtypes(include="object").columns.tolist()
    extra_columns = [
        column for column in object_columns
        if column not in known_columns and column != TEXT_COLUMN
    ]

    return known_columns + extra_columns


def clean_text_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Czyści wszystkie kolumny tekstowe w datasetcie.

    Args:
        dataframe: DataFrame z danymi wejściowymi.

    Returns:
        DataFrame z oczyszczonymi kolumnami tekstowymi.
    """
    processed = dataframe.copy()
    text_columns = get_text_columns(processed)

    for column in text_columns:
        processed[column] = processed[column].apply(clean_text)

    logger.info("Oczyszczono %s kolumn tekstowych.", len(text_columns))
    return processed


def build_text_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Tworzy kolumnę ``text`` z połączonych pól tekstowych oferty.

    Args:
        dataframe: DataFrame z oczyszczonymi polami tekstowymi.

    Returns:
        DataFrame z kolumną ``text`` gotową do modelowania.

    Raises:
        ValueError: Gdy brak dostępnych kolumn tekstowych do połączenia.
    """
    processed = dataframe.copy()
    source_columns = [
        column for column in TEXT_COLUMNS if column in processed.columns
    ]

    if not source_columns:
        raise ValueError("Brak kolumn tekstowych do utworzenia kolumny 'text'.")

    processed[TEXT_COLUMN] = (
        processed[source_columns]
        .astype(str)
        .agg(" ".join, axis=1)
        .str.strip()
        .replace("", MISSING_TEXT_PLACEHOLDER)
    )

    logger.info(
        "Utworzono kolumnę '%s' z %s pól tekstowych.",
        TEXT_COLUMN,
        len(source_columns),
    )
    return processed


def preprocess_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Uruchamia pełny pipeline preprocessingu danych.

    Args:
        dataframe: Surowy DataFrame po wczytaniu i walidacji.

    Returns:
        Przetworzony DataFrame gotowy do analizy i modelowania.
    """
    logger.info("Rozpoczęto preprocessing datasetu (%s rekordów).", len(dataframe))

    processed = remove_duplicates(dataframe)
    processed = handle_missing_values(processed)
    processed = clean_text_columns(processed)
    processed = build_text_column(processed)

    logger.info("Preprocessing zakończony. Liczba rekordów: %s.", len(processed))
    return processed
