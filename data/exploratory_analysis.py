"""Moduł eksploracyjnej analizy danych (EDA) datasetu ofert pracy."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config import LABEL_COLUMN

logger = logging.getLogger(__name__)


def compute_class_distribution(dataframe: pd.DataFrame) -> dict[str, float | int]:
    """Oblicza rozkład klas w datasetcie.

    Args:
        dataframe: DataFrame z kolumną etykiet.

    Returns:
        Słownik ze statystykami rozkładu klas.
    """
    total_records = len(dataframe)
    real_count = int((dataframe[LABEL_COLUMN] == 0).sum())
    fake_count = int((dataframe[LABEL_COLUMN] == 1).sum())

    return {
        "total_records": total_records,
        "real_offers": real_count,
        "fake_offers": fake_count,
        "real_percentage": round(real_count / total_records * 100, 2),
        "fake_percentage": round(fake_count / total_records * 100, 2),
    }


def generate_analysis_report(dataframe: pd.DataFrame, dataset_label: str) -> str:
    """Generuje raport tekstowy z podstawową analizą datasetu.

    Args:
        dataframe: DataFrame do analizy.
        dataset_label: Opis etapu danych (np. surowe / po preprocessingu).

    Returns:
        Sformatowany raport analizy w formie tekstu.
    """
    distribution = compute_class_distribution(dataframe)
    missing_values = dataframe.isnull().sum()
    missing_summary = missing_values[missing_values > 0]
    total_missing = int(missing_values.sum())

    lines = [
        "=" * 60,
        "RAPORT EKSPLORACYJNEJ ANALIZY DANYCH",
        "Fake Job Offer Detector",
        f"Etap danych: {dataset_label}",
        "=" * 60,
        "",
        "PODSTAWOWE STATYSTYKI",
        "-" * 40,
        f"Liczba rekordów: {distribution['total_records']}",
        f"Liczba kolumn: {dataframe.shape[1]}",
        f"Łączna liczba braków: {total_missing}",
        "",
        "ROZKŁAD KLAS",
        "-" * 40,
        f"Prawdziwe oferty (0): {distribution['real_offers']} "
        f"({distribution['real_percentage']}%)",
        f"Fałszywe oferty (1): {distribution['fake_offers']} "
        f"({distribution['fake_percentage']}%)",
        "",
        "TYPY DANYCH",
        "-" * 40,
        dataframe.dtypes.to_string(),
        "",
        "STATYSTYKI OPISOWE (KOLUMNY NUMERYCZNE)",
        "-" * 40,
        dataframe.describe(include="number").to_string(),
        "",
        "STATYSTYKI OPISOWE (KOLUMNY TEKSTOWE)",
        "-" * 40,
        dataframe.describe(include="object").to_string(),
        "",
        "BRAKUJĄCE WARTOŚCI",
        "-" * 40,
    ]

    if missing_summary.empty:
        lines.append("Brak brakujących wartości w datasetcie.")
    else:
        lines.append(missing_summary.to_string())

    lines.extend(["", "=" * 60])
    return "\n".join(lines)


def save_analysis_report(report: str, output_path: str | Path) -> None:
    """Zapisuje raport analizy do pliku tekstowego.

    Args:
        report: Treść raportu do zapisania.
        output_path: Ścieżka docelowa pliku raportu.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    logger.info("Zapisano raport analizy do pliku: %s", path)


def run_exploratory_analysis(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    dataset_label: str,
) -> dict[str, float | int]:
    """Wykonuje analizę eksploracyjną i zapisuje wyniki do pliku.

    Args:
        dataframe: DataFrame do analizy.
        output_path: Ścieżka docelowa pliku raportu.
        dataset_label: Opis etapu danych.

    Returns:
        Słownik ze statystykami rozkładu klas.
    """
    distribution = compute_class_distribution(dataframe)
    report = generate_analysis_report(dataframe, dataset_label)

    print(report)
    save_analysis_report(report, output_path)

    logger.info(
        "Analiza (%s) zakończona: %s rekordów, %s fałszywych, %s prawdziwych.",
        dataset_label,
        distribution["total_records"],
        distribution["fake_offers"],
        distribution["real_offers"],
    )
    return distribution
