"""Główny punkt wejścia pipeline'u przygotowania danych."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from config import (
    PROCESSED_ANALYSIS_REPORT_PATH,
    PROCESSED_DATA_PATH,
    RAW_ANALYSIS_REPORT_PATH,
    RAW_DATA_PATH,
)
from data.data_loader import DataLoadError, DataValidationError, load_dataset
from data.exploratory_analysis import run_exploratory_analysis
from data.preprocessing import preprocess_dataset


def configure_logging(verbose: bool = False) -> None:
    """Konfiguruje logger aplikacji.

    Args:
        verbose: Czy wyświetlać komunikaty na poziomie DEBUG.
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def save_processed_dataset(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Zapisuje przetworzony dataset do pliku CSV.

    Args:
        dataframe: Przetworzony DataFrame.
        output_path: Ścieżka docelowa pliku CSV.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)
    logging.getLogger(__name__).info(
        "Zapisano przetworzone dane do pliku: %s", output_path
    )


def run_pipeline(
    raw_path: Path,
    raw_analysis_path: Path,
    processed_analysis_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """Uruchamia pełny pipeline przygotowania danych.

    Args:
        raw_path: Ścieżka do surowego pliku CSV.
        raw_analysis_path: Ścieżka raportu EDA dla danych surowych.
        processed_analysis_path: Ścieżka raportu EDA po preprocessingu.
        output_path: Ścieżka docelowa przetworzonego datasetu.

    Returns:
        Przetworzony DataFrame.

    Raises:
        DataLoadError: Gdy wczytanie danych się nie powiedzie.
        DataValidationError: Gdy walidacja danych się nie powiedzie.
    """
    logger = logging.getLogger(__name__)

    logger.info("Krok 1/5: Wczytywanie danych z %s", raw_path)
    raw_data = load_dataset(raw_path)

    logger.info("Krok 2/5: Analiza eksploracyjna danych surowych")
    run_exploratory_analysis(raw_data, raw_analysis_path, "Dane surowe")

    logger.info("Krok 3/5: Preprocessing i czyszczenie danych")
    cleaned_data = preprocess_dataset(raw_data)

    logger.info("Krok 4/5: Analiza eksploracyjna danych po preprocessingu")
    run_exploratory_analysis(
        cleaned_data,
        processed_analysis_path,
        "Dane po preprocessingu",
    )

    logger.info("Krok 5/5: Zapis przetworzonych danych")
    save_processed_dataset(cleaned_data, output_path)

    logger.info("Pipeline zakończony pomyślnie.")
    return cleaned_data


def parse_arguments() -> argparse.Namespace:
    """Parsuje argumenty wiersza poleceń.

    Returns:
        Namespace z przekazanymi argumentami.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline przygotowania danych — Fake Job Offer Detector",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_DATA_PATH,
        help=f"Ścieżka do surowego pliku CSV (domyślnie: {RAW_DATA_PATH})",
    )
    parser.add_argument(
        "--raw-analysis-output",
        type=Path,
        default=RAW_ANALYSIS_REPORT_PATH,
        help="Ścieżka do raportu EDA danych surowych",
    )
    parser.add_argument(
        "--processed-analysis-output",
        type=Path,
        default=PROCESSED_ANALYSIS_REPORT_PATH,
        help="Ścieżka do raportu EDA po preprocessingu",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROCESSED_DATA_PATH,
        help="Ścieżka do przetworzonego pliku CSV",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Włącza szczegółowe logowanie (DEBUG)",
    )
    return parser.parse_args()


def main() -> int:
    """Uruchamia pipeline przygotowania danych.

    Returns:
        Kod wyjścia: 0 przy sukcesie, 1 przy błędzie.
    """
    args = parse_arguments()
    configure_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        run_pipeline(
            raw_path=args.input,
            raw_analysis_path=args.raw_analysis_output,
            processed_analysis_path=args.processed_analysis_output,
            output_path=args.output,
        )
    except (DataLoadError, DataValidationError) as exc:
        logger.error("Błąd pipeline'u: %s", exc)
        return 1
    except OSError as exc:
        logger.error("Błąd operacji na pliku: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
