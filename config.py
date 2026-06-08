"""Wspólna konfiguracja projektu Fake Job Offer Detector."""

from __future__ import annotations

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent

# --- Kolumny danych ---
TEXT_COLUMN: Final[str] = "text"
LABEL_COLUMN: Final[str] = "fraudulent"
MISSING_TEXT_PLACEHOLDER: Final[str] = "missing"
VALID_LABEL_VALUES: Final[frozenset[int]] = frozenset({0, 1})
REQUIRED_COLUMNS: Final[frozenset[str]] = frozenset({LABEL_COLUMN})

TEXT_COLUMNS: Final[list[str]] = [
    "title",
    "location",
    "department",
    "salary_range",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]

LABEL_NAMES: Final[dict[int, str]] = {
    0: "Real Job Offer",
    1: "Fake Job Offer",
}

# --- Ścieżki plików ---
RAW_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "fake_job_postings.csv"
PROCESSED_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_PATH: Final[Path] = PROCESSED_DIR / "cleaned_jobs.csv"
RAW_ANALYSIS_REPORT_PATH: Final[Path] = PROCESSED_DIR / "analysis_report_raw.txt"
PROCESSED_ANALYSIS_REPORT_PATH: Final[Path] = PROCESSED_DIR / "analysis_report_processed.txt"
MODEL_PATH: Final[Path] = PROJECT_ROOT / "model" / "saved_model.pkl"

# --- Parametry ML ---
RANDOM_STATE: Final[int] = 42
TEST_SIZE: Final[float] = 0.2
TFIDF_MAX_FEATURES: Final[int] = 10000
TFIDF_NGRAM_RANGE: Final[tuple[int, int]] = (1, 2)
