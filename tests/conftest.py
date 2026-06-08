"""Fixtures współdzielone przez testy projektu."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from config import LABEL_COLUMN, TEXT_COLUMN


@pytest.fixture
def sample_raw_csv(tmp_path: Path) -> Path:
    """Tworzy minimalny plik CSV z danymi testowymi."""
    dataframe = pd.DataFrame(
        {
            "job_id": [1, 2],
            "title": ["Marketing Intern", "Work From Home!!!"],
            "location": ["US, NY, New York", None],
            "department": ["Marketing", "Sales"],
            "salary_range": [None, "0-0"],
            "company_profile": ["Great company", "Earn $$$ fast"],
            "description": ["Job description here", "No experience needed"],
            "requirements": ["Degree required", None],
            "benefits": [None, "Flexible hours"],
            "telecommuting": [0, 1],
            "has_company_logo": [1, 0],
            "has_questions": [0, 1],
            "employment_type": ["Full-time", "Other"],
            "required_experience": ["Internship", "Not Applicable"],
            "required_education": ["Bachelor's Degree", None],
            "industry": ["Marketing", "Other"],
            "function": ["Marketing", "Other"],
            LABEL_COLUMN: [0, 1],
        }
    )
    file_path = tmp_path / "sample_jobs.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Zwraca minimalny DataFrame do testów preprocessingu."""
    return pd.DataFrame(
        {
            "title": ["Hello World!", "   "],
            "description": ["Job @ Company #1", None],
            "location": ["US, NY", "GB, LND"],
            LABEL_COLUMN: [0, 1],
        }
    )
