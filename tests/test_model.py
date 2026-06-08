"""Testy jednostkowe modułu model."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import MODEL_PATH
from model.model_utils import ModelArtifact, ModelLoadError, load_model_artifact
from model.predict import predict_job_offer, reset_model_cache


@pytest.fixture
def dummy_artifact(tmp_path: Path) -> Path:
    """Tworzy minimalny artefakt modelu do testów wczytywania."""
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("classifier", LogisticRegression(max_iter=200)),
        ]
    )
    texts = pd.Series(
        [
            "marketing intern new york",
            "work from home earn money fast",
            "software engineer python django",
            "real company full time job",
        ]
    )
    labels = pd.Series([0, 1, 0, 0])
    pipeline.fit(texts, labels)

    artifact: ModelArtifact = {
        "pipeline": pipeline,
        "vectorizer": pipeline.named_steps["tfidf"],
        "classifier": pipeline.named_steps["classifier"],
        "model_name": "Logistic Regression",
        "X_test": texts.iloc[:2],
        "y_test": labels.iloc[:2],
    }

    model_path = tmp_path / "saved_model.pkl"
    joblib.dump(artifact, model_path)
    return model_path


def test_load_model_artifact_success(dummy_artifact: Path) -> None:
    """Wczytuje zapisany artefakt modelu."""
    artifact = load_model_artifact(dummy_artifact)
    assert isinstance(artifact["pipeline"], Pipeline)
    assert artifact["model_name"] == "Logistic Regression"
    assert "vectorizer" in artifact
    assert "classifier" in artifact


def test_load_model_artifact_missing_file(tmp_path: Path) -> None:
    """Zgłasza ModelLoadError, gdy plik modelu nie istnieje."""
    with pytest.raises(ModelLoadError):
        load_model_artifact(tmp_path / "missing.pkl")


def test_predict_job_offer(dummy_artifact: Path) -> None:
    """Zwraca poprawną strukturę wyniku predykcji."""
    reset_model_cache()
    import model.predict as predict_module

    predict_module._cached_artifact = load_model_artifact(dummy_artifact)

    result = predict_job_offer("Marketing Intern in New York — full time role.")
    assert result["prediction"] in (0, 1)
    assert result["label"] in ("Real Job Offer", "Fake Job Offer")
    assert 0.0 <= result["confidence"] <= 1.0

    reset_model_cache()


def test_predict_job_offer_empty_text(dummy_artifact: Path) -> None:
    """Zgłasza ValueError dla pustego tekstu."""
    reset_model_cache()
    import model.predict as predict_module

    predict_module._cached_artifact = load_model_artifact(dummy_artifact)

    with pytest.raises(ValueError):
        predict_job_offer("   ")

    reset_model_cache()


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="Brak wytrenowanego modelu produkcyjnego.")
def test_production_model_loads() -> None:
    """Sprawdza, czy produkcyjny model wczytuje się poprawnie."""
    artifact = load_model_artifact(MODEL_PATH)
    assert artifact["model_name"]
    assert "pipeline" in artifact
