"""Moduł predykcji dla pojedynczych ofert pracy."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from config import LABEL_NAMES, MISSING_TEXT_PLACEHOLDER
from data.preprocessing import clean_text
from model.model_utils import ModelLoadError, PredictionResult, load_model_artifact

logger = logging.getLogger(__name__)

_cached_artifact: Optional[dict] = None


def reset_model_cache() -> None:
    """Czyści cache wczytanego modelu (używane w testach)."""
    global _cached_artifact  # pylint: disable=global-statement
    _cached_artifact = None


def _get_artifact() -> dict:
    """Zwraca wczytany artefakt modelu (z cache w pamięci).

    Returns:
        Słownik artefaktu modelu.

    Raises:
        ModelLoadError: Gdy plik modelu nie istnieje.
    """
    global _cached_artifact  # pylint: disable=global-statement

    if _cached_artifact is None:
        _cached_artifact = load_model_artifact()

    return _cached_artifact


def preprocess_input_text(text: str) -> str:
    """Stosuje ten sam preprocessing tekstu co w pipeline danych.

    Args:
        text: Surowy tekst oferty pracy.

    Returns:
        Oczyszczony tekst gotowy do predykcji modelu.
    """
    return clean_text(text)


def predict_job_offer(text: str) -> PredictionResult:
    """Klasyfikuje pojedynczą ofertę pracy jako prawdziwą lub fałszywą.

    Args:
        text: Treść oferty pracy (surowy lub połączony tekst pól).

    Returns:
        Słownik z kluczami:
            - prediction: 0 (prawdziwa) lub 1 (fałszywa)
            - label: czytelna etykieta klasy
            - confidence: prawdopodobieństwo przypisanej klasy

    Raises:
        ModelLoadError: Gdy zapisany model nie jest dostępny.
        ValueError: Gdy przekazany tekst jest pusty po preprocessingu.
    """
    if not text or not str(text).strip():
        raise ValueError("Tekst oferty pracy nie może być pusty.")

    cleaned_text = preprocess_input_text(text)
    if cleaned_text == MISSING_TEXT_PLACEHOLDER:
        raise ValueError("Tekst oferty pracy nie może być pusty.")

    artifact = _get_artifact()
    pipeline = artifact["pipeline"]

    prediction = int(pipeline.predict([cleaned_text])[0])
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    confidence = float(probabilities[prediction])

    result = PredictionResult(
        prediction=prediction,
        label=LABEL_NAMES[prediction],
        confidence=round(confidence, 4),
    )

    logger.info(
        "Predykcja: %s (confidence=%.4f)",
        result["label"],
        result["confidence"],
    )
    return result


def main() -> int:
    """Demonstracyjny punkt wejścia — predykcja przykładowej oferty.

    Returns:
        Kod wyjścia: 0 przy sukcesie, 1 przy błędzie.
    """
    logging.basicConfig(level=logging.INFO)

    sample_text = (
        "Marketing Intern — US, NY, New York. "
        "Experience with content management systems required."
    )

    try:
        result = predict_job_offer(sample_text)
    except (ModelLoadError, ValueError) as exc:
        logger.error("Błąd predykcji: %s", exc)
        return 1

    print("Przykładowa predykcja:")
    print(f"  Prediction:  {result['prediction']}")
    print(f"  Label:       {result['label']}")
    print(f"  Confidence:  {result['confidence']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
