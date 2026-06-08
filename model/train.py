"""Skrypt trenowania i porównania modeli klasyfikacji ofert pracy."""

from __future__ import annotations

import logging
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import (
    LABEL_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    TEXT_COLUMN,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
)
from model.model_utils import ModelArtifact, load_processed_dataset, save_model_artifact

logger = logging.getLogger(__name__)


def configure_logging() -> None:
    """Konfiguruje logger skryptu trenowania."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def create_tfidf_pipeline(classifier) -> Pipeline:
    """Tworzy pipeline Scikit-Learn z TF-IDF i klasyfikatorem.

    TF-IDF (Term Frequency-Inverse Document Frequency) zamienia tekst oferty
    na wektor liczbowy, uwzględniając jak często dane słowo występuje w
    dokumencie oraz jak rzadkie jest w całym korpusie. Dzięki temu słowa
    charakterystyczne dla fałszywych ofert otrzymują wyższą wagę niż
    powszechnie występujące słowa, co poprawia jakość klasyfikacji tekstu.

    Args:
        classifier: Estymator Scikit-Learn do klasyfikacji binarnej.

    Returns:
        Pipeline łączący TfidfVectorizer z przekazanym klasyfikatorem.
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=TFIDF_MAX_FEATURES,
                    ngram_range=TFIDF_NGRAM_RANGE,
                ),
            ),
            ("classifier", classifier),
        ]
    )


def get_candidate_models() -> dict[str, Pipeline]:
    """Zwraca słownik kandydatów modeli do porównania.

    Porównujemy trzy różne algorytmy, ponieważ każdy inaczej modeluje
    zależności w danych tekstowych: regresja logistyczna dobrze sprawdza
    się przy liniowych granicach decyzyjnych, Naive Bayes jest szybki
    i skuteczny przy danych tekstowych, a Random Forest łapie nieliniowe
    wzorce bez ręcznego inżynierii cech.

    Returns:
        Słownik nazwa modelu -> pipeline gotowy do trenowania.
    """
    return {
        "Logistic Regression": create_tfidf_pipeline(
            LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
        ),
        "Multinomial Naive Bayes": create_tfidf_pipeline(MultinomialNB()),
        "Random Forest": create_tfidf_pipeline(
            RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


def split_features_and_labels(
    dataframe: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Dzieli dataset na zbiory treningowy i testowy w proporcji 80/20.

    Args:
        dataframe: DataFrame z kolumnami ``text`` i ``fraudulent``.

    Returns:
        Krotka (X_train, X_test, y_train, y_test).
    """
    features = dataframe[TEXT_COLUMN]
    labels = dataframe[LABEL_COLUMN]

    return train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )


def evaluate_model(
    pipeline: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> float:
    """Oblicza F1-score modelu na zbiorze testowym.

    F1-score jest główną metryką, ponieważ dataset jest niezbalansowany
    (znacznie więcej prawdziwych niż fałszywych ofert). Accuracy mogłoby
    być mylące — model zawsze przewidujący klasę większościową uzyskałby
    wysoką dokładność, ale słabo wykrywałby oszustwa. F1 łączy precision
    i recall, co jest kluczowe przy wykrywaniu rzadkiej klasy fałszywych ofert.

    Args:
        pipeline: Wytrenowany pipeline modelu.
        x_test: Cechy tekstowe zbioru testowego.
        y_test: Etykiety zbioru testowego.

    Returns:
        Wartość F1-score na zbiorze testowym.
    """
    predictions = pipeline.predict(x_test)
    return float(f1_score(y_test, predictions))


def train_and_compare_models(
    x_train: pd.Series,
    x_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[str, Pipeline, dict[str, float]]:
    """Trenuje i porównuje wszystkie modele kandydatów.

    Args:
        x_train: Teksty treningowe.
        x_test: Teksty testowe.
        y_train: Etykiety treningowe.
        y_test: Etykiety testowe.

    Returns:
        Krotka (nazwa najlepszego modelu, pipeline, wyniki F1 wszystkich modeli).
    """
    results: dict[str, float] = {}
    best_name = ""
    best_pipeline: Pipeline | None = None
    best_f1 = -1.0

    for name, pipeline in get_candidate_models().items():
        logger.info("Trenowanie modelu: %s", name)
        pipeline.fit(x_train, y_train)
        f1_value = evaluate_model(pipeline, x_test, y_test)
        results[name] = f1_value
        logger.info("Model '%s' — F1-score: %.4f", name, f1_value)

        if f1_value > best_f1:
            best_f1 = f1_value
            best_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Nie udało się wytrenować żadnego modelu.")

    logger.info("Najlepszy model: %s (F1-score: %.4f)", best_name, best_f1)
    return best_name, best_pipeline, results


def build_model_artifact(
    model_name: str,
    pipeline: Pipeline,
    x_test: pd.Series,
    y_test: pd.Series,
) -> ModelArtifact:
    """Buduje artefakt do zapisu na dysku.

    Args:
        model_name: Nazwa wybranego modelu.
        pipeline: Wytrenowany pipeline zawierający vectorizer i klasyfikator.
        x_test: Teksty zbioru testowego.
        y_test: Etykiety zbioru testowego.

    Returns:
        Słownik artefaktu gotowy do serializacji joblib.
    """
    return ModelArtifact(
        pipeline=pipeline,
        vectorizer=pipeline.named_steps["tfidf"],
        classifier=pipeline.named_steps["classifier"],
        model_name=model_name,
        X_test=x_test,
        y_test=y_test,
    )


def print_comparison_table(results: dict[str, float]) -> None:
    """Wyświetla tabelę porównawczą F1-score wszystkich modeli.

    Args:
        results: Słownik nazwa modelu -> F1-score.
    """
    print("\nPorównanie modeli (F1-score na zbiorze testowym):")
    print("-" * 50)
    for name, score in sorted(results.items(), key=lambda item: item[1], reverse=True):
        print(f"  {name:<25} {score:.4f}")
    print("-" * 50)


def run_training() -> ModelArtifact:
    """Uruchamia pełny proces trenowania i zapisu najlepszego modelu.

    Returns:
        Zapisany artefakt najlepszego modelu.
    """
    dataframe = load_processed_dataset()
    x_train, x_test, y_train, y_test = split_features_and_labels(dataframe)

    logger.info(
        "Podział danych: train=%s, test=%s",
        len(x_train),
        len(x_test),
    )

    best_name, best_pipeline, results = train_and_compare_models(
        x_train, x_test, y_train, y_test
    )
    print_comparison_table(results)

    artifact = build_model_artifact(best_name, best_pipeline, x_test, y_test)
    save_model_artifact(artifact)

    print(f"\nZapisano najlepszy model: {best_name}")
    return artifact


def main() -> int:
    """Punkt wejścia skryptu trenowania.

    Returns:
        Kod wyjścia: 0 przy sukcesie, 1 przy błędzie.
    """
    configure_logging()

    try:
        run_training()
    except (OSError, ValueError, RuntimeError) as exc:
        logger.error("Błąd trenowania: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
