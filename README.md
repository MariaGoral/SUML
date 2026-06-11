# 🕵️ Fake Job Offer Detector

Fake Job Offer Detector to aplikacja wykorzystująca algorytmy Machine Learning do wykrywania fałszywych ofert pracy. Celem projektu jest analiza treści ogłoszenia o pracę oraz określenie, czy dana oferta jest prawdopodobnie prawdziwa, czy może stanowić próbę oszustwa.

System wykorzystuje techniki przetwarzania języka naturalnego (NLP) oraz modele klasyfikacyjne do analizy treści ofert pracy. Użytkownik może wkleić treść ogłoszenia, a aplikacja zwraca przewidywaną klasę oferty wraz z poziomem pewności predykcji.

Projekt został przygotowany zgodnie z wymaganiami przedmiotu i podzielony na następujące moduły:

- data/ – przygotowanie i przetwarzanie danych,
- model/ – trenowanie, ewaluacja i wykorzystanie modeli ML,
- app/ – aplikacja Streamlit udostępniająca model użytkownikowi.

---

## Funkcjonalności aplikacji

Aplikacja umożliwia:

- analizę treści ofert pracy,
- wykrywanie potencjalnie fałszywych ogłoszeń,
- klasyfikację ofert jako prawdziwe lub fałszywe,
- wyświetlanie poziomu pewności predykcji modelu,
- korzystanie z nowoczesnego interfejsu użytkownika,
- analizę pojedynczej oferty poprzez wklejenie jej treści do formularza.

---

## Technologie

W projekcie wykorzystano:

- Python,
- Streamlit,
- Pandas,
- NumPy,
- Scikit-learn,
- Joblib,
- TF-IDF Vectorizer.

Dodatkowo zastosowano:

- klasyfikację tekstu opartą na Machine Learning,
- techniki NLP do przetwarzania ofert pracy,
- model wykorzystujący reprezentację tekstu TF-IDF.

---

## Wykorzystany zbiór danych

Projekt wykorzystuje zbiór danych:

### Real / Fake Job Posting Prediction Dataset

Źródło:

https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction

Zbiór zawiera około 18 000 ofert pracy wraz z informacją określającą, czy dana oferta była prawdziwa czy fałszywa.

Wykorzystane informacje obejmują m.in.:

- tytuł stanowiska,
- lokalizację,
- dział firmy,
- opis stanowiska,
- wymagania,
- benefity,
- typ zatrudnienia,
- informację o autentyczności oferty.

---

## Opis działania aplikacji

### 1. Wprowadzenie treści oferty

Użytkownik wkleja treść ogłoszenia o pracę do pola tekstowego znajdującego się w aplikacji.

### 2. Analiza tekstu

Treść oferty zostaje poddana procesowi czyszczenia i przygotowania danych zgodnie z procedurą wykorzystaną podczas trenowania modelu.

Następnie tekst jest przekształcany do reprezentacji TF-IDF i przekazywany do wytrenowanego modelu klasyfikacyjnego.

### 3. Predykcja

Model określa prawdopodobieństwo przynależności oferty do jednej z klas:

- Real Job Offer,
- Fake Job Offer.

Wynik wraz z poziomem pewności prezentowany jest użytkownikowi w aplikacji.

---

## Interfejs użytkownika

W aplikacji zastosowano:

- własne style CSS,
- ciemny motyw kolorystyczny,
- zmodyfikowany wygląd przycisków,
- komunikaty o błędach i sukcesie,
- animację ładowania podczas wykonywania predykcji,
- wskaźnik prawdopodobieństwa klasyfikacji,
- responsywny układ oparty na Streamlit.

---

## Struktura projektu

```text
SUML/
│
├── app/
│   └── app.py
│
├── data/
│   ├── data_loader.py
│   ├── exploratory_analysis.py
│   ├── preprocessing.py
│   └── main.pkl
│
├── model/
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── model_utils.py
│   └── saved_model.pkl
│
├── model/
│   ├── conftest.py
│   ├── test_data.py
│   └── test_model.pkl
|
├── config.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Instalacja i uruchomienie lokalne

### 1. Sklonuj repozytorium

```bash
git clone https://github.com/MariaGoral/SUML.git
```

### 2. Przejdź do katalogu projektu

```bash
cd SUML
```

### 3. Zainstaluj wymagane biblioteki

```bash
pip install -r requirements.txt
```

### 4. Uruchom aplikację

```bash
python -m streamlit run app/app.py
```

Po uruchomieniu aplikacja będzie dostępna pod adresem:

```text
http://localhost:8501
```

---
