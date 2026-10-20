# Exploration of Experimental Data

Repozytorium na własne rozwiązania list laboratoryjnych. Każde laboratorium ma osobny
katalog, a wszystkie korzystają ze wspólnego środowiska Python 3.12 zarządzanego przez
[uv](https://docs.astral.sh/uv/).

## Pierwsze uruchomienie

Po sklonowaniu repozytorium uruchom w jego katalogu:

```bash
uv sync --locked
code .
```

`uv` pobierze odpowiedniego Pythona, utworzy `.venv` i zainstaluje biblioteki zgodnie
z `uv.lock`. Instrukcja instalacji `uv`: [oficjalna dokumentacja](https://docs.astral.sh/uv/getting-started/installation/).

W VS Code:

1. Zainstaluj rozszerzenia z rekomendacji projektu, jeśli edytor o nie poprosi.
2. Uruchom **Python: Select Interpreter** i wybierz środowisko `.venv`.
3. Otwórz `labs/lab01/solution.ipynb`.
4. W **Select Kernel → Python Environments** wybierz Pythona z tego samego `.venv`.

Notebook ma przygotowane miejsca na kod, wykresy i wnioski. Rozwiązania należy uzupełnić
samodzielnie. Można też pisać zwykłe pliki `.py` w katalogu danego laboratorium:

```bash
uv run python labs/lab01/solution.py
```

Powyższe polecenie zakłada, że wcześniej utworzysz `solution.py`.

## Układ katalogów

```text
experimental-data/
├── .vscode/                 # Interpreter, formatowanie i rozszerzenia VS Code
├── labs/
│   ├── lab01/
│   │   ├── README.md        # Informacje o laboratorium
│   │   ├── solution.ipynb   # Notebook na rozwiązania
│   │   ├── data/            # Dane wejściowe, np. CSV
│   │   ├── materials/       # Lokalne listy i materiały prowadzącego
│   │   └── outputs/         # Wykresy, eksporty i sprawozdanie
│   └── _template/           # Szablon kolejnego laboratorium
├── scripts/
│   └── new_lab.py           # Tworzenie następnego laboratorium
├── .python-version
├── pyproject.toml           # Biblioteki i ustawienia narzędzi
└── uv.lock                  # Dokładne wersje zależności
```

Pierwsze laboratorium: [regresja liniowa i kwartet Anscombe'a](labs/lab01/README.md).

## Kolejne laboratorium

```bash
uv run python scripts/new_lab.py 2 --title "Temat drugiego laboratorium"
```

Powstanie `labs/lab02/` z notebookiem, opisem oraz katalogami `data`, `materials`
i `outputs`. Istniejący katalog nie zostanie nadpisany. Numer i tytuł zostaną wpisane
do notebooka i opisu. Każda następna lista korzysta z tego samego środowiska `.venv`.

## Biblioteki i codzienna praca

Zainstalowane biblioteki: NumPy, pandas, SciPy, Matplotlib, seaborn, scikit-learn
i openpyxl. Narzędzia deweloperskie obejmują ipykernel, JupyterLab i Ruff.

```bash
# Dodanie biblioteki do projektu i aktualizacja uv.lock
uv add nazwa-biblioteki

# Sprawdzenie kodu oraz notebooków
uv run ruff check .

# Formatowanie kodu oraz notebooków
uv run ruff format .

# Opcjonalnie: praca w JupyterLab zamiast w VS Code
uv run jupyter lab
```

Polecenia uruchamiaj z głównego katalogu repozytorium. `uv run` korzysta ze środowiska
projektu, więc ręczna aktywacja `.venv` nie jest konieczna.

Notebook ustala katalog główny projektu i definiuje `DATA_DIR` oraz `OUTPUT_DIR`.
Przykładowo dane można później wczytać przez `pd.read_csv(DATA_DIR / "dane.csv")`,
a wykres zapisać przez `plt.savefig(OUTPUT_DIR / "wykres.png")`.

## Git i GitHub

Do repozytorium trafiają kod, notebooki, małe dane potrzebne do odtworzenia analiz
i wybrane wyniki. `.venv`, cache, pliki `.env` oraz zawartość katalogów `materials/`
są pomijane. Lokalny PDF z pierwszą listą znajduje się w `labs/lab01/materials/`.

Przed wysłaniem zmian przejrzyj pliki i wyniki komórek notebooka:

```bash
git status
git diff
git add labs/lab01
git commit -m "Add lab 01 solutions"
git push
```

Jeżeli dodajesz biblioteki, do tego samego commita dołącz `pyproject.toml` i `uv.lock`.

Szczegóły pracy z notebookami: [uv i Jupyter w VS Code](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code).
