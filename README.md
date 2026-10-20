# Exploration of Experimental Data

Personal solutions to lab assignments. Each lab has its own directory, and all labs
share a Python 3.12 environment managed with
[uv](https://docs.astral.sh/uv/).

## Getting started

After cloning the repository, run these commands from its root directory:

```bash
uv sync --locked
code .
```

`uv` downloads the required Python version, creates `.venv`, and installs the dependencies
specified in `uv.lock`. To install `uv`, see the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

In VS Code:

1. Install the recommended extensions if prompted.
2. Run **Python: Select Interpreter** and select the `.venv` environment.
3. Open `labs/lab01/solution.ipynb`.
4. Under **Select Kernel → Python Environments**, select Python from the same `.venv`.

The notebook provides space for code, plots, and conclusions. Complete the solutions
yourself. You can also write regular `.py` files in each lab directory:

```bash
uv run python labs/lab01/solution.py
```

Create `solution.py` before running the command above.

## Project structure

```text
experimental-data/
├── .vscode/                 # VS Code interpreter, formatting, and extensions
├── labs/
│   ├── lab01/
│   │   ├── README.md        # Lab notes and requirements
│   │   ├── solution.ipynb   # Notebook for solutions
│   │   ├── data/            # Input data, such as CSV files
│   │   ├── materials/       # Local assignment sheets and course materials
│   │   └── outputs/         # Plots, exports, and the lab report
│   └── _template/           # Template for future labs
├── scripts/
│   └── new_lab.py           # Creates a new lab directory
├── .python-version
├── pyproject.toml           # Dependencies and tool settings
└── uv.lock                  # Exact dependency versions
```

First lab: [linear regression and Anscombe's quartet](labs/lab01/README.md).

## Adding a lab

```bash
uv run python scripts/new_lab.py 2 --title "Second lab topic"
```

This creates `labs/lab02/` with a notebook, a README, and the `data`, `materials`, and
`outputs` directories. Existing lab directories are never overwritten. The lab number
and title are added to the notebook and README. All labs use the same `.venv` environment.
Use an English title, as all README files in this repository are written in English.

## Libraries and everyday commands

Installed libraries include NumPy, pandas, SciPy, Matplotlib, seaborn, scikit-learn,
and openpyxl. Development tools include ipykernel, JupyterLab, and Ruff.

```bash
# Add a dependency and update uv.lock
uv add package-name

# Check Python files and notebooks
uv run ruff check .

# Format Python files and notebooks
uv run ruff format .

# Optional: work in JupyterLab
uv run jupyter lab
```

Run these commands from the repository root. `uv run` uses the project environment,
so you do not need to activate `.venv` manually.

The notebook locates the project root and defines `DATA_DIR` and `OUTPUT_DIR`.
For example, load data with `pd.read_csv(DATA_DIR / "data.csv")` and save a plot
with `plt.savefig(OUTPUT_DIR / "plot.png")`.

## Git and GitHub

The repository contains code, notebooks, small datasets needed to reproduce analyses,
and selected results. `.venv`, caches, `.env` files, and the contents of `materials/`
directories are ignored. The local PDF for the first assignment is stored in
`labs/lab01/materials/`.

Review your changes and notebook outputs before pushing:

```bash
git status
git diff
git add labs/lab01
git commit -m "Add lab 01 solutions"
git push
```

When adding dependencies, include `pyproject.toml` and `uv.lock` in the same commit.

For more details, see [using uv with Jupyter in VS Code](https://docs.astral.sh/uv/guides/integration/jupyter/#using-jupyter-from-vs-code).
