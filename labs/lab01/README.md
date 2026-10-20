# Lab 01 — Linear regression and Anscombe's quartet

Solutions to all three tasks. The report briefly describes the work and results,
then gives conclusions for each task.

## Run

From the repository root:

```bash
uv sync --locked
uv run python labs/lab01/analysis.py
```

Or open `solution.ipynb` in VS Code, select the `.venv` kernel, and run all cells.
Sample variances use the divisor `n - 1`. Task 2 removes values outside the
`1.5 × IQR` bounds for `y` in series III. Task 3 uses series 1, 9, and 10.

## Results

- `outputs/lab01.xlsx` — one worksheet with all four report tables, nine editable
  charts, source data, and Excel formulas. Tables are black and white; charts use
  the default Office colors. The workbook was checked in Microsoft Excel.
- `outputs/anscombe.csv`, `.json`, `.xml` — the same 44 observations in three formats.
- `analysis.py` — calculations and report figures/tables.
- `report/main.tex` — the Polish report, with generated tables and figures.

The full task 3 input is in `data/dane_zad3.csv`; its provenance is recorded in
`data/provenance.json`. Numerical checks can be run with:

```bash
uv run python -m unittest discover -s labs/lab01 -p 'test_*.py'
```

## PDF and ZIP archives

Copy `report/author.example.tex` to `report/author.tex` and enter the authors and
group. With TeX Live and `latexmk` installed, run:

```bash
uv run python labs/lab01/package_submission.py
```

This creates `outputs/report.pdf`, `submission/overleaf.zip`, and
`submission/cw1_<index>_<index>.zip`. The Overleaf archive is a standalone project
using `main.tex` and pdfLaTeX. The submission ZIP contains the PDF, workbook,
notebook, code, data, and LaTeX sources.

Personal author details, personalized PDFs, and submission ZIPs are ignored by Git.
