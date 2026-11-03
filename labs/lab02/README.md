# Lab 02 - Polynomial least squares and regularization

Complete solutions to all three tasks in the supplied assignment. Its PDF filename
contains `lab3`, but the document heading identifies it as **Laboratory 2**.
The Polish report follows the concise style of lab 01. No Excel workbook is required.

## Run

Use the existing project environment from the repository root:

```bash
uv sync --locked
uv run python labs/lab02/cwiczenie2.py
```

Alternatively, run all cells in `solution.ipynb` using the project's `.venv` kernel.
The analysis is self-contained and generates its observations from fixed NumPy
`default_rng` seeds: 2026 for tasks 1-2 and 2027 for task 3.
The input formulas, sample sizes, intervals, and noise level follow the assignment.

## Results

- `data/task1.csv` and `data/task3.csv` contain the generated observations.
- `outputs/task1_statistics.csv` and `outputs/task2_statistics.csv` contain all
  one-variable coefficients and scores. `mse_true` and `r2_true` compare predictions
  with the noise-free function at the 25 sample arguments; `mse_y` and `r2_y` compare
  predictions with the noisy fitting response.
- `outputs/task3_coefficients.csv` and `outputs/task3_statistics.csv` contain the
  two-variable coefficients and errors on the training points and full 40 by 40
  evaluation grid, including its boundaries.
- `report/main.tex`, `report/results.tex`, `report/tables/`, and `report/images/`
  contain the editable Polish LaTeX report and generated figures and tables.

Least squares uses `numpy.linalg.lstsq` without forming the normal equations.
Ridge and LASSO standardize each nonconstant polynomial feature separately, using
population standard deviations. The intercept is not penalized. Reported coefficients
are converted back to the original monomial basis. Ridge uses the SVD solver; LASSO
uses cyclic coordinate descent with `tol=1e-12` and `max_iter=100000`.
Their alpha scales differ because scikit-learn normalizes the LASSO squared loss
by `2 * n_samples`, while Ridge uses an unnormalized sum of squared errors.

Implementation details were checked against the official
[NumPy least-squares documentation](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html),
[LAPACK GELSD description](https://netlib.org/lapack/explore-html/d9/d67/group__gelsd_ga0bee7e1b9e7e43f59ecf2419b2759c42.html),
[Ridge documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
and [LASSO documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html).

## Verification

```bash
uv run python -m unittest discover -s labs/lab02 -p 'test_*.py'
uv run ruff check labs/lab02
```

The numerical checks compare least squares with an independent QR solution, verify
both score references against scikit-learn, check Ridge/LASSO optimality conditions,
and independently evaluate the mixed polynomial terms and grid errors.

## PDF and submission

With TeX Live and `latexmk` installed:

```bash
uv run python labs/lab02/package_submission.py
```

This reruns the analysis, compiles `outputs/sprawozdanie.pdf`, and creates
`submission/cw2_275462_275486.zip`. The archive contains exactly two files at its root:

- `cwiczenie2.py`
- `sprawozdanie.pdf`

The Python analysis works after extracting just those two files into an empty folder.
It requires NumPy, pandas, Matplotlib, and scikit-learn, as provided by this repository.
The packaging script, tests, README, source observations, and LaTeX files are excluded
from the submission ZIP. The final ZIP is tracked in Git.

To edit the report in Overleaf, upload the contents of `report/` and select `main.tex`
with pdfLaTeX. Generated table and figure files are already included. The default
authors, indices, and group match lab 01 and are stored in `author.example.tex`;
an optional local `author.tex` overrides them. No bibliography is needed.
