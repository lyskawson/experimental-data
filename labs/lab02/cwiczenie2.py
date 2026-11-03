from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from sklearn.linear_model import Lasso, Ridge

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
OUTPUT_DIR = LAB_DIR / "outputs"
REPORT_DIR = LAB_DIR / "report"
IMAGE_DIR = REPORT_DIR / "images"
TABLE_DIR = REPORT_DIR / "tables"
SEED_1D = 2026
SEED_2D = 2027
ALPHAS = {"Ridge": (0.01, 0.1, 1.0, 10.0), "LASSO": (0.0001, 0.001, 0.01, 0.1)}


def generate_1d() -> pd.DataFrame:
    rng = np.random.default_rng(SEED_1D)
    x = np.linspace(0, 4, 25)
    reference = np.log2(x + 1)
    noise = 0.1 * rng.standard_normal(len(x))
    return pd.DataFrame({"x": x, "y_true": reference, "noise": noise, "y": reference + noise})


def surface(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sin(np.log(x + 2) * np.sqrt(2 * y))


def generate_2d() -> pd.DataFrame:
    rng = np.random.default_rng(SEED_2D)
    x, y = 4 * rng.random(200), 4 * rng.random(200)
    return pd.DataFrame({"x": x, "y": y, "z": surface(x, y)})


def powers_2d(degree: int) -> list[tuple[int, int]]:
    return [(i, total - i) for total in range(degree + 1) for i in range(total, -1, -1)]


def design_2d(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    return np.column_stack([x**i * y**j for i, j in powers_2d(degree)])


def fit_ols(design: np.ndarray, response: np.ndarray) -> np.ndarray:
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("The polynomial design matrix must have full column rank.")
    return coefficients


def fit_regularized(
    x: np.ndarray, response: np.ndarray, degree: int, method: str, alpha: float
) -> np.ndarray:
    features = np.vander(x, degree + 1, increasing=True)[:, 1:]
    means, scales = features.mean(axis=0), features.std(axis=0, ddof=0)
    standardized = (features - means) / scales
    if method == "Ridge":
        model = Ridge(alpha=alpha, solver="svd")
    elif method == "LASSO":
        model = Lasso(alpha=alpha, tol=1e-12, max_iter=100_000, selection="cyclic")
    else:
        raise ValueError(f"Unknown regularization method: {method}")
    model.fit(standardized, response)
    slopes = model.coef_ / scales
    intercept = model.intercept_ - means @ slopes
    return np.r_[intercept, slopes]


def scores(reference: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(reference) - prediction
    squared_error = float(np.sum(residual**2))
    total = float(np.sum((reference - np.mean(reference)) ** 2))
    return squared_error / np.size(reference), 1 - squared_error / total


def analyze_1d(data: pd.DataFrame) -> pd.DataFrame:
    x, y, reference = data.x.to_numpy(), data.y.to_numpy(), data.y_true.to_numpy()
    rows = []
    for degree in (1, 2, 3):
        design = np.vander(x, degree + 1, increasing=True)
        fits = [("MNK", 0.0, fit_ols(design, y))]
        if degree > 1:
            for method, alphas in ALPHAS.items():
                fits.extend(
                    (method, alpha, fit_regularized(x, y, degree, method, alpha))
                    for alpha in alphas
                )
        for method, alpha, coefficients in fits:
            prediction = design @ coefficients
            mse_true, r2_true = scores(reference, prediction)
            mse_y, r2_y = scores(y, prediction)
            row = {
                "method": method,
                "degree": degree,
                "alpha": alpha,
                "mse_true": mse_true,
                "r2_true": r2_true,
                "mse_y": mse_y,
                "r2_y": r2_y,
            }
            row.update({f"a{i}": float(value) for i, value in enumerate(coefficients)})
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_2d(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, np.ndarray]]:
    coordinates = np.linspace(0, 4, 40)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    reference = surface(grid_x, grid_y)
    rows, coefficients = [], {}
    for degree in (2, 3):
        design = design_2d(data.x, data.y, degree)
        coefficients[degree] = fit_ols(design, data.z.to_numpy())
        grid_design = design_2d(grid_x, grid_y, degree)
        prediction = (grid_design @ coefficients[degree]).reshape(grid_x.shape)
        error = reference - prediction
        peak = np.unravel_index(np.abs(error).argmax(), error.shape)
        rows.append(
            {
                "degree": degree,
                "n_coefficients": design.shape[1],
                "rank": int(np.linalg.matrix_rank(design)),
                "condition_number": float(np.linalg.cond(design)),
                "mse_training": scores(data.z.to_numpy(), design @ coefficients[degree])[0],
                "mse_grid": scores(reference, prediction)[0],
                "max_abs_error": float(np.abs(error).max()),
                "max_error_x": float(grid_x[peak]),
                "max_error_y": float(grid_y[peak]),
            }
        )
    return pd.DataFrame(rows), coefficients


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(IMAGE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def row_coefficients(row: pd.Series) -> np.ndarray:
    return np.array([row[f"a{i}"] for i in range(int(row.degree) + 1)])


def plot_observations(ax: plt.Axes, data: pd.DataFrame, grid: np.ndarray) -> None:
    ax.scatter(data.x, data.y, s=19, color="#156082", label="$y$ (z szumem)", zorder=4)
    ax.plot(grid, np.log2(grid + 1), "k--", lw=1.5, label="$y_0$ (bez szumu)", zorder=3)
    ax.set(xlabel="$x$", ylabel="$y$", xlim=(0, 4), ylim=(-0.2, 2.6))


def plot_1d(data: pd.DataFrame, results: pd.DataFrame) -> None:
    grid = np.linspace(0, 4, 401)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.85), sharey=True, layout="constrained")
    for ax, (_, row) in zip(axes, results.query("method == 'MNK'").iterrows(), strict=True):
        plot_observations(ax, data, grid)
        prediction = np.polynomial.polynomial.polyval(grid, row_coefficients(row))
        ax.plot(grid, prediction, color="#E97132", lw=1.8, label="MNK")
        ax.set_title(f"Stopień {int(row.degree)}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    save_figure(fig, "task1_polynomials")

    for method in ALPHAS:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.35), sharey=True, layout="constrained")
        for ax, degree in zip(axes, (2, 3), strict=True):
            plot_observations(ax, data, grid)
            baseline = results.query("method == 'MNK' and degree == @degree").iloc[0]
            ax.plot(
                grid,
                np.polynomial.polynomial.polyval(grid, row_coefficients(baseline)),
                color="0.45",
                ls=":",
                lw=2,
                label="MNK",
            )
            group = results.query("method == @method and degree == @degree")
            for color, (_, row) in zip(
                ("#E97132", "#196B24", "#A02B93", "#0F9ED5"), group.iterrows()
            ):
                ax.plot(
                    grid,
                    np.polynomial.polynomial.polyval(grid, row_coefficients(row)),
                    color=color,
                    lw=1.6,
                    label=rf"$\alpha={row.alpha:g}$",
                )
            ax.set_title(f"{method}, stopień {degree}")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
        save_figure(fig, f"task2_{method.lower()}")


def plot_2d(data: pd.DataFrame, coefficients: dict[int, np.ndarray]) -> None:
    coordinates = np.linspace(0, 4, 40)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    reference = surface(grid_x, grid_y)
    predictions = {
        degree: (design_2d(grid_x, grid_y, degree) @ values).reshape(grid_x.shape)
        for degree, values in coefficients.items()
    }
    error_limit = max(
        float(np.abs(reference - prediction).max()) for prediction in predictions.values()
    )
    surface_min = min(float(reference.min()), *(float(p.min()) for p in predictions.values()))
    surface_max = max(float(reference.max()), *(float(p.max()) for p in predictions.values()))
    fig = plt.figure(figsize=(10, 5.5), layout="constrained")
    for row, degree in enumerate((2, 3)):
        ax = fig.add_subplot(2, 2, 2 * row + 1, projection="3d")
        ax.plot_surface(
            grid_x,
            grid_y,
            predictions[degree],
            cmap="viridis",
            norm=Normalize(surface_min, surface_max),
            linewidth=0,
            alpha=0.9,
        )
        ax.scatter(data.x, data.y, data.z, s=5, color="black", alpha=0.6, label="Dane uczące")
        ax.set(
            title=f"Aproksymacja: stopień {degree}",
            zlabel="$p(x,y)$",
            zlim=(surface_min, surface_max),
        )
        if row == 0:
            ax.legend(loc="upper left", fontsize=8, frameon=False)
        error_ax = fig.add_subplot(2, 2, 2 * row + 2, projection="3d")
        error_ax.plot_surface(
            grid_x,
            grid_y,
            reference - predictions[degree],
            cmap="coolwarm",
            norm=Normalize(-error_limit, error_limit),
            linewidth=0,
        )
        error_ax.set(
            title=f"Błąd $f-p$: stopień {degree}",
            zlabel="$f(x,y)-p(x,y)$",
            zlim=(-error_limit, error_limit),
        )
        for panel in (ax, error_ax):
            panel.set(xlabel="$x$", ylabel="$y$", xlim=(0, 4), ylim=(0, 4))
            panel.set_xticks(range(5))
            panel.set_yticks(range(5))
            panel.view_init(elev=26, azim=-125)
            panel.set_box_aspect((1, 1, 0.65))
            panel.tick_params(labelsize=8, pad=0)
            panel.xaxis.labelpad = 0
            panel.yaxis.labelpad = 0
            panel.zaxis.labelpad = 2
    save_figure(fig, "task3_surfaces")


def number(value: float, precision: int = 6) -> str:
    if abs(value) < 0.5 * 10 ** (-precision):
        value = 0.0
    return rf"\num{{{value:.{precision}f}}}"


def write_table(name: str, headers: list[str], rows: list[list[str]]) -> None:
    lines = [rf"\begin{{tabular}}{{l{'r' * (len(headers) - 1)}}}", r"\toprule"]
    lines += [" & ".join(headers) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (TABLE_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def write_report_results(
    results: pd.DataFrame, surface_results: pd.DataFrame, coefficients: dict[int, np.ndarray]
) -> None:
    ols = results.query("method == 'MNK'").set_index("degree")
    rows = []
    for degree, row in ols.iterrows():
        values = [number(row[f"a{i}"]) if i <= degree else "--" for i in range(4)]
        rows.append(
            [str(degree), *values, number(row.r2_y), number(row.r2_true), number(row.mse_true)]
        )
    write_table(
        "task1_statistics",
        ["Stopień", "$a_0$", "$a_1$", "$a_2$", "$a_3$", "$R_y^2$", "$R_0^2$", r"$\mathrm{MSE}_0$"],
        rows,
    )
    for degree in (2, 3):
        rows = []
        for _, row in results.query("degree == @degree").iterrows():
            rows.append(
                [row.method, number(row.alpha, 4) if row.alpha else "--"]
                + [number(row[f"a{i}"]) for i in range(degree + 1)]
                + [number(row.mse_true)]
            )
        write_table(
            f"task2_degree{degree}",
            ["Metoda", r"$\alpha$"]
            + [f"$a_{i}$" for i in range(degree + 1)]
            + [r"$\mathrm{MSE}_0$"],
            rows,
        )
    terms = ["1", "x", "y", "x^2", "xy", "y^2", "x^3", "x^2y", "xy^2", "y^3"]
    write_table(
        "task3_coefficients",
        ["Wyraz", "Stopień 2", "Stopień 3"],
        [
            [f"${term}$", number(coefficients[2][i]) if i < 6 else "--", number(coefficients[3][i])]
            for i, term in enumerate(terms)
        ],
    )
    write_table(
        "task3_statistics",
        [
            "Stopień",
            r"$\mathrm{MSE}_{\mathrm{ucz}}$",
            r"$\mathrm{MSE}_{\mathrm{siatka}}$",
            r"$\max|f-p|$",
        ],
        [
            [
                str(int(row.degree)),
                number(row.mse_training),
                number(row.mse_grid),
                number(row.max_abs_error),
            ]
            for row in surface_results.itertuples()
        ],
    )
    best_ridge = results.query("method == 'Ridge'").sort_values("mse_true").iloc[0]
    best_lasso = results.query("method == 'LASSO'").sort_values("mse_true").iloc[0]
    surface_results = surface_results.set_index("degree")
    macros = {
        "LinearMse": ols.loc[1, "mse_true"],
        "QuadraticMse": ols.loc[2, "mse_true"],
        "CubicMse": ols.loc[3, "mse_true"],
        "CubicNoisyMse": ols.loc[3, "mse_y"],
        "BestRidgeMse": best_ridge.mse_true,
        "BestLassoMse": best_lasso.mse_true,
        "RidgeGain": 100 * (1 - best_ridge.mse_true / ols.loc[int(best_ridge.degree), "mse_true"]),
        "LassoGain": 100 * (1 - best_lasso.mse_true / ols.loc[int(best_lasso.degree), "mse_true"]),
        "SurfaceGain": 100
        * (1 - surface_results.loc[3, "mse_grid"] / surface_results.loc[2, "mse_grid"]),
        "SurfaceMaxCubic": surface_results.loc[3, "max_abs_error"],
    }
    lines = [
        rf"\newcommand{{\{key}}}{{{number(value, 1 if 'Gain' in key else 6)}}}"
        for key, value in macros.items()
    ]
    (REPORT_DIR / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for directory in (DATA_DIR, OUTPUT_DIR, IMAGE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    data_1d, data_2d = generate_1d(), generate_2d()
    data_1d.to_csv(DATA_DIR / "task1.csv", index=False)
    data_2d.to_csv(DATA_DIR / "task3.csv", index=False)
    results = analyze_1d(data_1d)
    surface_results, coefficients = analyze_2d(data_2d)
    results.query("method == 'MNK'").to_csv(OUTPUT_DIR / "task1_statistics.csv", index=False)
    results.query("degree > 1").to_csv(OUTPUT_DIR / "task2_statistics.csv", index=False)
    surface_results.to_csv(OUTPUT_DIR / "task3_statistics.csv", index=False)
    pd.DataFrame(
        [
            {"degree": degree, "power_x": i, "power_y": j, "coefficient": float(value)}
            for degree, values in coefficients.items()
            for (i, j), value in zip(powers_2d(degree), values, strict=True)
        ]
    ).to_csv(OUTPUT_DIR / "task3_coefficients.csv", index=False)
    configure_plots()
    plot_1d(data_1d, results)
    plot_2d(data_2d, coefficients)
    write_report_results(results, surface_results, coefficients)
    print(results[["method", "degree", "alpha", "mse_true", "r2_true"]].to_string(index=False))
    print(surface_results.to_string(index=False))


if __name__ == "__main__":
    main()
