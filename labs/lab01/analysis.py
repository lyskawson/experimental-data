"""Reproducible solutions to lab 01: regression, serialization, and visual diagnostics."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
OUTPUT_DIR = LAB_DIR / "outputs"
REPORT_DIR = LAB_DIR / "report"
IMAGE_DIR = REPORT_DIR / "images"
TABLE_DIR = REPORT_DIR / "tables"
SELECTED_SERIES = ("seria1", "seria9", "seria10")
COLORS = ("#245B78", "#B55A36", "#387B69", "#977022")


def fit_ols(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    """Compute centered ordinary least squares with an intercept; report both variances."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 3:
        raise ValueError("Provide equally sized one-dimensional arrays with at least 3 points.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Observations must be finite.")
    dx, dy = x - x.mean(), y - y.mean()
    sxx, syy = float(dx @ dx), float(dy @ dy)
    if sxx <= 0 or syy <= 0:
        raise ValueError("Both variables must have a nonzero variance.")
    sxy = float(dx @ dy)
    beta1 = sxy / sxx
    beta0 = float(y.mean() - beta1 * x.mean())
    residual = y - (beta0 + beta1 * x)
    sse = float(residual @ residual)
    return {
        "n": len(x),
        "mean_x": float(x.mean()),
        "mean_y": float(y.mean()),
        "var_x_sample": float(x.var(ddof=1)),
        "var_y_sample": float(y.var(ddof=1)),
        "var_x_population": float(x.var(ddof=0)),
        "var_y_population": float(y.var(ddof=0)),
        "pearson_r": sxy / np.sqrt(sxx * syy),
        "beta0": beta0,
        "beta1": beta1,
        "r_squared": 1 - sse / syy,
        "sse": sse,
        "rmse": float(np.sqrt(sse / len(x))),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize series without changing their order or original observations."""
    rows = []
    for name, group in frame.groupby("series", sort=False):
        rows.append({"series": name, **fit_ols(group.x.to_numpy(), group.y.to_numpy())})
    return pd.DataFrame(rows)


def load_anscombe() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "anscombe.csv", float_precision="round_trip")


def load_task3() -> pd.DataFrame:
    frame = pd.read_csv(DATA_DIR / "dane_zad3.csv", float_precision="round_trip")
    frame = frame.rename(columns={"seria danych": "series"})
    if list(frame.columns) != ["series", "x", "y"] or frame.isna().any().any():
        raise ValueError("Task 3 requires complete series, x, and y columns.")
    expected = {f"seria{i}" for i in range(1, 14)}
    if set(frame.series) != expected or not frame.groupby("series").size().eq(142).all():
        raise ValueError("Expected the 13 complete series from the supplied screenshot.")
    return frame


def detect_outliers(
    frame: pd.DataFrame, threshold: float = 3.5, measurement_resolution: float = 0.01
) -> pd.DataFrame:
    """Flag robust residuals using a Theil-Sen line and the normal-scaled MAD.

    The threshold is fixed before comparing R-squared values. No original row is deleted.
    The known y resolution supplies a rounding-noise floor if the MAD collapses.
    """
    if threshold <= 0:
        raise ValueError("The residual threshold must be positive.")
    if measurement_resolution <= 0:
        raise ValueError("The measurement resolution must be positive.")
    x, y = frame.x.to_numpy(), frame.y.to_numpy()
    fit_ols(x, y)
    robust = stats.theilslopes(y, x, method="separate")
    residual = y - (robust.intercept + robust.slope * x)
    center = float(np.median(residual))
    scale = float(stats.median_abs_deviation(residual, scale="normal"))
    deviation = np.abs(residual - center)
    scoring_scale = max(scale, measurement_resolution / np.sqrt(12))
    score = deviation / scoring_scale
    result = frame.copy()
    result["robust_residual"] = residual
    result["robust_score"] = score
    result["is_outlier"] = score > threshold
    result.attrs.update(
        robust_beta0=float(robust.intercept),
        robust_beta1=float(robust.slope),
        residual_median=center,
        residual_mad_normal=scale,
        scoring_scale=float(scoring_scale),
        measurement_resolution=measurement_resolution,
        threshold=threshold,
    )
    return result


def export_formats(frame: pd.DataFrame, destination: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Serialize the same ordered 44 observations with deterministic UTF-8 settings."""
    destination.mkdir(parents=True, exist_ok=True)
    records = frame[["series", "x", "y"]].to_dict(orient="records")
    frame.to_csv(destination / "anscombe.csv", index=False, lineterminator="\n")
    for name, indent in (("anscombe.json", None), ("anscombe_pretty.json", 2)):
        options = {"separators": (",", ":")} if indent is None else {"indent": indent}
        text = json.dumps(records, ensure_ascii=False, allow_nan=False, **options) + "\n"
        (destination / name).write_text(text, encoding="utf-8")

    def xml_tree() -> ET.Element:
        root = ET.Element("observations")
        for record in records:
            observation = ET.SubElement(root, "observation")
            for key, value in record.items():
                ET.SubElement(observation, key).text = str(value)
        return root

    for name, pretty in (("anscombe.xml", False), ("anscombe_pretty.xml", True)):
        root = xml_tree()
        if pretty:
            ET.indent(root, space="  ")
        payload = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
        (destination / name).write_bytes(payload)
    names = (
        "anscombe.csv",
        "anscombe.json",
        "anscombe_pretty.json",
        "anscombe.xml",
        "anscombe_pretty.xml",
    )
    rows = [{"file": name, "bytes": (destination / name).stat().st_size} for name in names]
    result = pd.DataFrame(rows)
    result["relative_to_csv"] = result.bytes / result.bytes.iloc[0]
    return result


def configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "savefig.dpi": 180,
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMAGE_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(IMAGE_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def plot_anscombe(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        2, 2, figsize=(10, 6.8), sharex=True, sharey=True, layout="constrained"
    )
    residual_fig, residual_axes = plt.subplots(
        2, 2, figsize=(10, 5.6), sharex=True, sharey=True, layout="constrained"
    )
    for index, (name, group) in enumerate(frame.groupby("series", sort=False)):
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        ax = axes.flat[index]
        ax.scatter(group.x, group.y, color=COLORS[index], s=35, label="Obserwacje", zorder=3)
        grid = np.array([3, 20])
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color="#333C45", label="Regresja MNK")
        ax.set(xlim=(3, 20), ylim=(2.5, 14.5), xlabel="x", ylabel="y", title=f"Seria {name}")
        ax.text(
            0.04,
            0.94,
            f"r = {fit['pearson_r']:.6f}\nR² = {fit['r_squared']:.6f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        if index == 0:
            ax.legend(loc="lower right", fontsize=9)
        residual = group.y - (fit["beta0"] + fit["beta1"] * group.x)
        ax = residual_axes.flat[index]
        ax.axhline(0, color="#555555", linewidth=1)
        ax.scatter(group.x, residual, color=COLORS[index], s=30, zorder=3)
        ax.set(
            xlim=(3, 20), ylim=(-2.6, 4.2), xlabel="x", ylabel="Reszta y - ŷ", title=f"Seria {name}"
        )
    save_figure(fig, "anscombe_regression")
    save_figure(residual_fig, "anscombe_residuals")


def plot_outliers(diagnostics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 4.3), sharex=True, sharey=True, layout="constrained"
    )
    retained = diagnostics.loc[~diagnostics.is_outlier]
    removed = diagnostics.loc[diagnostics.is_outlier]
    for ax, group, title in zip(
        axes,
        (diagnostics, retained),
        ("Przed filtrowaniem: 11 punktów", "Po filtrowaniu: 10 punktów"),
        strict=True,
    ):
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        grid = np.array([3, 15])
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color=COLORS[0], label="Regresja MNK")
        ax.scatter(retained.x, retained.y, s=38, color=COLORS[0], zorder=3)
        ax.scatter(
            removed.x,
            removed.y,
            marker="x",
            s=90,
            linewidths=2,
            color=COLORS[1],
            label="Punkt oznaczony jako odstający",
            zorder=4,
        )
        ax.set(xlabel="x", ylabel="y", title=title, xlim=(3, 15), ylim=(4, 14))
        ax.text(0.04, 0.94, f"R² = {fit['r_squared']:.6f}", transform=ax.transAxes, va="top")
    axes[0].legend(loc="lower right", fontsize=8)
    save_figure(fig, "outlier_comparison")


def plot_task3(frame: pd.DataFrame) -> None:
    descriptions = {"seria1": "dino", "seria9": "circle", "seria10": "bullseye"}
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.4), sharex=True, sharey=True, layout="constrained"
    )
    for index, (ax, name) in enumerate(zip(axes, SELECTED_SERIES, strict=True)):
        group = frame.loc[frame.series == name]
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        ax.scatter(group.x, group.y, s=13, color=COLORS[index], alpha=0.9, zorder=3)
        grid = np.array([0, 100])
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color=COLORS[3], linewidth=1.7)
        ax.set(xlim=(0, 100), ylim=(0, 105), xlabel="x", title=f"{name} ({descriptions[name]})")
        ax.set_aspect("equal", adjustable="box")
        ax.text(
            0.03,
            0.96,
            f"r = {fit['pearson_r']:.6f}\nR² = {fit['r_squared']:.6f}",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    axes[0].set_ylabel("y")
    save_figure(fig, "task3_regression")


def latex_number(value: float | int, precision: int = 6) -> str:
    return rf"\num{{{value:.{precision}f}}}"


def write_table(name: str, header: str, rows: list[list[str]], alignment: str) -> None:
    lines = [rf"\begin{{tabular}}{{{alignment}}}", r"\toprule", header + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (TABLE_DIR / f"{name}.tex").write_text("\n".join(lines), encoding="utf-8")


def write_report_tables(
    anscombe: pd.DataFrame,
    task3: pd.DataFrame,
    formats: pd.DataFrame,
    diagnostics: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    for name, summary in (("anscombe_statistics", anscombe), ("task3_statistics", task3)):
        rows = [
            [
                row.series,
                str(int(row.n)),
                *[
                    latex_number(getattr(row, key))
                    for key in ("mean_x", "mean_y", "var_x_sample", "var_y_sample", "pearson_r")
                ],
            ]
            for row in summary.itertuples()
        ]
        write_table(
            name, r"Seria & $n$ & $\bar{x}$ & $\bar{y}$ & $s_x^2$ & $s_y^2$ & $r$", rows, "lrrrrrr"
        )
    for name, summary in (("anscombe_regression", anscombe), ("task3_regression", task3)):
        rows = [
            [
                row.series,
                *[
                    latex_number(getattr(row, key))
                    for key in ("beta0", "beta1", "r_squared", "rmse")
                ],
            ]
            for row in summary.itertuples()
        ]
        write_table(name, r"Seria & $\hat\beta_0$ & $\hat\beta_1$ & $R^2$ & RMSE", rows, "lrrrr")
    rows = [
        [row.series, latex_number(row.var_x_population), latex_number(row.var_y_population)]
        for row in anscombe.itertuples()
    ]
    write_table(
        "population_variances", r"Seria & $v_x$ (dzielnik $n$) & $v_y$ (dzielnik $n$)", rows, "lrr"
    )
    rows = [
        [
            r"\texttt{" + row.file.replace("_", r"\_") + "}",
            str(row.bytes),
            latex_number(row.relative_to_csv, 3),
        ]
        for row in formats.itertuples()
    ]
    write_table("serialization", r"Plik & Rozmiar [B] & Względem CSV", rows, "lrr")
    rows = [
        [
            row.series,
            str(int(row.n)),
            *[latex_number(getattr(row, key)) for key in ("beta0", "beta1", "r_squared", "rmse")],
        ]
        for row in comparison.itertuples()
    ]
    write_table(
        "outlier_summary",
        r"Dane & $n$ & $\hat\beta_0$ & $\hat\beta_1$ & $R^2$ & RMSE",
        rows,
        "lrrrrr",
    )
    rows = [
        [
            str(i),
            latex_number(row.x, 2),
            latex_number(row.y, 2),
            latex_number(row.robust_residual),
            latex_number(row.robust_score, 3),
            "tak" if row.is_outlier else "nie",
        ]
        for i, row in enumerate(diagnostics.itertuples(), 1)
    ]
    write_table(
        "outlier_diagnostics",
        r"Nr & $x$ & $y$ & $e_i^{\mathrm{TS}}$ & $z_i$ & Odrzucenie",
        rows,
        "rrrrrl",
    )


def run_analysis() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    anscombe = load_anscombe()
    if list(anscombe.groupby("series", sort=False).size()) != [11] * 4:
        raise ValueError("Anscombe's quartet must contain 11 observations in each of four series.")
    task3 = load_task3()
    selected = task3.loc[task3.series.isin(SELECTED_SERIES)]
    summary1, summary3 = summarize(anscombe), summarize(selected)
    formats = export_formats(anscombe)
    diagnostics = detect_outliers(anscombe.loc[anscombe.series == "III"])
    retained = diagnostics.loc[~diagnostics.is_outlier]
    comparison = pd.DataFrame(
        [
            {"series": "Przed", **fit_ols(diagnostics.x.to_numpy(), diagnostics.y.to_numpy())},
            {"series": "Po", **fit_ols(retained.x.to_numpy(), retained.y.to_numpy())},
        ]
    )
    for name, frame in (
        ("task1_statistics", summary1),
        ("task3_statistics", summary3),
        ("serialization_sizes", formats),
        ("outlier_diagnostics", diagnostics),
        ("outlier_comparison", comparison),
    ):
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False, lineterminator="\n")
    configure_plots()
    plot_anscombe(anscombe)
    plot_outliers(diagnostics)
    plot_task3(task3)
    write_report_tables(summary1, summary3, formats, diagnostics, comparison)
    parameters = {**diagnostics.attrs, "selected_series": list(SELECTED_SERIES)}
    (OUTPUT_DIR / "analysis_parameters.json").write_text(
        json.dumps(parameters, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    macro_values = {
        "FullRtwo": comparison.iloc[0].r_squared,
        "CleanRtwo": comparison.iloc[1].r_squared,
        "FullSlope": comparison.iloc[0].beta1,
        "CleanSlope": comparison.iloc[1].beta1,
        "FullIntercept": comparison.iloc[0].beta0,
        "CleanIntercept": comparison.iloc[1].beta0,
        "RobustSlope": parameters["robust_beta1"],
        "RobustIntercept": parameters["robust_beta0"],
        "ResidualCenter": parameters["residual_median"],
        "ResidualScale": parameters["residual_mad_normal"],
        "OutlierScore": diagnostics.loc[diagnostics.is_outlier, "robust_score"].iloc[0],
    }
    macros = [
        rf"\newcommand{{\{name}}}{{{latex_number(value)}}}" for name, value in macro_values.items()
    ]
    (REPORT_DIR / "results.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
    return {
        "task1": summary1,
        "task3": summary3,
        "formats": formats,
        "diagnostics": diagnostics,
        "comparison": comparison,
    }


if __name__ == "__main__":
    results = run_analysis()
    for name in ("task1", "formats", "comparison", "task3"):
        print(f"\n{name}\n{results[name].to_string(index=False)}")
