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

LAB_DIR = Path(__file__).resolve().parent
DATA_DIR = LAB_DIR / "data"
OUTPUT_DIR = LAB_DIR / "outputs"
REPORT_DIR = LAB_DIR / "report"
IMAGE_DIR = REPORT_DIR / "images"
TABLE_DIR = REPORT_DIR / "tables"
SELECTED_SERIES = ("seria1", "seria9", "seria10")
COLORS = ("#156082", "#E97132")


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


def detect_outliers(frame: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Flag y values outside the Tukey IQR bounds; keep the original rows."""
    if multiplier <= 0:
        raise ValueError("The IQR multiplier must be positive.")
    y = frame.y.to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError("Observations must be finite.")
    q1, q3 = np.quantile(y, [0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    result = frame.copy()
    result["is_outlier"] = (y < lower) | (y > upper)
    result.attrs.update(
        q1=float(q1),
        q3=float(q3),
        iqr=float(iqr),
        lower=float(lower),
        upper=float(upper),
        multiplier=multiplier,
    )
    return result


def export_formats(frame: pd.DataFrame, destination: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Write the same records to CSV, JSON, and XML and measure their sizes."""
    destination.mkdir(parents=True, exist_ok=True)
    records = frame[["series", "x", "y"]].to_dict(orient="records")
    frame.to_csv(destination / "anscombe.csv", index=False, lineterminator="\n")
    payload = json.dumps(records, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    (destination / "anscombe.json").write_text(payload + "\n", encoding="utf-8")
    root = ET.Element("observations")
    for record in records:
        observation = ET.SubElement(root, "observation")
        for key, value in record.items():
            ET.SubElement(observation, key).text = str(value)
    payload = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    (destination / "anscombe.xml").write_bytes(payload)
    return pd.DataFrame(
        [
            {"file": name, "bytes": (destination / name).stat().st_size}
            for name in ("anscombe.csv", "anscombe.json", "anscombe.xml")
        ]
    )


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
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True, layout="constrained")
    for ax, (name, group) in zip(axes.flat, frame.groupby("series", sort=False), strict=True):
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        grid = np.array([3, 20])
        ax.scatter(group.x, group.y, color=COLORS[0], s=30, label="Dane", zorder=3)
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color=COLORS[1], label="Regresja liniowa")
        ax.set(xlim=(3, 20), ylim=(2, 14), xlabel="x", ylabel="y", title=f"Seria {name}")
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, "anscombe_regression")


def plot_outliers(diagnostics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 3.3), sharex=True, sharey=True, layout="constrained"
    )
    retained = diagnostics.loc[~diagnostics.is_outlier]
    for ax, group, title in zip(
        axes,
        (diagnostics, retained),
        ("Przed usunięciem punktu", "Po usunięciu punktu"),
        strict=True,
    ):
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        grid = np.array([3, 15])
        ax.scatter(group.x, group.y, color=COLORS[0], s=30, label="Dane", zorder=3)
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color=COLORS[1], label="Regresja liniowa")
        ax.set(xlim=(3, 15), ylim=(4, 14), xlabel="x", ylabel="y", title=title)
    axes[0].legend(fontsize=8)
    save_figure(fig, "outlier_comparison")


def plot_task3(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4.2), sharex=True, sharey=True, layout="constrained"
    )
    for ax, name in zip(axes, SELECTED_SERIES, strict=True):
        group = frame.loc[frame.series == name]
        fit = fit_ols(group.x.to_numpy(), group.y.to_numpy())
        grid = np.array([0, 100])
        ax.scatter(group.x, group.y, color=COLORS[0], s=13, zorder=3)
        ax.plot(grid, fit["beta0"] + fit["beta1"] * grid, color=COLORS[1])
        ax.set(xlim=(0, 100), ylim=(0, 105), xlabel="x", title=name)
        ax.set_aspect("equal", adjustable="box")
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
    anscombe: pd.DataFrame, task3: pd.DataFrame, formats: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    keys = ("mean_x", "mean_y", "var_x_sample", "var_y_sample", "pearson_r", "beta0", "beta1")
    header = r"Seria & $\bar{x}$ & $\bar{y}$ & $s_x^2$ & $s_y^2$ & $r$ & $\beta_0$ & $\beta_1$"
    for name, summary in (("anscombe_statistics", anscombe), ("task3_statistics", task3)):
        rows = [
            [row.series, *[latex_number(getattr(row, key), 4) for key in keys]]
            for row in summary.itertuples()
        ]
        write_table(name, header, rows, "lrrrrrrr")
    rows = [[row.file.rsplit(".", 1)[1].upper(), str(row.bytes)] for row in formats.itertuples()]
    write_table("serialization", r"Format & Rozmiar [B]", rows, "lr")
    rows = [
        [
            row.series,
            str(int(row.n)),
            latex_number(row.beta0, 4),
            latex_number(row.beta1, 4),
            latex_number(row.r_squared, 6),
        ]
        for row in comparison.itertuples()
    ]
    write_table("outlier_summary", r"Dane & $n$ & $\beta_0$ & $\beta_1$ & $R^2$", rows, "lrrrr")


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
    write_report_tables(summary1, summary3, formats, comparison)
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
        "LowerBound": parameters["lower"],
        "UpperBound": parameters["upper"],
    }
    macros = [
        rf"\newcommand{{\{name}}}{{{latex_number(value, 3 if 'Bound' in name else 6)}}}"
        for name, value in macro_values.items()
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
