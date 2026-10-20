"""Compile the report and create the ePortal and standalone Overleaf ZIP archives."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import zipfile
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LAB_DIR.parents[1]
REPORT_DIR = LAB_DIR / "report"
SUBMISSION_DIR = LAB_DIR / "submission"


def report_sources() -> list[Path]:
    sources = list(REPORT_DIR.glob("*.tex"))
    sources.extend((REPORT_DIR / "tables").glob("*.tex"))
    sources.extend((REPORT_DIR / "images").glob("*.pdf"))
    sources.extend((REPORT_DIR / "images").glob("*.png"))
    return sorted(sources)


def main() -> None:
    required = (
        "analysis.py",
        "solution.ipynb",
        "test_analysis.py",
        "data/anscombe.csv",
        "data/dane_zad3.csv",
        "outputs/lab01.xlsx",
        "outputs/task1_statistics.csv",
        "outputs/task3_statistics.csv",
        "report/main.tex",
        "report/results.tex",
    )
    for name in required:
        if not (LAB_DIR / name).is_file():
            raise SystemExit(f"Required file is missing: {name}. Run the analysis first.")
    author_file = REPORT_DIR / "author.tex"
    if not author_file.is_file():
        raise SystemExit(
            "Copy report/author.example.tex to report/author.tex and enter your details."
        )
    author = author_file.read_text(encoding="utf-8")
    match = re.search(r"\\newcommand\{\\StudentIndices\}\{([\d, ]+)\}", author)
    if match is None:
        raise SystemExit("author.tex must define StudentIndices as comma-separated index numbers.")
    indices = re.findall(r"\d+", match[1])
    if not indices or len(indices) != len(set(indices)):
        raise SystemExit("Enter at least one index number; index numbers must not be repeated.")
    if shutil.which("latexmk") is None:
        raise SystemExit("Install latexmk/TeX Live to compile locally, or use Overleaf.")

    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=REPORT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    pdf = LAB_DIR / "outputs" / "report.pdf"
    shutil.copy2(REPORT_DIR / "main.pdf", pdf)
    SUBMISSION_DIR.mkdir(exist_ok=True)
    overleaf = SUBMISSION_DIR / "overleaf.zip"
    with zipfile.ZipFile(overleaf, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in report_sources():
            archive.write(path, path.relative_to(REPORT_DIR).as_posix())

    included: dict[str, Path] = {}
    for name in ("README.md", "pyproject.toml", "uv.lock", ".python-version", "scripts/new_lab.py"):
        included[name] = PROJECT_ROOT / name
    for name in (
        "README.md",
        "analysis.py",
        "solution.ipynb",
        "test_analysis.py",
        "package_submission.py",
    ):
        path = LAB_DIR / name
        included[path.relative_to(PROJECT_ROOT).as_posix()] = path
    for path in (LAB_DIR / "data").iterdir():
        if path.suffix in (".csv", ".json"):
            included[path.relative_to(PROJECT_ROOT).as_posix()] = path
    for path in (LAB_DIR / "outputs").iterdir():
        if path.suffix in (".csv", ".json", ".xml") or path.name == "lab01.xlsx":
            included[path.relative_to(PROJECT_ROOT).as_posix()] = path
    for path in report_sources():
        included[path.relative_to(PROJECT_ROOT).as_posix()] = path
    for path in (PROJECT_ROOT / "labs" / "_template").rglob("*"):
        if path.is_file() and (path.name == ".gitkeep" or path.suffix in (".md", ".ipynb")):
            included[path.relative_to(PROJECT_ROOT).as_posix()] = path
    included["report.pdf"] = pdf
    included["overleaf.zip"] = overleaf
    for name, path in included.items():
        if not path.is_file():
            raise SystemExit(f"Required file is missing: {name}")

    filename = "cw1_" + "_".join(indices) + ".zip"
    destination = SUBMISSION_DIR / filename
    manifest = []
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in sorted(included.items()):
            payload = path.read_bytes()
            archive.writestr(name, payload)
            manifest.append(f"{hashlib.sha256(payload).hexdigest()}  {name}")
        archive.writestr("MANIFEST.sha256", "\n".join(manifest) + "\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The ZIP integrity check failed.")
        if "report.pdf" not in archive.namelist() or "overleaf.zip" not in archive.namelist():
            raise RuntimeError("The ZIP is missing a required deliverable.")
    print(f"Report: {pdf}")
    print(f"Overleaf project: {overleaf}")
    print(f"ePortal submission: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
