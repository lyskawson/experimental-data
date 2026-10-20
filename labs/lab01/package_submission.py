"""Compile the report and package only the Python code, workbook, and PDF."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPORT_DIR = LAB_DIR / "report"
SUBMISSION_DIR = LAB_DIR / "submission"


def main() -> None:
    required = (
        "analysis.py",
        "outputs/lab01.xlsx",
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
    included = {
        "analysis.py": LAB_DIR / "analysis.py",
        "lab01.xlsx": LAB_DIR / "outputs" / "lab01.xlsx",
        "report.pdf": pdf,
    }

    filename = "cw1_" + "_".join(indices) + ".zip"
    destination = SUBMISSION_DIR / filename
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in included.items():
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("The ZIP integrity check failed.")
        if set(archive.namelist()) != set(included):
            raise RuntimeError("The ZIP must contain exactly the Python code, workbook, and PDF.")
    print(f"Report: {pdf}")
    print(f"ePortal submission: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
