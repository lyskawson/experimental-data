from __future__ import annotations

import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent
REPORT_DIR = LAB_DIR / "report"
SUBMISSION_DIR = LAB_DIR / "submission"


def main() -> None:
    if shutil.which("latexmk") is None:
        raise SystemExit("Install latexmk/TeX Live to compile locally, or use Overleaf.")
    author_file = REPORT_DIR / "author.tex"
    if not author_file.is_file():
        author_file = REPORT_DIR / "author.example.tex"
    author = author_file.read_text(encoding="utf-8")
    match = re.search(r"\\newcommand\{\\StudentIndices\}\{([\d, ]+)\}", author)
    if match is None:
        raise SystemExit("StudentIndices must contain comma-separated index numbers.")
    indices = re.findall(r"\d+", match[1])
    if len(indices) != 2 or len(set(indices)) != 2:
        raise SystemExit("Enter two different student index numbers.")

    subprocess.run([sys.executable, str(LAB_DIR / "analysis.py")], check=True)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
        cwd=REPORT_DIR,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    pdf = LAB_DIR / "outputs" / "report.pdf"
    shutil.copy2(REPORT_DIR / "main.pdf", pdf)
    included = {"analysis.py": LAB_DIR / "analysis.py", "report.pdf": pdf}
    SUBMISSION_DIR.mkdir(exist_ok=True)
    destination = SUBMISSION_DIR / ("cw2_" + "_".join(indices) + ".zip")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, path in included.items():
            archive.write(path, name)
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != set(included):
            raise RuntimeError("The submission must contain exactly analysis.py and report.pdf.")
        for name, path in included.items():
            if archive.read(name) != path.read_bytes():
                raise RuntimeError(f"The archived {name} differs from the final file.")
    print(f"Report: {pdf}")
    print(f"Submission: {destination} ({destination.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
