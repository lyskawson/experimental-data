"""Utwórz katalog laboratorium na podstawie labs/_template."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def create_lab(project_root: Path, number: int, title: str) -> Path:
    """Skopiuj szablon i uzupełnij numer, tytuł oraz ścieżkę notebooka."""
    if number < 1:
        raise ValueError("Numer laboratorium musi być dodatni.")

    name = f"lab{number:02d}"
    template = project_root / "labs" / "_template"
    destination = project_root / "labs" / name
    heading = f"Laboratorium {number:02d} - {title}"

    # Sprawdź i przygotuj zawartość przed utworzeniem katalogu docelowego.
    notebook = json.loads((template / "solution.ipynb").read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = [f"# {heading}\n"]
    for cell in notebook["cells"]:
        cell["source"] = [line.replace('"_template"', f'"{name}"') for line in cell["source"]]

    readme = (template / "README.md").read_text(encoding="utf-8")
    readme = readme.replace("Laboratorium - szablon", heading, 1)

    shutil.copytree(template, destination)
    (destination / "README.md").write_text(readme, encoding="utf-8")
    (destination / "solution.ipynb").write_text(
        json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", type=int, help="Numer laboratorium, np. 2")
    parser.add_argument("--title", default="Do uzupełnienia", help="Temat laboratorium")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]

    try:
        destination = create_lab(project_root, args.number, args.title.strip() or "Do uzupełnienia")
    except FileExistsError:
        parser.error(f"Katalog lab{args.number:02d} już istnieje. Niczego nie nadpisano.")
    except ValueError as error:
        parser.error(str(error))

    print(f"Utworzono {destination.relative_to(project_root)}/")
    print(f"Otwórz {destination.relative_to(project_root)}/solution.ipynb w VS Code.")


if __name__ == "__main__":
    main()
