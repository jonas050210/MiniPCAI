#!/usr/bin/env python3
"""Verify that built wheels/sdists ship the application data.

Regression guard for the packaging defect where an installed MiniPCAI could
neither resolve targets nor train ("trained model not found", ``DatasetError``)
because ``data/`` stayed in the repository.

    python scripts/check_wheel_contents.py [dist/]

Exit codes: 0 = ok, 1 = a required file is missing, 2 = no artifact found.
"""

from __future__ import annotations

import sys
import tarfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

REQUIRED_WHEEL_FILES = (
    "minipcai/__init__.py",
    "minipcai/cli.py",
    "minipcai/data/registry.json",
    "minipcai/ui/app.py",
)
OPTIONAL_BUT_EXPECTED = ("LICENSE",)

# Files that must never be packaged.
FORBIDDEN_SUBSTRINGS = ("/tests/", "/.git/", "__pycache__", "models/model.joblib")


def check_wheel(path: Path) -> list[str]:
    problems: list[str] = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        dataset = [name for name in names if "minipcai/data/intent_dataset.v" in name]
        for required in REQUIRED_WHEEL_FILES:
            if required not in names:
                problems.append(f"{path.name}: missing {required}")
        if not dataset:
            problems.append(f"{path.name}: no versioned dataset in minipcai/data/")
        for name in names:
            if any(fragment in f"/{name}" for fragment in FORBIDDEN_SUBSTRINGS):
                problems.append(f"{path.name}: unexpected file {name}")
    if not any(name.endswith("LICENSE") for name in names):
        print(f"note: {path.name} carries no LICENSE file (setuptools < 68?)")
    return problems


def check_sdist(path: Path) -> list[str]:
    problems: list[str] = []
    with tarfile.open(path) as archive:
        names = archive.getnames()
    if not any(name.endswith("minipcai/data/registry.json") for name in names):
        problems.append(f"{path.name}: missing minipcai/data/registry.json")
    if not any(name.endswith("LICENSE") for name in names):
        problems.append(f"{path.name}: missing LICENSE")
    return problems


def main(argv: list[str]) -> int:
    dist = Path(argv[1]) if len(argv) > 1 else REPO_ROOT / "dist"
    if not dist.is_dir():
        print(f"FAIL: {dist} does not exist (run 'python -m build')", file=sys.stderr)
        return 2
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if not wheels:
        print(f"FAIL: no wheel in {dist}", file=sys.stderr)
        return 2

    problems: list[str] = []
    for wheel in wheels:
        problems.extend(check_wheel(wheel))
    for sdist in sdists:
        problems.extend(check_sdist(sdist))

    if problems:
        print("FAIL: packaging is incomplete:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print(
        f"OK: {len(wheels)} wheel(s) and {len(sdists)} sdist(s) ship the application data"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
