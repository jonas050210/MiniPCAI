"""Loading and validation of the versioned intent dataset (JSONL)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from minipcai.intents import ALL_LABELS

CATEGORIES: tuple[str, ...] = ("normal", "typo", "unknown", "ambiguous")
_REQUIRED_FIELDS = {"id", "text", "label", "category", "version"}
_FILENAME_VERSION_RE = re.compile(r"\.v(\d+)\.jsonl$")
_MAX_TEXT_LENGTH = 300


@dataclass(frozen=True)
class Example:
    id: str
    text: str
    label: str
    category: str
    version: int


@dataclass(frozen=True)
class Dataset:
    examples: tuple[Example, ...]
    version: int
    sha256: str
    path: Path

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(example.text for example in self.examples)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(example.label for example in self.examples)

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(example.category for example in self.examples)

    def label_counts(self) -> dict[str, int]:
        counts = {label: 0 for label in ALL_LABELS}
        for example in self.examples:
            counts[example.label] += 1
        return counts

    def category_counts(self) -> dict[str, int]:
        counts = {category: 0 for category in CATEGORIES}
        for example in self.examples:
            counts[example.category] += 1
        return counts


class DatasetError(ValueError):
    """Raised when the dataset file is malformed or inconsistent."""


def load_dataset(path: Path | str) -> Dataset:
    """Load and validate the dataset at ``path``.

    Validation enforces: required fields, known labels/categories, unique ids,
    unique texts (no conflicting duplicates), consistent version and a
    filename/row version match when the file follows the ``*.vN.jsonl`` scheme.
    """
    path = Path(path)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")

    raw = path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()

    examples: list[Example] = []
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    version: int | None = None

    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"line {line_number}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise DatasetError(f"line {line_number}: expected a JSON object")
        missing = _REQUIRED_FIELDS - row.keys()
        if missing:
            raise DatasetError(f"line {line_number}: missing fields {sorted(missing)}")
        if row["label"] not in ALL_LABELS:
            raise DatasetError(f"line {line_number}: unknown label {row['label']!r}")
        if row["category"] not in CATEGORIES:
            raise DatasetError(f"line {line_number}: unknown category {row['category']!r}")
        if not isinstance(row["version"], int):
            raise DatasetError(f"line {line_number}: version must be an integer")
        if version is None:
            version = row["version"]
        elif version != row["version"]:
            raise DatasetError(f"line {line_number}: inconsistent version {row['version']}")
        text = row["text"].strip()
        if not text or len(text) > _MAX_TEXT_LENGTH:
            raise DatasetError(f"line {line_number}: invalid text length")
        if row["id"] in seen_ids:
            raise DatasetError(f"line {line_number}: duplicate id {row['id']!r}")
        if text in seen_texts:
            raise DatasetError(f"line {line_number}: duplicate text {text!r}")
        seen_ids.add(row["id"])
        seen_texts.add(text)
        examples.append(
            Example(
                id=row["id"], text=text, label=row["label"],
                category=row["category"], version=row["version"],
            )
        )

    if not examples:
        raise DatasetError("dataset is empty")
    assert version is not None

    match = _FILENAME_VERSION_RE.search(path.name)
    if match and int(match.group(1)) != version:
        raise DatasetError(
            f"filename declares version {match.group(1)} but rows declare {version}"
        )

    return Dataset(
        examples=tuple(examples), version=version, sha256=sha256, path=path
    )
