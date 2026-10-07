#!/usr/bin/env python3
"""Fail when the packaged dataset no longer matches the trained model metadata.

``models/metadata.json`` records the dataset version, the example count and the
SHA-256 of the file the model was trained on. Editing the dataset (or replacing
the file) without retraining would silently invalidate every quality number in
``models/metrics.json``, so CI checks the fingerprint on every push:

    python scripts/check_dataset_fingerprint.py

Exit codes: 0 = match, 1 = mismatch, 2 = metadata/file missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from minipcai import paths  # noqa: E402


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    metadata_path = REPO_ROOT / "models" / "metadata.json"
    dataset_path = paths.default_dataset_path()
    if not metadata_path.is_file():
        print(f"FAIL: {metadata_path} is missing", file=sys.stderr)
        return 2
    if not dataset_path.is_file():
        print(f"FAIL: packaged dataset {dataset_path} is missing", file=sys.stderr)
        return 2

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded = metadata.get("dataset") or {}
    expected_hash = recorded.get("sha256")
    expected_version = recorded.get("version")
    expected_count = recorded.get("n_examples")

    actual_hash = sha256_of(dataset_path)
    problems: list[str] = []
    if expected_hash and actual_hash != expected_hash:
        problems.append(f"sha256: metadata {expected_hash[:16]}… != file {actual_hash[:16]}…")
    if expected_version is not None and paths._DATASET_NAME_RE.match(dataset_path.name):
        actual_version = int(paths._DATASET_NAME_RE.match(dataset_path.name).group(1))
        if actual_version != expected_version:
            problems.append(f"version: metadata v{expected_version} != file v{actual_version}")
    if expected_count is not None:
        with open(dataset_path, encoding="utf-8") as handle:
            actual_count = sum(1 for line in handle if line.strip())
        if actual_count != expected_count:
            problems.append(f"examples: metadata {expected_count} != file {actual_count}")

    if problems:
        print("FAIL: the packaged dataset does not match models/metadata.json:", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        print("Retrain with 'minipcai train' and commit models/metadata.json.", file=sys.stderr)
        return 1
    print(
        f"OK: dataset {dataset_path.name} matches the model metadata "
        f"(v{expected_version}, {expected_count} examples, {actual_hash[:12]}…)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
