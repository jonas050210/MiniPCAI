"""Tests for dataset integrity and the loader's validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from minipcai.config import DEFAULT_DATASET_PATH
from minipcai.dataset import DatasetError, load_dataset
from minipcai.intents import ALL_LABELS


class TestRealDataset:
    def test_loads_and_is_valid(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        assert dataset.version == 1
        assert len(dataset.examples) > 700
        assert len(dataset.sha256) == 64

    def test_all_labels_have_enough_examples(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        counts = dataset.label_counts()
        for label in ALL_LABELS:
            assert counts[label] >= 30, f"label {label} has only {counts[label]} examples"

    def test_all_categories_present(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        counts = dataset.category_counts()
        assert counts["normal"] > 400
        assert counts["typo"] >= 100
        assert counts["unknown"] >= 75
        assert counts["ambiguous"] >= 20

    def test_ambiguous_and_unknown_examples_share_the_unknown_label(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        for example in dataset.examples:
            if example.category in ("unknown", "ambiguous"):
                assert example.label == "unknown"

    def test_ids_are_unique_and_prefixed_with_label(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        ids = [example.id for example in dataset.examples]
        assert len(ids) == len(set(ids))
        for example in dataset.examples:
            assert example.id.startswith(example.label + "-")


def _write(tmp_path: Path, rows: list[dict], name: str = "intent_dataset.v1.jsonl") -> Path:
    path = tmp_path / name
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
    )
    return path


VALID_ROW = {
    "id": "open_app-0001",
    "text": "öffne notepad",
    "label": "open_app",
    "category": "normal",
    "version": 1,
}


class TestLoaderValidation:
    def test_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="not found"):
            load_dataset(tmp_path / "nope.jsonl")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "intent_dataset.v1.jsonl"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(DatasetError, match="invalid JSON"):
            load_dataset(path)

    def test_missing_field(self, tmp_path):
        row = dict(VALID_ROW, id=None)
        del row["id"]
        with pytest.raises(DatasetError, match="missing fields"):
            load_dataset(_write(tmp_path, [VALID_ROW, row]))

    def test_unknown_label(self, tmp_path):
        row = dict(VALID_ROW, id="x-0001", label="reboot_pc")
        with pytest.raises(DatasetError, match="unknown label"):
            load_dataset(_write(tmp_path, [row]))

    def test_unknown_category(self, tmp_path):
        row = dict(VALID_ROW, category="weird")
        with pytest.raises(DatasetError, match="unknown category"):
            load_dataset(_write(tmp_path, [row]))

    def test_duplicate_id(self, tmp_path):
        with pytest.raises(DatasetError, match="duplicate id"):
            load_dataset(_write(tmp_path, [VALID_ROW, VALID_ROW]))

    def test_duplicate_text(self, tmp_path):
        row = dict(VALID_ROW, id="x-0001")
        with pytest.raises(DatasetError, match="duplicate text"):
            load_dataset(_write(tmp_path, [VALID_ROW, row]))

    def test_inconsistent_version(self, tmp_path):
        row = dict(VALID_ROW, id="x-0001", version=2)
        with pytest.raises(DatasetError, match="inconsistent version"):
            load_dataset(_write(tmp_path, [VALID_ROW, row]))

    def test_filename_version_mismatch(self, tmp_path):
        with pytest.raises(DatasetError, match="declares version"):
            load_dataset(_write(tmp_path, [VALID_ROW], name="intent_dataset.v2.jsonl"))

    def test_empty_dataset(self, tmp_path):
        path = tmp_path / "intent_dataset.v1.jsonl"
        path.write_text("\n\n", encoding="utf-8")
        with pytest.raises(DatasetError, match="empty"):
            load_dataset(path)

    def test_empty_text_rejected(self, tmp_path):
        row = dict(VALID_ROW, id="x-0001", text="   ")
        with pytest.raises(DatasetError, match="invalid text length"):
            load_dataset(_write(tmp_path, [row]))

    def test_too_long_text_rejected(self, tmp_path):
        row = dict(VALID_ROW, id="x-0001", text="x" * 301)
        with pytest.raises(DatasetError, match="invalid text length"):
            load_dataset(_write(tmp_path, [row]))
