"""Tests for the training/calibration/evaluation tooling."""

from __future__ import annotations

import pytest

from minipcai.config import DEFAULT_DATASET_PATH, Thresholds
from minipcai.dataset import load_dataset
from minipcai.train import calibrate_thresholds, evaluate_split, safety_summary, split_dataset


@pytest.fixture(scope="module")
def trained(tmp_path_factory: pytest.TempPathFactory):
    """Train a real model once and return (classifier, thresholds, splits)."""
    from minipcai.model import SklearnIntentClassifier, build_estimator

    dataset = load_dataset(DEFAULT_DATASET_PATH)
    splits = split_dataset(dataset, seed=42)
    estimator = build_estimator()
    estimator.fit(
        [text for text, _, _ in splits["train"]],
        [label for _, label, _ in splits["train"]],
    )
    classifier = SklearnIntentClassifier(
        estimator=estimator, labels=tuple(estimator.classes_)
    )
    thresholds, table = calibrate_thresholds(classifier, splits["val"])
    return classifier, thresholds, splits, table


class TestSplits:
    def test_splits_are_disjoint_and_complete(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        splits = split_dataset(dataset, seed=42)
        texts = [text for text, _, _ in splits["train"]]
        texts += [text for text, _, _ in splits["val"]]
        texts += [text for text, _, _ in splits["test"]]
        assert len(texts) == len(dataset.examples)
        assert len(set(texts)) == len(texts)

    def test_splits_are_deterministic(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        assert split_dataset(dataset, seed=42) == split_dataset(dataset, seed=42)

    def test_every_label_appears_in_every_split(self):
        dataset = load_dataset(DEFAULT_DATASET_PATH)
        splits = split_dataset(dataset, seed=42)
        for name, rows in splits.items():
            labels = {label for _, label, _ in rows}
            assert labels == set(dataset.label_counts()), name


class TestCalibration:
    def test_returns_thresholds_and_full_table(self, trained):
        from minipcai.train import _CONFIDENCE_GRID, _MARGIN_GRID

        _, thresholds, _, table = trained
        assert isinstance(thresholds, Thresholds)
        assert 0.0 < thresholds.min_confidence < 1.0
        assert 0.0 < thresholds.min_margin < 1.0
        assert len(table) == len(_CONFIDENCE_GRID) * len(_MARGIN_GRID)

    def test_chosen_thresholds_are_safe_on_the_guard_split(self, trained):
        """The pair is vetoed when it accepts a wrong intent on the guard split."""
        _, thresholds, splits, table = trained
        chosen = [
            row for row in table
            if row["min_confidence"] == thresholds.min_confidence
            and row["min_margin"] == thresholds.min_margin
        ][0]
        assert chosen["accepted_wrong"] == 0
        assert chosen["guard_accepted_wrong"] == 0
        assert all("guard_accepted_wrong" in row for row in table)

    def test_chosen_thresholds_are_safe_on_validation(self, trained):
        classifier, thresholds, splits, _ = trained
        metrics = evaluate_split(classifier, splits["val"], thresholds, classifier.labels)
        safety = safety_summary(metrics)
        # The calibrated gates must never accept a wrongly labeled in-scope
        # request on the validation split.
        assert safety["accepted_wrong_in_scope"] == 0

    def test_safety_summary_counts(self):
        metrics = {
            "n": 4,
            "category_decisions": {
                "normal": {"n": 1, "accepted_correct": 1, "accepted_wrong": 1,
                           "rejected": 0},
                "typo": {"n": 1, "accepted_correct": 0, "accepted_wrong": 0,
                         "rejected": 1},
                "unknown": {"n": 1, "accepted_correct": 0, "accepted_wrong": 1,
                            "rejected": 0},
                "ambiguous": {"n": 1, "accepted_correct": 0, "accepted_wrong": 1,
                              "rejected": 0},
            },
        }
        summary = safety_summary(metrics)
        assert summary == {
            "accepted_wrong_in_scope": 1,
            "accepted_out_of_scope": 2,
            "rejected": 1,
            "n": 4,
        }


class TestTrainArtifacts:
    def test_train_writes_safety_blocks(self, tmp_path):
        from minipcai.train import train

        metrics = train(dataset_path=DEFAULT_DATASET_PATH, models_dir=tmp_path,
                        quiet=True)
        assert set(metrics["safety"]) == {"validation", "test"}
        assert metrics["safety"]["test"]["accepted_wrong_in_scope"] == 0
        assert metrics["safety"]["test"]["n"] > 0

    def test_metrics_contain_per_class_and_confusion(self, tmp_path):
        from minipcai.train import train

        metrics = train(dataset_path=DEFAULT_DATASET_PATH, models_dir=tmp_path,
                        quiet=True)
        per_class = metrics["test"]["per_class"]
        assert set(per_class) == set(metrics["dataset"]["label_counts"])
        for stats in per_class.values():
            assert 0.0 <= stats["precision"] <= 1.0
            assert 0.0 <= stats["recall"] <= 1.0
        confusion = metrics["test"]["confusion"]
        assert sum(sum(row.values()) for row in confusion.values()) == metrics["test"]["n"]

    def test_committed_metadata_matches_committed_metrics(self):
        import json

        from minipcai.config import MODELS_DIR

        metadata = json.loads((MODELS_DIR / "metadata.json").read_text(encoding="utf-8"))
        metrics = json.loads((MODELS_DIR / "metrics.json").read_text(encoding="utf-8"))
        assert metadata["thresholds"] == metrics["thresholds"]
        assert metadata["metrics_summary"]["test_accuracy"] == metrics["test"]["accuracy"]
        assert metadata["safety_summary"]["test_accepted_wrong_in_scope"] == 0


class TestRecordedPaths:
    """Committed metadata must not leak machine-specific absolute paths."""

    def test_repo_paths_are_recorded_relative(self):
        from minipcai.train import _REPO_ROOT, _portable_path

        assert _portable_path(_REPO_ROOT / "minipcai" / "data" / "registry.json") == \
            "minipcai/data/registry.json"

    def test_outside_paths_are_recorded_absolute(self, tmp_path):
        from minipcai.train import _portable_path

        outside = tmp_path / "somewhere.json"
        assert _portable_path(outside) == str(outside.resolve())

    def test_committed_metadata_uses_relative_paths(self):
        import json
        from pathlib import Path

        metadata = json.loads(
            (Path(__file__).resolve().parent.parent / "models" / "metadata.json")
            .read_text(encoding="utf-8")
        )
        assert not metadata["dataset"]["path"].startswith("/")
        assert not metadata["registry"]["path"].startswith("/")
