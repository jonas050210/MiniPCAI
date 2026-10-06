"""Tests for the model wrapper: training artifacts, prediction, persistence."""

from __future__ import annotations

import json

import pytest

from minipcai.intents import ALL_LABELS
from minipcai.model import ModelError, SklearnIntentClassifier, build_estimator


class TestTrainedModel:
    def test_labels_cover_all_intents(self, trained_model):
        model = SklearnIntentClassifier.load(trained_model)
        assert set(model.labels) == set(ALL_LABELS)

    def test_predict_known_texts(self, trained_model):
        model = SklearnIntentClassifier.load(trained_model)
        assert model.predict("öffne notepad").label == "open_app"
        assert model.predict("wie wird das wetter morgen").label == "unknown"
        assert model.predict("was ist 12*4").label == "calc"
        # Typo tolerance through char n-grams.
        assert model.predict("öfne notepad").label == "open_app"

    def test_prediction_is_well_formed(self, trained_model):
        model = SklearnIntentClassifier.load(trained_model)
        prediction = model.predict("stelle einen timer auf 5 minuten")
        assert prediction.label == "timer"
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.ranked) == len(ALL_LABELS)
        probabilities = [prob for _, prob in prediction.ranked]
        assert probabilities == sorted(probabilities, reverse=True)
        assert sum(probabilities) == pytest.approx(1.0, abs=1e-6)

    def test_metadata_contains_required_fields(self, trained_model):
        model = SklearnIntentClassifier.load(trained_model)
        metadata = model.metadata
        assert metadata["dataset"]["version"] == 1
        assert metadata["dataset"]["n_examples"] > 700
        assert len(metadata["dataset"]["sha256"]) == 64
        assert 0 < metadata["thresholds"]["min_confidence"] < 1
        assert 0 < metadata["thresholds"]["min_margin"] < 1
        assert metadata["metrics_summary"]["test_accuracy"] > 0.85
        assert metadata["sklearn_version"]

    def test_metadata_and_metrics_files_written(self, trained_model):
        for name in ("metadata.json", "metrics.json"):
            path = trained_model.parent / name
            assert path.is_file(), f"{name} missing"
            data = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)


class TestPersistence:
    def test_save_load_roundtrip(self, trained_model, tmp_path):
        model = SklearnIntentClassifier.load(trained_model)
        target = tmp_path / "roundtrip.joblib"
        model.save(target, metadata={"foo": "bar"})
        loaded = SklearnIntentClassifier.load(target)
        assert loaded.labels == model.labels
        for text in ("öffne notepad", "was ist 1+1", "hallo"):
            assert loaded.predict(text).label == model.predict(text).label
        assert loaded.metadata["foo"] == "bar"

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(ModelError, match="minipcai-train"):
            SklearnIntentClassifier.load(tmp_path / "missing.joblib")

    def test_load_foreign_artifact(self, tmp_path):
        import joblib

        path = tmp_path / "foreign.joblib"
        joblib.dump({"something": "else"}, path)
        with pytest.raises(ModelError, match="unsupported model artifact"):
            SklearnIntentClassifier.load(path)


class TestEstimator:
    def test_build_estimator_pipeline(self):
        estimator = build_estimator()
        names = [name for name, _ in estimator.steps]
        assert names == ["features", "classifier"]
