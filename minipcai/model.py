"""Intent classifier abstraction plus the sklearn implementation.

The pipeline only depends on the small :class:`IntentModel` interface, so the
trained sklearn model can later be replaced by any other implementation
(different library, retrained data, a service, ...) without touching the rest
of the code base.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from minipcai.textutils import normalize_for_model

MODEL_FORMAT = "minipcai-sklearn-v1"


@dataclass(frozen=True)
class Prediction:
    """A classified request with the full ranked probability distribution."""

    label: str
    confidence: float
    ranked: tuple[tuple[str, float], ...]  # sorted by probability, descending


class IntentModel(Protocol):
    """Interface every intent model must implement."""

    @property
    def labels(self) -> tuple[str, ...]: ...

    def predict(self, text: str) -> Prediction: ...


class ModelError(RuntimeError):
    """Raised when a model artifact is missing or incompatible."""


def build_estimator() -> Pipeline:
    """Build the lightweight CPU-friendly sklearn pipeline.

    Word unigrams/bigrams capture vocabulary, character n-grams provide
    robustness against typos and German morphology. LogisticRegression keeps
    training well under a second on CPU for a dataset of this size and
    provides calibrated-enough probabilities for the confidence gates.

    Both vectorizers share the :func:`~minipcai.textutils.normalize_for_model`
    preprocessor, so casing and punctuation variants of a request produce the
    same features while arithmetic operators survive for calculator requests.
    """
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=1,
                    preprocessor=normalize_for_model,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 4),
                    sublinear_tf=True,
                    lowercase=True,
                    min_df=1,
                    preprocessor=normalize_for_model,
                ),
            ),
        ]
    )
    classifier = LogisticRegression(C=10.0, max_iter=3000, solver="lbfgs")
    return Pipeline([("features", features), ("classifier", classifier)])


class SklearnIntentClassifier:
    """Trained sklearn model wrapped behind the :class:`IntentModel` interface."""

    def __init__(self, estimator: Pipeline, labels: tuple[str, ...],
                 metadata: dict[str, Any] | None = None):
        self._estimator = estimator
        self._labels = tuple(labels)
        self.metadata = metadata or {}

    @property
    def labels(self) -> tuple[str, ...]:
        return self._labels

    def predict(self, text: str) -> Prediction:
        probabilities = self._estimator.predict_proba([text])[0]
        ranked = sorted(
            zip(self._labels, probabilities.tolist(), strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        label, confidence = ranked[0]
        return Prediction(label=label, confidence=confidence, ranked=tuple(ranked))

    # -- persistence ----------------------------------------------------------
    def save(self, path: Path | str, metadata: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": MODEL_FORMAT,
            "labels": list(self._labels),
            "metadata": metadata if metadata is not None else self.metadata,
            "estimator": self._estimator,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: Path | str) -> SklearnIntentClassifier:
        path = Path(path)
        if not path.is_file():
            raise ModelError(
                f"trained model not found at {path}. Run 'minipcai-train' first."
            )
        try:
            payload = joblib.load(path)
        except Exception as exc:  # corrupt/truncated artifact or foreign pickle
            raise ModelError(f"could not read model artifact at {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != MODEL_FORMAT:
            raise ModelError(f"unsupported model artifact at {path}")
        labels = payload.get("labels")
        if (
            not isinstance(labels, list)
            or not labels
            or not all(isinstance(label, str) and label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ModelError("model artifact declares no valid, unique labels")
        if "estimator" not in payload:
            raise ModelError("model artifact contains no estimator")
        return cls(
            estimator=payload["estimator"],
            labels=tuple(labels),
            metadata=payload.get("metadata") or {},
        )


def write_metadata_files(models_dir: Path, metadata: dict[str, Any],
                         metrics: dict[str, Any]) -> None:
    """Write ``metadata.json`` and ``metrics.json`` next to the model artifact."""
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (models_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
