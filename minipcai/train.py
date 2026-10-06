"""Training, calibration and evaluation of the intent classifier.

Produces three artifacts in ``models/``:

* ``model.joblib``  - the trained sklearn pipeline (not committed to git),
* ``metadata.json`` - model/dataset/threshold metadata (committed),
* ``metrics.json``  - full evaluation report (committed).

Design notes:

* stratified 70/15/15 train/validation/test split with a fixed seed;
* thresholds (``min_confidence``, ``min_margin``) are calibrated on the
  validation split by maximizing accepted-and-correct decisions while heavily
  penalizing accepted-but-wrong ones;
* the test split is evaluated once, with per-class precision/recall/F1, a
  confusion matrix and per-category (normal/typo/unknown/ambiguous) decision
  statistics, including the safety-relevant "accepted although the label is
  unknown" rate.
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedShuffleSplit

from minipcai.config import DEFAULT_DATASET_PATH, MODELS_DIR, Thresholds
from minipcai.dataset import Dataset, load_dataset
from minipcai.model import (
    SklearnIntentClassifier,
    build_estimator,
    write_metadata_files,
)

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

_CONFIDENCE_GRID = (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
_MARGIN_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
# Safety bounds for threshold selection on the validation split: no wrongly
# accepted request with a valid label at all, and at most one unknown/
# ambiguous example accepted (such accepts remain runtime-safe because target
# resolution rejects target-less requests, but the gates should stay cautious).
_MAX_WRONG_ACCEPTS = 0
_MAX_UNKNOWN_ACCEPTS = 1
# Fallback penalties (used only when no threshold pair satisfies the bounds).
_WRONG_ACCEPT_PENALTY = 10.0
_UNKNOWN_ACCEPT_PENALTY = 25.0


def split_dataset(
    dataset: Dataset, seed: int = SEED
) -> dict[str, list[tuple[str, str, str]]]:
    """Stratified split into train/val/test; returns (text, label, category) lists."""
    texts = list(dataset.texts)
    labels = list(dataset.labels)
    categories = list(dataset.categories)

    first = StratifiedShuffleSplit(n_splits=1, test_size=VAL_FRACTION + TEST_FRACTION,
                                   random_state=seed)
    train_idx, rest_idx = next(first.split(texts, labels))
    rest = list(rest_idx)
    second = StratifiedShuffleSplit(
        n_splits=1,
        test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        random_state=seed + 1,
    )
    val_pos, test_pos = next(second.split([texts[i] for i in rest],
                                          [labels[i] for i in rest]))

    def pack(indices: list[int]) -> list[tuple[str, str, str]]:
        return [(texts[i], labels[i], categories[i]) for i in indices]

    return {
        "train": pack(train_idx),
        "val": pack([rest[i] for i in val_pos]),
        "test": pack([rest[i] for i in test_pos]),
    }


def calibrate_thresholds(
    classifier: SklearnIntentClassifier,
    val_split: list[tuple[str, str, str]],
) -> tuple[Thresholds, list[dict[str, Any]]]:
    """Grid-search confidence/margin thresholds on the validation split."""
    predictions = [classifier.predict(text) for text, _, _ in val_split]
    table: list[dict[str, Any]] = []
    for min_confidence in _CONFIDENCE_GRID:
        for min_margin in _MARGIN_GRID:
            accepted_correct = 0
            accepted_wrong = 0
            accepted_unknown = 0
            rejected = 0
            for prediction, (_, label, _) in zip(predictions, val_split, strict=True):
                top_label, top_confidence = prediction.ranked[0]
                second_confidence = (
                    prediction.ranked[1][1] if len(prediction.ranked) > 1 else 0.0
                )
                accepts = (
                    top_label != "unknown"
                    and top_confidence >= min_confidence
                    and (top_confidence - second_confidence) >= min_margin
                )
                if not accepts:
                    rejected += 1
                elif label == "unknown":
                    # an unknown/ambiguous example slipped through the gates
                    accepted_unknown += 1
                elif top_label == label:
                    accepted_correct += 1
                else:
                    accepted_wrong += 1
            n = len(val_split)
            score = (
                accepted_correct
                - _WRONG_ACCEPT_PENALTY * accepted_wrong
                - _UNKNOWN_ACCEPT_PENALTY * accepted_unknown
            ) / n
            table.append(
                {
                    "min_confidence": min_confidence,
                    "min_margin": min_margin,
                    "accepted_correct": accepted_correct,
                    "accepted_wrong": accepted_wrong,
                    "accepted_unknown": accepted_unknown,
                    "rejected": rejected,
                    "score": round(score, 4),
                }
            )
    # Choose the most permissive thresholds that respect the safety bounds on
    # the validation split; fall back to the penalty score when nothing
    # qualifies. Ties are broken towards stricter gates.
    eligible = [
        row
        for row in table
        if row["accepted_wrong"] <= _MAX_WRONG_ACCEPTS
        and row["accepted_unknown"] <= _MAX_UNKNOWN_ACCEPTS
    ]
    pool = eligible or table
    best = max(
        pool,
        key=lambda row: (
            row["accepted_correct"] if eligible else row["score"],
            row["min_confidence"],
            row["min_margin"],
        ),
    )
    thresholds = Thresholds(
        min_confidence=best["min_confidence"], min_margin=best["min_margin"]
    )
    return thresholds, table


def evaluate_split(
    classifier: SklearnIntentClassifier,
    split: list[tuple[str, str, str]],
    thresholds: Thresholds,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    """Full evaluation of one split: classification metrics + decision gates."""
    predictions = [classifier.predict(text) for text, _, _ in split]
    predicted_labels = [prediction.label for prediction in predictions]
    true_labels = [label for _, label, _ in split]

    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, labels=list(labels), zero_division=0
    )
    per_class = {
        label: {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f), 4),
            "support": int(s),
        }
        for label, p, r, f, s in zip(labels, precision, recall, f1, support, strict=True)
    }
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(labels))
    confusion = {
        labels[i]: {labels[j]: int(matrix[i][j]) for j in range(len(labels))}
        for i in range(len(labels))
    }

    # Decision statistics under the runtime gates.
    category_stats: dict[str, dict[str, int]] = {}
    for category in ("normal", "typo", "unknown", "ambiguous"):
        category_stats[category] = {
            "n": 0, "accepted_correct": 0, "accepted_wrong": 0, "rejected": 0,
        }
    for prediction, (_, label, category) in zip(predictions, split, strict=True):
        top_label, top_confidence = prediction.ranked[0]
        second_confidence = (
            prediction.ranked[1][1] if len(prediction.ranked) > 1 else 0.0
        )
        accepts = (
            top_label != "unknown"
            and top_confidence >= thresholds.min_confidence
            and (top_confidence - second_confidence) >= thresholds.min_margin
        )
        stats = category_stats[category]
        stats["n"] += 1
        if not accepts:
            stats["rejected"] += 1
        elif top_label == label:
            stats["accepted_correct"] += 1
        else:
            stats["accepted_wrong"] += 1

    return {
        "n": len(split),
        "accuracy": round(float(accuracy_score(true_labels, predicted_labels)), 4),
        "macro_f1": round(
            float(f1_score(true_labels, predicted_labels, labels=list(labels),
                           zero_division=0, average="macro")), 4
        ),
        "per_class": per_class,
        "confusion": confusion,
        "category_decisions": category_stats,
    }


def train(
    dataset_path: Path | str = DEFAULT_DATASET_PATH,
    models_dir: Path | str = MODELS_DIR,
    seed: int = SEED,
    quiet: bool = False,
) -> dict[str, Any]:
    """Train, calibrate, evaluate and persist the model; returns the metrics."""
    dataset = load_dataset(dataset_path)
    splits = split_dataset(dataset, seed=seed)

    train_texts = [text for text, _, _ in splits["train"]]
    train_labels = [label for _, label, _ in splits["train"]]
    estimator = build_estimator()
    estimator.fit(train_texts, train_labels)
    classifier = SklearnIntentClassifier(
        estimator=estimator, labels=tuple(estimator.classes_)
    )

    thresholds, calibration_table = calibrate_thresholds(classifier, splits["val"])
    val_metrics = evaluate_split(classifier, splits["val"], thresholds, classifier.labels)
    test_metrics = evaluate_split(classifier, splits["test"], thresholds, classifier.labels)

    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    metadata: dict[str, Any] = {
        "model_format": "minipcai-sklearn-v1",
        "created_at": created_at,
        "trained_on": platform.platform(),
        "python_version": sys.version.split()[0],
        "sklearn_version": sklearn.__version__,
        "dataset": {
            "path": str(dataset.path),
            "version": dataset.version,
            "sha256": dataset.sha256,
            "n_examples": len(dataset.examples),
            "label_counts": dataset.label_counts(),
            "category_counts": dataset.category_counts(),
        },
        "labels": list(classifier.labels),
        "feature_config": {
            "word": {"analyzer": "word", "ngram_range": [1, 2], "sublinear_tf": True},
            "char": {"analyzer": "char_wb", "ngram_range": [2, 4], "sublinear_tf": True},
        },
        "classifier": {"type": "LogisticRegression", "C": 5.0, "max_iter": 3000},
        "split": {
            "seed": seed,
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "thresholds": {
            "min_confidence": thresholds.min_confidence,
            "min_margin": thresholds.min_margin,
        },
        "metrics_summary": {
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
        },
    }

    models_dir = Path(models_dir)
    classifier.save(models_dir / "model.joblib", metadata=metadata)
    metrics: dict[str, Any] = {
        "created_at": created_at,
        "dataset": metadata["dataset"],
        "split": metadata["split"],
        "thresholds": metadata["thresholds"],
        "threshold_calibration": {
            "grids": {
                "min_confidence": list(_CONFIDENCE_GRID),
                "min_margin": list(_MARGIN_GRID),
                "wrong_accept_penalty": _WRONG_ACCEPT_PENALTY,
                "unknown_accept_penalty": _UNKNOWN_ACCEPT_PENALTY,
            },
            "chosen": {
                "min_confidence": thresholds.min_confidence,
                "min_margin": thresholds.min_margin,
            },
            "top_rows": sorted(calibration_table, key=lambda row: -row["score"])[:10],
        },
        "validation": val_metrics,
        "test": test_metrics,
    }
    write_metadata_files(models_dir, metadata, metrics)

    if not quiet:
        _print_report(dataset, splits, thresholds, val_metrics, test_metrics)
    return metrics


def _print_report(
    dataset: Dataset,
    splits: dict[str, list[tuple[str, str, str]]],
    thresholds: Thresholds,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> None:
    print("=" * 62)
    print("MiniPCAI intent model - training report")
    print("=" * 62)
    print(f"dataset        : {dataset.path} (v{dataset.version}, {len(dataset.examples)} examples)")
    print(
        f"split          : train={len(splits['train'])} "
        f"val={len(splits['val'])} test={len(splits['test'])}"
    )
    print(f"thresholds     : min_confidence={thresholds.min_confidence} "
          f"min_margin={thresholds.min_margin}")
    print()
    print(f"validation     : accuracy={val_metrics['accuracy']:.4f} "
          f"macro_f1={val_metrics['macro_f1']:.4f}")
    print(f"test           : accuracy={test_metrics['accuracy']:.4f} "
          f"macro_f1={test_metrics['macro_f1']:.4f}")
    print()
    print("test decisions by example category (gates applied):")
    print(f"{'category':<12}{'n':>5}{'ok':>6}{'wrong':>7}{'rejected':>10}")
    for category, stats in test_metrics["category_decisions"].items():
        print(
            f"{category:<12}{stats['n']:>5}{stats['accepted_correct']:>6}"
            f"{stats['accepted_wrong']:>7}{stats['rejected']:>10}"
        )
    print()
    print("test per-class F1:")
    for label, stats in test_metrics["per_class"].items():
        print(f"  {label:<14} f1={stats['f1']:.3f}  support={stats['support']}")
    print()
    print("Artifacts written to models/: model.joblib, metadata.json, metrics.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minipcai-train",
        description="Train the MiniPCAI intent classifier.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    train(dataset_path=args.dataset, models_dir=args.models_dir, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
