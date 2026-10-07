"""Training, calibration and evaluation of the intent classifier.

Produces three artifacts in ``models/``:

* ``model.joblib``  - the trained sklearn pipeline (not committed to git),
* ``metadata.json`` - model/dataset/threshold metadata (committed),
* ``metrics.json``  - full evaluation report (committed).

Design notes:

* stratified 70/15/15 train/validation/test split with a fixed seed;
* thresholds (``min_confidence``, ``min_margin``) are calibrated on the
  validation split. Grid pairs that satisfy the preferred safety bounds (no
  wrongly accepted in-scope request, at most one unknown/ambiguous accept) are
  preferred, taking the one with the most accepted-and-correct decisions. When
  no pair satisfies the bounds - which is the case for the v2 validation split,
  where the lowest reachable ``accepted_unknown`` is 4 - selection falls back
  to the penalty score, which heavily weights accepted-but-wrong and
  accepted-unknown decisions. The chosen row, including its
  ``accepted_unknown`` count, is recorded in ``metrics.json``;
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

from minipcai.config import Thresholds
from minipcai.dataset import Dataset, load_dataset
from minipcai.evaluate import grouped_holdout, registry_resolvability
from minipcai.model import (
    SklearnIntentClassifier,
    build_estimator,
    write_metadata_files,
)
from minipcai.paths import (
    default_dataset_path,
    default_models_dir,
    packaged_registry_path,
)
from minipcai.registry import Registry, RegistryError

SEED = 42
TRAIN_FRACTION = 0.70
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

_CONFIDENCE_GRID = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75)
_MARGIN_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
# Threshold selection is a risk/coverage trade-off, not a guess:
#
#   hard rule : never accept a request whose label is a *known* intent with the
#               wrong intent (accepted_wrong == 0);
#   budget    : out-of-scope requests ("unknown"/"ambiguous" examples) may slip
#               through the gates for at most _MAX_OUT_OF_SCOPE_RATE of the
#               validation examples. Those accepts are not executed blindly:
#               target resolution, the parameter parsers and the security policy
#               still have to agree (a web search, for instance, additionally
#               requires an explicit search word), so the realistic outcome is a
#               clarifying message, not a wrong action;
#   objective : maximize the number of accepted *correct* requests, i.e. keep
#               unnecessary rejections as small as possible. Ties break towards
#               the lower (more permissive but still budget-conforming) gates.
#
# If no grid pair satisfies the hard rule, the penalty score below is used as a
# fallback and the chosen row is reported in metrics.json.
_MAX_WRONG_ACCEPTS = 0
_MAX_OUT_OF_SCOPE_RATE = 0.02
_MIN_OUT_OF_SCOPE_BUDGET = 1
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
    # Budget rule: the hard rule (no wrongly accepted labelled request) must
    # hold; within the out-of-scope budget the pair accepting the most correct
    # requests wins. Ties go to the *stricter* gates, so the most conservative
    # pair that still reaches the best coverage is chosen.
    budget = max(
        _MIN_OUT_OF_SCOPE_BUDGET, int(round(_MAX_OUT_OF_SCOPE_RATE * len(val_split)))
    )
    eligible = [
        row
        for row in table
        if row["accepted_wrong"] <= _MAX_WRONG_ACCEPTS
        and row["accepted_unknown"] <= budget
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


def safety_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    """Summarize the safety-relevant gate decisions of one split.

    ``accepted_wrong`` counts requests that were classified as an action intent
    with the wrong label and still passed the gates. For ``normal``/``typo``
    examples that is a genuine misclassification; for ``unknown``/``ambiguous``
    examples the request is out of scope and must be rejected or handled
    safely by the downstream target resolution. Both numbers are reported so
    that regressions are visible in the committed metrics.
    """
    decisions = metrics["category_decisions"]
    in_scope_wrong = sum(
        decisions[category]["accepted_wrong"]
        for category in ("normal", "typo")
    )
    out_of_scope_accepted = sum(
        decisions[category]["accepted_wrong"]
        for category in ("unknown", "ambiguous")
    )
    rejected = sum(decisions[category]["rejected"] for category in decisions)
    return {
        "accepted_wrong_in_scope": in_scope_wrong,
        "accepted_out_of_scope": out_of_scope_accepted,
        "rejected": rejected,
        "n": metrics["n"],
    }


def _registry_report(
    dataset: Dataset, registry_path: Path | str | None
) -> dict[str, Any] | None:
    """Describe the registry the model is expected to work against.

    Includes the dataset/target agreement (resolvability) so that a model that
    was trained on targets the registry does not contain shows up as a metric
    instead of as user-visible rejections.
    """
    path = Path(registry_path) if registry_path is not None else packaged_registry_path()
    if not path.is_file():
        return None
    try:
        registry = Registry.load(path)
    except RegistryError as exc:
        return {"path": str(path), "error": str(exc)}
    report = registry_resolvability(dataset, registry)
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "entries": registry.stats(),
        "resolvability": report.to_dict(),
    }


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def train(
    dataset_path: Path | str | None = None,
    models_dir: Path | str | None = None,
    seed: int = SEED,
    quiet: bool = False,
    registry_path: Path | str | None = None,
    grouped_eval: bool = False,
    promotion_gate: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Train, calibrate, evaluate and persist the model; returns the metrics.

    ``grouped_eval`` additionally evaluates with near-duplicate-disjoint folds
    (slower, honest) and ``promotion_gate`` can enforce CI thresholds - when a
    gate fails a :class:`~minipcai.train.PromotionGateError` is raised, so a
    regression cannot silently ship.
    """
    dataset_path = dataset_path or default_dataset_path()
    models_dir = models_dir or default_models_dir()
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
    val_safety = safety_summary(val_metrics)
    test_safety = safety_summary(test_metrics)

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
            "word": {
                "analyzer": "word",
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "preprocessor": "minipcai.textutils.normalize_for_model",
            },
            "char": {
                "analyzer": "char_wb",
                "ngram_range": [2, 4],
                "sublinear_tf": True,
                "preprocessor": "minipcai.textutils.normalize_for_model",
            },
        },
        "classifier": {"type": "LogisticRegression", "C": 10.0, "max_iter": 3000},
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
        "safety_summary": {
            "test_accepted_wrong_in_scope": test_safety["accepted_wrong_in_scope"],
            "test_accepted_out_of_scope": test_safety["accepted_out_of_scope"],
        },
    }

    registry_report = _registry_report(dataset, registry_path)
    if registry_report is not None:
        metadata["registry"] = registry_report

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
            "budget": {
                "max_wrong_accepts": _MAX_WRONG_ACCEPTS,
                "max_out_of_scope_rate": _MAX_OUT_OF_SCOPE_RATE,
                "out_of_scope_budget": max(
                    _MIN_OUT_OF_SCOPE_BUDGET,
                    int(round(_MAX_OUT_OF_SCOPE_RATE * len(splits["val"]))),
                ),
            },
            "chosen": {
                "min_confidence": thresholds.min_confidence,
                "min_margin": thresholds.min_margin,
            },
            "top_rows": sorted(calibration_table, key=lambda row: -row["score"])[:10],
        },
        "validation": val_metrics,
        "test": test_metrics,
        "safety": {
            "validation": val_safety,
            "test": test_safety,
        },
    }
    if registry_report is not None:
        metrics["registry"] = registry_report
    if grouped_eval:
        metrics["grouped_holdout"] = grouped_holdout(dataset, thresholds)
    if promotion_gate:
        _check_promotion_gate(metrics, promotion_gate)
    write_metadata_files(models_dir, metadata, metrics)

    if not quiet:
        _print_report(
            dataset, splits, thresholds, val_metrics, test_metrics, test_safety,
            registry_report, metrics.get("grouped_holdout"),
        )
    return metrics


class PromotionGateError(RuntimeError):
    """Raised when a trained model does not meet the configured quality gates."""


def _check_promotion_gate(metrics: dict[str, Any], gate: dict[str, float]) -> None:
    """Enforce minimum quality thresholds (used by CI)."""
    test = metrics["test"]
    safety = metrics["safety"]["test"]
    grouped = metrics.get("grouped_holdout", {})
    checks = {
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["macro_f1"],
        "grouped_accuracy": grouped.get("accuracy"),
        "resolvability_ratio": (metrics.get("registry") or {})
        .get("resolvability", {})
        .get("ratio"),
    }
    failures: list[str] = []
    for name, minimum in gate.items():
        value = checks.get(name)
        if value is None:
            continue
        if value < minimum:
            failures.append(f"{name}={value} < required {minimum}")
    if safety["accepted_wrong_in_scope"] != 0:
        failures.append(
            "accepted_wrong_in_scope="
            f"{safety['accepted_wrong_in_scope']} (must stay 0 on the test split)"
        )
    if failures:
        raise PromotionGateError("model quality gates failed: " + "; ".join(failures))


def parse_gate(spec: str | None) -> dict[str, float] | None:
    """Parse ``--gate name=value,name=value`` into a gate dictionary."""
    if not spec:
        return None
    gate: dict[str, float] = {}
    for chunk in spec.split(","):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise ValueError(f"invalid gate entry: {chunk!r} (expected name=value)")
        key, value = chunk.split("=", 1)
        gate[key.strip()] = float(value)
    return gate


def _print_report(
    dataset: Dataset,
    splits: dict[str, list[tuple[str, str, str]]],
    thresholds: Thresholds,
    val_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    test_safety: dict[str, Any],
    registry_report: dict[str, Any] | None = None,
    grouped: dict[str, Any] | None = None,
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
    safety = test_safety
    print()
    print(
        f"safety (test)  : accepted-wrong in-scope={safety['accepted_wrong_in_scope']} "
        f"out-of-scope={safety['accepted_out_of_scope']} "
        f"rejected={safety['rejected']}/{safety['n']}"
    )
    if safety["accepted_wrong_in_scope"]:
        print(
            "WARNING: the calibrated gates accepted at least one in-scope request "
            "with the wrong intent on the test split. Review the dataset/thresholds."
        )
    print()
    print("test per-class F1:")
    for label, stats in test_metrics["per_class"].items():
        print(f"  {label:<14} f1={stats['f1']:.3f}  support={stats['support']}")
    if grouped:
        print()
        print(
            f"grouped holdout (near-duplicate disjoint): accuracy={grouped['accuracy']:.4f} "
            f"macro_f1={grouped['macro_f1']:.4f} accepted={grouped['accepted_ratio']:.0%}"
        )
    if registry_report and "resolvability" in registry_report:
        report = registry_report["resolvability"]
        print()
        print(
            f"registry       : {registry_report['path']} "
            f"({registry_report['entries']['entries']} entries)"
        )
        print(
            f"dataset/target agreement: {report['resolved']}/{report['total']} "
            f"targeted examples resolve ({report['ratio']:.0%})"
        )
        for key, count in list(report["failures"].items())[:6]:
            print(f"  unresolved: {key} ({count})")
    print()
    print("Artifacts written to models/: model.joblib, metadata.json, metrics.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="minipcai-train",
        description="Train the MiniPCAI intent classifier.",
    )
    parser.add_argument("--dataset", type=Path, default=default_dataset_path())
    parser.add_argument("--models-dir", type=Path, default=default_models_dir())
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--registry", type=Path, default=packaged_registry_path(),
                        help="registry to measure dataset/target agreement against")
    parser.add_argument("--grouped-eval", action="store_true",
                        help="also evaluate with near-duplicate-disjoint folds (slower)")
    parser.add_argument("--gate", default=None,
                        help="fail when quality drops below name=value,name=value")
    args = parser.parse_args(argv)
    try:
        train(
            dataset_path=args.dataset,
            models_dir=args.models_dir,
            seed=args.seed,
            registry_path=args.registry,
            grouped_eval=args.grouped_eval,
            promotion_gate=parse_gate(args.gate),
        )
    except PromotionGateError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
