"""Evaluation helpers that go beyond a random stratified split.

Two questions are answered here:

1. **How much does near-duplicate leakage inflate the headline metrics?**
   The dataset contains many typo/paraphrase variants of the same utterance.
   A random split therefore puts near-identical rows into train and test.
   :func:`near_duplicate_groups` clusters examples that are within a small
   Damerau-Levenshtein distance of each other (same label), and
   :func:`grouped_holdout` evaluates with cluster-disjoint folds, which is the
   honest number.

2. **Do the dataset and the registry agree?** A model that is trained on
   ``"starte word"`` but whose registry has no ``word`` entry will classify
   confidently and then reject - the user experiences that as "it does not
   understand me". :func:`registry_resolvability` measures that gap so it
   cannot grow unnoticed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from minipcai.config import Thresholds
from minipcai.dataset import Dataset
from minipcai.intents import TARGETED_INTENTS
from minipcai.model import SklearnIntentClassifier, build_estimator
from minipcai.registry import Registry
from minipcai.targets import TargetError, resolve_target
from minipcai.textutils import damerau_levenshtein, normalize


@dataclass(frozen=True)
class ResolvabilityReport:
    resolved: int
    total: int
    failures: dict[str, int]  # "intent/reason" -> count

    @property
    def ratio(self) -> float:
        return self.resolved / self.total if self.total else 1.0

    def to_dict(self) -> dict:
        return {
            "resolved": self.resolved,
            "total": self.total,
            "ratio": round(self.ratio, 4),
            "failures": self.failures,
        }


def near_duplicate_groups(
    texts: Iterable[str],
    labels: Iterable[str],
    max_distance: int = 2,
    max_neighbors: int = 15,
) -> list[int]:
    """Cluster near-duplicate examples of the same label.

    A blocked candidate search (cosine similarity over character n-grams)
    keeps this cheap; the actual decision uses the Damerau-Levenshtein distance.
    Returns one group id per input row.
    """
    texts = list(texts)
    labels = list(labels)
    normalized = [normalize(text) for text in texts]
    if not texts:
        return []
    parent = list(range(len(texts)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=1)
    matrix = vectorizer.fit_transform(normalized)
    neighbors = NearestNeighbors(
        n_neighbors=min(max_neighbors, len(texts)), metric="cosine"
    ).fit(matrix)
    _, indices = neighbors.kneighbors(matrix)
    for index, row in enumerate(indices):
        for other in row[1:]:
            other = int(other)
            if labels[index] != labels[other]:
                continue
            left, right = normalized[index], normalized[other]
            if abs(len(left) - len(right)) > max_distance:
                continue
            if damerau_levenshtein(left, right) <= max_distance:
                union(index, other)
    return [find(index) for index in range(len(texts))]


def grouped_holdout(
    dataset: Dataset,
    thresholds: Thresholds | None = None,
    folds: int = 5,
    max_distance: int = 2,
) -> dict:
    """Cross-validate with cluster-disjoint folds (honest generalization)."""
    texts = list(dataset.texts)
    labels = list(dataset.labels)
    groups = near_duplicate_groups(texts, labels, max_distance=max_distance)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    accuracies: list[float] = []
    macro_f1: list[float] = []
    accepted = 0
    total = 0
    thresholds = thresholds or Thresholds()
    for train_index, test_index in splitter.split(texts, labels, groups):
        estimator = build_estimator()
        estimator.fit(
            [texts[i] for i in train_index], [labels[i] for i in train_index]
        )
        classifier = SklearnIntentClassifier(
            estimator=estimator, labels=tuple(estimator.classes_)
        )
        predicted = [classifier.predict(texts[i]).label for i in test_index]
        truth = [labels[i] for i in test_index]
        accuracies.append(float(accuracy_score(truth, predicted)))
        macro_f1.append(
            float(f1_score(truth, predicted, average="macro", zero_division=0))
        )
        for index in test_index:
            top_label, confidence = classifier.predict(texts[index]).ranked[0]
            second = (
                classifier.predict(texts[index]).ranked[1][1]
                if len(classifier.predict(texts[index]).ranked) > 1
                else 0.0
            )
            if (
                top_label != "unknown"
                and confidence >= thresholds.min_confidence
                and confidence - second >= thresholds.min_margin
            ):
                accepted += 1
            total += 1
    return {
        "folds": len(accuracies),
        "n": len(texts),
        "groups": len(set(groups)),
        "accuracy": round(sum(accuracies) / len(accuracies), 4),
        "macro_f1": round(sum(macro_f1) / len(macro_f1), 4),
        "accepted_ratio": round(accepted / total, 4) if total else 0.0,
        "max_distance": max_distance,
    }


def registry_resolvability(dataset: Dataset, registry: Registry) -> ResolvabilityReport:
    """How many targeted `normal` examples resolve to a real registry target."""
    resolved = 0
    total = 0
    failures: Counter[str] = Counter()
    for example in dataset.examples:
        if example.label not in TARGETED_INTENTS or example.category != "normal":
            continue
        total += 1
        try:
            resolve_target(example.label, example.text, registry)
            resolved += 1
        except TargetError as exc:
            failures[f"{example.label}/{exc.reason}"] += 1
    return ResolvabilityReport(
        resolved=resolved, total=total, failures=dict(failures.most_common())
    )
