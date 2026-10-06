"""Default configuration and paths for MiniPCAI."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

# Resolve paths relative to the repository checkout. The MVP is designed to be
# run from a clone (`pip install -e .`), so data lives next to the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

_DATASET_NAME_RE = re.compile(r"^intent_dataset\.v(\d+)\.jsonl$")


def discover_dataset_path(data_dir: Path | str | None = None) -> Path:
    """Return the highest-versioned dataset file in ``data_dir``.

    Datasets are versioned through their file name (``intent_dataset.vN.jsonl``)
    so that publishing a new dataset version does not require a code change.
    Falls back to the v1 path when no versioned file exists; the loader then
    reports the missing file with a clear error.
    """
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    try:
        candidates = list(directory.glob("intent_dataset.v*.jsonl"))
    except OSError:  # unreadable directory: fall back to the default name
        candidates = []
    best: tuple[int, Path] | None = None
    for candidate in candidates:
        match = _DATASET_NAME_RE.match(candidate.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), candidate)
    return best[1] if best is not None else DATA_DIR / "intent_dataset.v1.jsonl"


DEFAULT_DATASET_PATH = discover_dataset_path()
DEFAULT_REGISTRY_PATH = DATA_DIR / "registry.json"
DEFAULT_MODEL_PATH = MODELS_DIR / "model.joblib"

# Audit log location: outside the repository, per user.
DEFAULT_AUDIT_PATH = Path.home() / ".minipcai" / "audit.jsonl"

# Input limits.
MAX_INPUT_LENGTH = 300

# Timer bounds (seconds). One second up to 24 hours.
TIMER_MIN_SECONDS = 1
TIMER_MAX_SECONDS = 24 * 60 * 60

# find_file limits.
FIND_FILE_MAX_RESULTS = 20
FIND_FILE_MAX_VISITED = 20_000

# Calc limits.
CALC_MAX_EXPRESSION_LENGTH = 120
CALC_MAX_NODES = 64
CALC_MAX_LITERAL = 1e15
CALC_MAX_EXPONENT = 1000
# Results with more decimal digits than this are rejected instead of being
# converted to a huge string (the conversion itself is quadratic in CPython).
CALC_MAX_RESULT_DIGITS = 1000


@dataclass(frozen=True)
class Thresholds:
    """Confidence gates applied to the classifier prediction.

    A request is only executed when the top intent is not ``unknown``, its
    probability is at least ``min_confidence`` and it beats the runner-up by at
    least ``min_margin``. Both values are calibrated during training and stored
    in the model metadata.
    """

    min_confidence: float = 0.50
    min_margin: float = 0.15


EXECUTOR_MODES: tuple[str, ...] = ("dry-run", "windows")


@dataclass
class Config:
    """Bundle of all configurable paths and runtime options."""

    model_path: Path = DEFAULT_MODEL_PATH
    registry_path: Path = DEFAULT_REGISTRY_PATH
    dataset_path: Path = DEFAULT_DATASET_PATH
    audit_path: Path = DEFAULT_AUDIT_PATH
    models_dir: Path = MODELS_DIR
    executor_mode: str = "dry-run"  # "dry-run" | "windows"
    thresholds: Thresholds = field(default_factory=Thresholds)

    def __post_init__(self) -> None:
        if self.executor_mode not in EXECUTOR_MODES:
            raise ValueError(
                f"Unknown executor mode: {self.executor_mode!r} "
                f"(expected one of {', '.join(EXECUTOR_MODES)})"
            )


def default_config() -> Config:
    """Return the default configuration."""
    return Config()
