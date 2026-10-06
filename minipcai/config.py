"""Default configuration and paths for MiniPCAI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Resolve paths relative to the repository checkout. The MVP is designed to be
# run from a clone (`pip install -e .`), so data lives next to the package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

DEFAULT_DATASET_PATH = DATA_DIR / "intent_dataset.v1.jsonl"
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


def default_config() -> Config:
    """Return the default configuration."""
    return Config()
