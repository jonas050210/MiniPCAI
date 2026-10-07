"""Paths, limits and thresholds for MiniPCAI.

Path resolution is delegated to :mod:`minipcai.paths`, which keeps a source
checkout working exactly as before while making an installed wheel fully
functional (packaged defaults + writable per-user state).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from minipcai import paths
from minipcai.paths import (
    PACKAGED_DATA_DIR,
    REPO_ROOT,
    discover_dataset_path,
    is_source_checkout,
)
from minipcai.policy import SecurityPolicy, default_policy, policy_from_env

PROJECT_ROOT = REPO_ROOT
DATA_DIR = PACKAGED_DATA_DIR
MODELS_DIR = paths.default_models_dir()

DEFAULT_DATASET_PATH = paths.default_dataset_path()
DEFAULT_REGISTRY_PATH = paths.packaged_registry_path()
DEFAULT_MODEL_PATH = MODELS_DIR / paths.MODEL_FILENAME
DEFAULT_AUDIT_PATH = paths.default_audit_path()

__all__ = [
    "CALC_MAX_EXPRESSION_LENGTH",
    "CALC_MAX_LITERAL",
    "CALC_MAX_NODES",
    "CALC_MAX_RESULT_DIGITS",
    "DATA_DIR",
    "DEFAULT_AUDIT_PATH",
    "DEFAULT_DATASET_PATH",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_REGISTRY_PATH",
    "EXECUTOR_MODES",
    "FIND_FILE_MAX_RESULTS",
    "FIND_FILE_MAX_VISITED",
    "MAX_INPUT_LENGTH",
    "MODELS_DIR",
    "PROJECT_ROOT",
    "TIMER_MAX_SECONDS",
    "TIMER_MIN_SECONDS",
    "WEB_SEARCH_MAX_QUERY_LENGTH",
    "Config",
    "Thresholds",
    "default_config",
    "discover_dataset_path",
    "is_source_checkout",
]

# Input limits.
MAX_INPUT_LENGTH = 300

# Timer bounds (seconds). One second up to 24 hours.
TIMER_MIN_SECONDS = 1
TIMER_MAX_SECONDS = 24 * 60 * 60

# find_file limits.
FIND_FILE_MAX_RESULTS = 20
FIND_FILE_MAX_VISITED = 20_000
# find_file time budget: a search never blocks longer than this (seconds).
FIND_FILE_TIME_BUDGET_SECONDS = 5.0

# Web search limits.
WEB_SEARCH_MAX_QUERY_LENGTH = 200

# Calc limits.
CALC_MAX_EXPRESSION_LENGTH = 120
CALC_MAX_NODES = 64
CALC_MAX_LITERAL = 1e15
CALC_MAX_EXPONENT = 1000
# Results with more decimal digits than this are rejected instead of being
# converted to a huge string (the conversion itself is quadratic in CPython).
CALC_MAX_RESULT_DIGITS = 1000

# Audit log rotation (bytes per file, number of kept files including current).
AUDIT_MAX_BYTES = 5 * 1024 * 1024
AUDIT_MAX_FILES = 5


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

# Supported reply languages.
LANGUAGES: tuple[str, ...] = ("en", "de")


@dataclass
class Config:
    """Bundle of all configurable paths and runtime options.

    Kept as a plain dataclass so that tests and embedders can build one
    explicitly; :func:`default_config` applies the environment-aware defaults.
    """

    model_path: Path = DEFAULT_MODEL_PATH
    registry_path: Path = DEFAULT_REGISTRY_PATH
    dataset_path: Path = DEFAULT_DATASET_PATH
    audit_path: Path = DEFAULT_AUDIT_PATH
    models_dir: Path = MODELS_DIR
    executor_mode: str = "dry-run"  # "dry-run" | "windows"
    thresholds: Thresholds = field(default_factory=Thresholds)
    policy: SecurityPolicy = field(default_factory=default_policy)
    language: str = "en"
    auto_confirm: bool = False

    def __post_init__(self) -> None:
        if self.executor_mode not in EXECUTOR_MODES:
            raise ValueError(
                f"Unknown executor mode: {self.executor_mode!r} "
                f"(expected one of {', '.join(EXECUTOR_MODES)})"
            )
        if self.language not in LANGUAGES:
            raise ValueError(
                f"Unknown language: {self.language!r} "
                f"(expected one of {', '.join(LANGUAGES)})"
            )


def default_config() -> Config:
    """Return the default configuration (environment overrides applied)."""
    return Config(policy=policy_from_env())
