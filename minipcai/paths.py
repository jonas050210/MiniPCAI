"""Where MiniPCAI keeps its data - both in a checkout and when installed.

Resolution rules (first match wins):

* packaged defaults live next to the code in ``minipcai/data`` so that a
  ``pip install``ed wheel is fully functional;
* writable state (trained model, audit log, user registry, configuration)
  lives in a user directory (``%LOCALAPPDATA%\\MiniPCAI`` on Windows,
  ``~/.minipcai`` elsewhere);
* when MiniPCAI runs from a source checkout (``pyproject.toml`` next to the
  package) the repository's ``models/`` directory is used for artifacts so
  that development keeps working exactly as before.

Everything is overridable through ``--model``/``--registry``/``--audit`` flags
or the ``MINIPCAI_*`` environment variables, which keeps the CLI scriptable.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PACKAGED_DATA_DIR = PACKAGE_DIR / "data"
REPO_ROOT = PACKAGE_DIR.parent

REGISTRY_FILENAME = "registry.json"
MODEL_FILENAME = "model.joblib"
AUDIT_FILENAME = "audit.jsonl"
CONFIG_FILENAME = "config.toml"
MODELS_DIRNAME = "models"


def is_source_checkout() -> bool:
    """True when the package is used directly from a repository checkout."""
    return (REPO_ROOT / "pyproject.toml").is_file() and (
        REPO_ROOT / "minipcai" / "__init__.py"
    ).is_file()


def _is_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".minipcai-write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def packaged_data_dir() -> Path:
    """Directory with the files that ship with MiniPCAI."""
    return PACKAGED_DATA_DIR


def packaged_registry_path() -> Path:
    return PACKAGED_DATA_DIR / REGISTRY_FILENAME


def state_dir(environ: dict[str, str] | None = None) -> Path:
    """Directory for writable per-user state (audit log, user registry, model)."""
    env = os.environ if environ is None else environ
    if sys.platform == "win32":
        base = env.get("LOCALAPPDATA") or env.get("APPDATA")
        if base:
            return Path(base) / "MiniPCAI"
        return Path.home() / ".minipcai"
    return Path.home() / ".minipcai"


def user_config_path(environ: dict[str, str] | None = None) -> Path:
    return state_dir(environ) / CONFIG_FILENAME


def user_registry_path(environ: dict[str, str] | None = None) -> Path:
    return state_dir(environ) / REGISTRY_FILENAME


def user_models_dir(environ: dict[str, str] | None = None) -> Path:
    return state_dir(environ) / MODELS_DIRNAME


def default_models_dir() -> Path:
    """Where trained artifacts belong: the checkout, else the user directory."""
    if is_source_checkout() and _is_writable_directory(REPO_ROOT / MODELS_DIRNAME):
        return REPO_ROOT / MODELS_DIRNAME
    return user_models_dir()


def default_audit_path() -> Path:
    return state_dir() / AUDIT_FILENAME


_DATASET_NAME_RE = re.compile(r"^intent_dataset\.v(\d+)\.jsonl$")


def discover_dataset_path(data_dir: Path | str | None = None) -> Path:
    """Return the highest-versioned dataset file in ``data_dir``.

    Datasets are versioned through their file name (``intent_dataset.vN.jsonl``)
    so that publishing a new dataset version does not require a code change.
    Falls back to the v1 path when no versioned file exists; the loader then
    reports the missing file with a clear error.
    """
    directory = Path(data_dir) if data_dir is not None else PACKAGED_DATA_DIR
    try:
        candidates = list(directory.glob("intent_dataset.v*.jsonl"))
    except OSError:  # unreadable directory: fall back to the default name
        candidates = []
    best: tuple[int, Path] | None = None
    for candidate in candidates:
        match = _DATASET_NAME_RE.match(candidate.name)
        if match and (best is None or int(match.group(1)) > best[0]):
            best = (int(match.group(1)), candidate)
    return best[1] if best is not None else directory / "intent_dataset.v1.jsonl"


def default_dataset_path() -> Path:
    """The highest-versioned packaged dataset, else the v1 fallback path."""
    return discover_dataset_path(PACKAGED_DATA_DIR)


def resolve_registry_path(explicit: Path | str | None = None) -> Path:
    """Resolve the registry file to use.

    Order: explicit argument, ``MINIPCAI_REGISTRY``, the user registry (seeded
    from the packaged default when missing), the packaged default.
    """
    if explicit is not None:
        return Path(explicit)
    from_env = os.environ.get("MINIPCAI_REGISTRY")
    if from_env:
        return Path(from_env)
    user_path = user_registry_path()
    if user_path.is_file():
        return user_path
    if is_source_checkout() and packaged_registry_path().is_file():
        return packaged_registry_path()
    return seed_user_registry()


def seed_user_registry(overwrite: bool = False) -> Path:
    """Copy the packaged registry into the user directory (first run).

    Returns the path of the user registry. Existing user registries are never
    touched unless ``overwrite`` is set, so hand-curated targets survive
    upgrades.
    """
    target = user_registry_path()
    if target.is_file() and not overwrite:
        return target
    source = packaged_registry_path()
    if not source.is_file():
        return target
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    except OSError:
        return source
    return target


def registry_is_user_copy(path: Path | str) -> bool:
    try:
        return Path(path).resolve() == user_registry_path().resolve()
    except OSError:  # pragma: no cover - defensive
        return False


def describe_layout() -> dict[str, str]:
    """Human-readable layout summary used by ``minipcai doctor``."""
    return {
        "package": str(PACKAGE_DIR),
        "packaged data": str(PACKAGED_DATA_DIR),
        "state": str(state_dir()),
        "models": str(default_models_dir()),
        "checkout": "yes" if is_source_checkout() else "no",
    }
