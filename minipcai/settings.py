"""Persistent configuration (TOML) with a documented precedence order.

    command line  >  environment (``MINIPCAI_*``)  >  config.toml  >  defaults

The file lives in the user's state directory and is created by
``minipcai setup``; every field is optional. Secrets do not exist in MiniPCAI,
so plain text is fine - but the file is *not* part of the repository and never
shipped.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from minipcai import paths
from minipcai.config import LANGUAGES, Thresholds
from minipcai.policy import DEFAULT_CONFIRM_INTENTS, SecurityPolicy, policy_from_env

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass
class Settings:
    """Everything a user can configure without editing code."""

    executor_mode: str = "dry-run"
    language: str = "de"
    auto_confirm: bool = False
    store_text_in_audit: bool = True
    audit_max_bytes: int = 5 * 1024 * 1024
    audit_max_files: int = 5
    find_file_time_budget: float = 5.0
    max_active_timers: int = 32
    allow_untrusted_apps: bool = False
    confirm_intents: tuple[str, ...] = tuple(sorted(DEFAULT_CONFIRM_INTENTS))
    extra_trusted_roots: tuple[str, ...] = ()
    model_path: str = ""
    registry_path: str = ""
    audit_path: str = ""
    models_dir: str = ""
    # GUI preferences (stored here so the GUI needs no second config file)
    theme: str = "system"
    show_details: bool = True
    hotkey: str = "Ctrl+Alt+M"

    # -- derived -------------------------------------------------------------
    def to_policy(self) -> SecurityPolicy:
        # The environment escape hatches (MINIPCAI_ALLOW_UNTRUSTED_APPS,
        # MINIPCAI_EXTRA_TRUSTED_ROOTS) apply here as well, so a directly
        # constructed Settings behaves like one loaded from the CLI.
        # Confirmations are a settings concern: MINIPCAI_NO_CONFIRM maps to
        # ``auto_confirm`` in ``apply_environment``.
        base = policy_from_env()
        policy = replace(
            base,
            confirm_intents=frozenset(self.confirm_intents),
            trusted_app_roots=base.trusted_app_roots + tuple(self.extra_trusted_roots),
            max_active_timers=self.max_active_timers,
        )
        if self.allow_untrusted_apps:
            policy = policy.allow_untrusted_app_paths()
        return policy

    def resolved_model_path(self) -> Path:
        if self.model_path:
            return Path(self.model_path)
        return paths.default_models_dir() / paths.MODEL_FILENAME

    def resolved_registry_path(self) -> Path:
        if self.registry_path:
            return Path(self.registry_path)
        return paths.resolve_registry_path()

    def resolved_audit_path(self) -> Path:
        return Path(self.audit_path) if self.audit_path else paths.default_audit_path()

    def resolved_models_dir(self) -> Path:
        return Path(self.models_dir) if self.models_dir else paths.default_models_dir()

    def apply_environment(self, environ: dict[str, str] | None = None) -> Settings:
        """Apply ``MINIPCAI_*`` overrides (below CLI, above the file)."""
        env = os.environ if environ is None else environ
        settings = self
        mapping = {
            "MINIPCAI_MODEL": "model_path",
            "MINIPCAI_REGISTRY": "registry_path",
            "MINIPCAI_AUDIT": "audit_path",
            "MINIPCAI_MODELS_DIR": "models_dir",
            "MINIPCAI_LANGUAGE": "language",
            "MINIPCAI_EXECUTOR": "executor_mode",
        }
        for variable, attribute in mapping.items():
            value = env.get(variable)
            if value:
                settings = replace(settings, **{attribute: value})
        if env.get("MINIPCAI_NO_CONFIRM", "").strip() in {"1", "true", "yes"}:
            settings = replace(settings, auto_confirm=True)
        if env.get("MINIPCAI_ALLOW_UNTRUSTED_APPS", "").strip() in {"1", "true", "yes"}:
            settings = replace(settings, allow_untrusted_apps=True)
        if env.get("MINIPCAI_PRIVACY_AUDIT", "").strip() in {"1", "true", "yes"}:
            settings = replace(settings, store_text_in_audit=False)
        extra = env.get("MINIPCAI_EXTRA_TRUSTED_ROOTS", "").strip()
        if extra:
            roots = tuple(part.strip() for part in extra.split(";") if part.strip())
            settings = replace(
                settings, extra_trusted_roots=settings.extra_trusted_roots + roots
            )
        return settings

    def validate(self) -> list[str]:
        """Return a list of problems (empty when the settings are usable)."""
        problems: list[str] = []
        if self.language not in LANGUAGES:
            problems.append(
                f"language must be one of {', '.join(LANGUAGES)} (got {self.language!r})"
            )
        if self.executor_mode not in ("dry-run", "windows"):
            problems.append(f"unknown executor mode {self.executor_mode!r}")
        if self.audit_max_files < 1:
            problems.append("audit_max_files must be at least 1")
        if self.audit_max_bytes < 0:
            problems.append("audit_max_bytes must not be negative")
        if self.max_active_timers < 1:
            problems.append("max_active_timers must be at least 1")
        if self.find_file_time_budget < 0:
            problems.append("find_file_time_budget must not be negative")
        return problems


def _coerce(name: str, value: Any, current: Any) -> Any:
    """Coerce a TOML value to the type of the current default."""
    if isinstance(current, bool):
        if isinstance(value, bool):
            return value
        raise ValueError(f"{name} must be true or false")
    if isinstance(current, int) and not isinstance(current, bool):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return int(value)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        return value
    if isinstance(current, tuple):
        if not isinstance(value, (list, tuple)) or not all(
            isinstance(item, str) for item in value
        ):
            raise ValueError(f"{name} must be a list of strings")
        return tuple(value)
    raise ValueError(f"{name} has an unsupported type")


def load_settings(path: Path | str | None = None, use_environment: bool = True) -> Settings:
    """Load settings from TOML, then apply environment overrides."""
    settings = Settings()
    config_path = Path(path) if path is not None else paths.user_config_path()
    if config_path.is_file():
        try:
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot read {config_path}: {exc}") from exc
        section = data.get("minipcai", data)
        if not isinstance(section, dict):
            raise ValueError(f"{config_path}: expected a [minipcai] table")
        updates: dict[str, Any] = {}
        for name, value in section.items():
            if name not in Settings.__dataclass_fields__:
                continue
            current = getattr(settings, name)
            updates[name] = _coerce(name, value, current)
        settings = replace(settings, **updates)
    if use_environment:
        settings = settings.apply_environment()
    return settings


def save_settings(settings: Settings, path: Path | str | None = None) -> Path:
    """Write settings as TOML (used by ``minipcai setup`` and the GUI)."""
    config_path = Path(path) if path is not None else paths.user_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(settings)
    lines = ["# MiniPCAI configuration - generated file, safe to edit by hand.", "[minipcai]"]
    for key in sorted(data):
        value = data[key]
        if isinstance(value, tuple):
            rendered = "[" + ", ".join(_toml_string(item) for item in value) + "]"
        elif isinstance(value, str):
            rendered = _toml_string(value)
        elif isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path


def _toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def settings_summary(settings: Settings) -> dict[str, Any]:
    """Compact dict for ``doctor``/``--json`` output."""
    return {
        "executor_mode": settings.executor_mode,
        "language": settings.language,
        "auto_confirm": settings.auto_confirm,
        "store_text_in_audit": settings.store_text_in_audit,
        "allow_untrusted_apps": settings.allow_untrusted_apps,
        "confirm_intents": list(settings.confirm_intents),
        "model_path": str(settings.resolved_model_path()),
        "registry_path": str(settings.resolved_registry_path()),
        "audit_path": str(settings.resolved_audit_path()),
        "models_dir": str(settings.resolved_models_dir()),
    }


def thresholds_from_metadata(metadata: dict[str, Any]) -> Thresholds:
    """Read calibrated thresholds; ``None`` fields fall back to defaults."""
    stored = metadata.get("thresholds") if isinstance(metadata, dict) else None
    if not isinstance(stored, dict):
        return Thresholds()
    try:
        return Thresholds(
            min_confidence=float(stored["min_confidence"]),
            min_margin=float(stored["min_margin"]),
        )
    except (KeyError, TypeError, ValueError):
        return Thresholds()


__all__ = [
    "Settings",
    "load_settings",
    "save_settings",
    "settings_summary",
    "thresholds_from_metadata",
]
