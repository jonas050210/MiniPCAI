"""``minipcai doctor``: one command that answers "why is it not working?".

Checks the installation end to end: files that must be present, whether the
registry is usable *on this machine*, whether the model matches the dataset it
was trained on, whether the audit log is writable and untampered, which policy
is active, and whether a GUI can start at all.

Every check returns ``ok``, ``warn`` or ``fail`` plus an actionable message, so
a support request can be answered with a single copy/paste of the output.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any

from minipcai import __version__, paths
from minipcai.audit import AuditError, AuditLogger, verify_chain
from minipcai.dataset import DatasetError, load_dataset
from minipcai.model import ModelError, SklearnIntentClassifier, verify_dataset_fingerprint
from minipcai.registry import Registry, RegistryError
from minipcai.settings import Settings

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


def run_checks(
    settings: Settings | None = None,
    check_files: bool = True,
    check_gui: bool = True,
) -> list[Check]:
    """Run all checks; returns them in a stable order."""
    settings = settings or Settings()
    checks: list[Check] = []
    checks.append(_check_environment())
    checks.append(_check_packaged_data())
    registry, registry_check = _check_registry(settings, check_files)
    checks.append(registry_check)
    checks.extend(_check_model(settings))
    checks.append(_check_audit(settings))
    checks.append(_check_policy(settings))
    if check_gui:
        checks.append(_check_gui())
    return checks


def _check_environment() -> Check:
    supported = sys.version_info >= (3, 10)
    detail = f"Python {sys.version.split()[0]} on {sys.platform}"
    return Check(
        "environment",
        OK if supported else FAIL,
        detail if supported else detail + " (3.10+ required)",
        data={"python": sys.version.split()[0], "platform": sys.platform,
              "version": __version__, "layout": paths.describe_layout()},
    )


_REQUIRED_DATA = ("registry.json",)


def _check_packaged_data() -> Check:
    data_dir = paths.packaged_data_dir()
    missing = [name for name in _REQUIRED_DATA if not (data_dir / name).is_file()]
    dataset = paths.default_dataset_path()
    if not dataset.is_file():
        missing.append(dataset.name)
    if missing:
        return Check(
            "packaged data",
            FAIL,
            f"missing in {data_dir}: {', '.join(missing)}",
            data={"dir": str(data_dir)},
        )
    return Check(
        "packaged data",
        OK,
        f"{data_dir} ({dataset.name})",
        data={"dir": str(data_dir), "dataset": str(dataset)},
    )


def _check_registry(
    settings: Settings, check_files: bool
) -> tuple[Registry | None, Check]:
    path = settings.resolved_registry_path()
    if not path.is_file():
        return None, Check(
            "registry",
            FAIL,
            f"not found: {path} (run 'minipcai setup')",
            data={"path": str(path)},
        )
    try:
        registry = Registry.load(path, policy=settings.to_policy())
    except RegistryError as exc:
        return None, Check("registry", FAIL, str(exc), data={"path": str(path)})
    stats = registry.stats()
    problems = registry.missing_targets() if check_files else []
    if problems:
        listed = "; ".join(
            f"{entry.section}/{entry.id}: {reason}" for entry, reason in problems[:8]
        )
        return registry, Check(
            "registry",
            WARN,
            f"{stats['entries']} entries, {len(problems)} unusable on this machine: {listed}",
            data={"path": str(path), "stats": stats, "problems": [
                {"section": entry.section, "id": entry.id, "reason": reason}
                for entry, reason in problems
            ]},
        )
    detail = f"{path} - {stats['entries']} entries"
    if not check_files:
        detail += " (existence not verified; use --check-files)"
    return registry, Check("registry", OK, detail, data={"path": str(path), "stats": stats})


def _check_model(settings: Settings) -> list[Check]:
    checks: list[Check] = []
    model_path = settings.resolved_model_path()
    if not model_path.is_file():
        return [
            Check(
                "model",
                FAIL,
                f"not found: {model_path} (run 'minipcai train')",
                data={"path": str(model_path)},
            )
        ]
    try:
        model = SklearnIntentClassifier.load(model_path)
    except ModelError as exc:
        return [Check("model", FAIL, str(exc), data={"path": str(model_path)})]
    metadata = model.metadata or {}
    dataset_info = metadata.get("dataset") or {}
    detail = (
        f"{model_path.name}, {len(model.labels)} labels, "
        f"dataset v{dataset_info.get('version', '?')} "
        f"({dataset_info.get('n_examples', '?')} examples)"
    )
    checks.append(
        Check(
            "model",
            OK,
            detail,
            data={
                "path": str(model_path),
                "labels": list(model.labels),
                "thresholds": metadata.get("thresholds"),
                "created_at": metadata.get("created_at"),
            },
        )
    )
    thresholds = metadata.get("thresholds")
    if not isinstance(thresholds, dict) or "min_confidence" not in thresholds:
        checks.append(
            Check(
                "thresholds",
                FAIL,
                "model metadata has no calibrated thresholds; retrain with 'minipcai train'",
            )
        )
    try:
        verify_dataset_fingerprint(metadata)
        checks.append(Check("dataset fingerprint", OK, "model matches the dataset"))
    except ModelError as exc:
        checks.append(Check("dataset fingerprint", FAIL, str(exc)))
    try:
        dataset = load_dataset()
        checks.append(
            Check(
                "dataset",
                OK,
                f"{dataset.path.name} (v{dataset.version}, {len(dataset.examples)} examples)",
                data={"sha256": dataset.sha256},
            )
        )
    except DatasetError as exc:
        checks.append(Check("dataset", FAIL, str(exc)))
    return checks


def _check_audit(settings: Settings) -> Check:
    path = settings.resolved_audit_path()
    logger = AuditLogger(
        path,
        max_bytes=settings.audit_max_bytes,
        max_files=settings.audit_max_files,
        store_text=settings.store_text_in_audit,
    )
    try:
        logger.check_writable()
    except AuditError as exc:
        return Check("audit log", FAIL, f"{exc}", data={"path": str(path)})
    if not path.is_file():
        return Check(
            "audit log",
            OK,
            f"writable, no records yet: {path}",
            data={"path": str(path), "privacy_mode": not settings.store_text_in_audit},
        )
    ok, checked, bad_line = verify_chain(path)
    rotated = logger.rotation_files()
    size_kib = path.stat().st_size / 1024
    if not ok:
        return Check(
            "audit log",
            FAIL,
            f"hash chain broken at line {bad_line} - the log was modified",
            data={"path": str(path), "checked": checked},
        )
    detail = (
        f"{path} - {checked} records, {size_kib:.1f} KiB, chain verified"
        + (f", {len(rotated)} rotated file(s)" if rotated else "")
    )
    return Check(
        "audit log",
        OK,
        detail,
        data={"path": str(path), "records": checked, "rotated": [str(p) for p in rotated]},
    )


def _check_policy(settings: Settings) -> Check:
    policy = settings.to_policy()
    detail = (
        f"confirmations: {', '.join(sorted(policy.confirm_intents)) or 'off'} | "
        f"trusted app roots: {len(policy.trusted_roots_expanded())} | "
        f"untrusted app paths: {'allowed' if not policy.enforce_trusted_app_roots else 'refused'}"
    )
    return Check("security policy", OK, detail, data={
        "confirm_intents": sorted(policy.confirm_intents),
        "enforce_trusted_app_roots": policy.enforce_trusted_app_roots,
        "blocked_executables": len(policy.blocked_executables),
        "blocked_file_extensions": len(policy.blocked_file_extensions),
        "max_active_timers": policy.max_active_timers,
    })


def _check_gui() -> Check:
    if importlib.util.find_spec("PySide6") is None:
        return Check(
            "gui",
            WARN,
            "PySide6 is not installed (pip install 'minipcai[gui]'); the CLI works without it",
        )
    try:
        from PySide6.QtWidgets import QApplication  # noqa: F401
    except Exception as exc:  # pragma: no cover - platform specific
        return Check("gui", WARN, f"PySide6 is installed but Qt cannot load: {exc}")
    return Check("gui", OK, "PySide6 available")


def has_failures(checks: list[Check]) -> bool:
    return any(check.status == FAIL for check in checks)


def format_report(checks: list[Check], json_output: bool = False) -> str:
    if json_output:
        return json.dumps([check.to_dict() for check in checks], indent=2, ensure_ascii=False)
    width = max(len(check.name) for check in checks)
    icons = {OK: "OK  ", WARN: "WARN", FAIL: "FAIL"}
    lines = [f"MiniPCAI doctor - {paths.describe_layout()['package']}"]
    for check in checks:
        lines.append(f"[{icons[check.status]}] {check.name.ljust(width)}  {check.detail}")
    failures = [check for check in checks if check.status == FAIL]
    warnings = [check for check in checks if check.status == WARN]
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} problem(s) found - MiniPCAI will not run correctly.")
    elif warnings:
        lines.append(f"Everything essential works ({len(warnings)} warning(s)).")
    else:
        lines.append("Everything works.")
    if shutil.which("minipcai") is None:
        lines.append("Note: the 'minipcai' script is not on PATH; run it as 'python -m minipcai'.")
    return "\n".join(lines)


def run(settings: Settings | None = None, check_files: bool = True,
        check_gui: bool = True, json_output: bool = False) -> tuple[int, str]:
    checks = run_checks(settings=settings, check_files=check_files, check_gui=check_gui)
    return (1 if has_failures(checks) else 0), format_report(checks, json_output=json_output)


def to_dict(checks: list[Check]) -> list[dict[str, Any]]:
    return [check.to_dict() for check in checks]
