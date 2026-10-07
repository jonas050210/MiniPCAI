"""``minipcai registry``: validate, list and edit the target registry.

The registry is the security boundary of MiniPCAI, which is why it is edited
by hand - but "by hand" must not mean "in an unknown JSON file with an unknown
schema". This module implements the editing commands, and every change is
validated by the very same loader that the runtime uses (plus the security
policy), so an entry that would be refused at runtime is refused here as well.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from minipcai import paths
from minipcai.policy import (
    SecurityPolicy,
    effective_policy,
    executable_name,
    expand_env,
    policy_from_env,
)
from minipcai.registry import REGISTRY_SECTIONS, Registry, RegistryError, registry_summary
from minipcai.settings import load_settings

_FIELD_BY_SECTION = {
    "apps": "executable",
    "files": "path",
    "folders": "path",
    "websites": "url",
    "searchers": "url_template",
}


def _target_registry_path(args: argparse.Namespace) -> tuple[Path, bool]:
    """Return ``(path, is_user_copy)`` for the registry to work on."""
    if getattr(args, "registry", None):
        return Path(args.registry), paths.registry_is_user_copy(args.registry)
    if getattr(args, "user", False):
        return paths.seed_user_registry(), True
    try:
        settings = load_settings()
        return settings.resolved_registry_path(), paths.registry_is_user_copy(
            settings.resolved_registry_path()
        )
    except ValueError:
        return paths.resolve_registry_path(), paths.registry_is_user_copy(
            paths.resolve_registry_path()
        )


class RegistryToolError(RuntimeError):
    """A user-facing problem while editing the registry."""


def _parse_pairs(values: list[str], field: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if "=" in value:
            entry_id, target = value.split("=", 1)
            entry_id = entry_id.strip()
            target = target.strip()
        else:
            entry_id = target = ""
        if not entry_id or not target:
            raise RegistryToolError(
                f"expected ID=PATH for --add-{field[:-1] if field != 'apps' else 'app'}, "
                f"got {value!r}"
            )
        pairs.append((entry_id, target))
    return pairs


def _entry_payload(
    section: str, entry_id: str, target: str, aliases: list[str], searchable: bool
) -> dict:
    payload: dict = {
        "id": entry_id,
        "aliases": aliases or [entry_id],
        _FIELD_BY_SECTION[section]: target,
    }
    if section == "folders" and searchable:
        payload["searchable"] = True
    return payload


def _policy_for(args: argparse.Namespace) -> SecurityPolicy:
    """Policy for this command: environment escape hatches + ``--allow-untrusted``."""
    policy = policy_from_env()
    if getattr(args, "allow_untrusted", False):
        policy = policy.allow_untrusted_app_paths()
    return policy


def _describe_entry(section: str, payload: dict) -> str:
    return f"{section[:-1]} '{payload['id']}' -> {payload[_FIELD_BY_SECTION[section]]}"


def add_entries(args: argparse.Namespace, path: Path) -> int:
    policy = _policy_for(args)
    aliases = [alias.strip() for alias in args.alias if alias.strip()]
    additions: list[tuple[str, dict]] = []
    for section, values in (
        ("apps", args.add_app),
        ("files", args.add_file),
        ("folders", args.add_folder),
        ("websites", args.add_website),
        ("searchers", args.add_searcher),
    ):
        for entry_id, target in _parse_pairs(values, section):
            additions.append(
                (
                    section,
                    _entry_payload(section, entry_id, target, aliases, args.searchable),
                )
            )
    if not additions:
        return 0

    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Error: cannot read {path}: {exc}", file=sys.stderr)
            return 2
    else:
        data = {"version": 1, **{section: [] for section in REGISTRY_SECTIONS}}
    for section in REGISTRY_SECTIONS:
        data.setdefault(section, [])
    if not isinstance(data.get("version"), int):
        data["version"] = 1

    existing_ids = {
        entry.get("id")
        for section in REGISTRY_SECTIONS
        for entry in data.get(section, [])
        if isinstance(entry, dict)
    }
    for section, payload in additions:
        if payload["id"] in existing_ids:
            print(
                f"Error: an entry with id '{payload['id']}' already exists.",
                file=sys.stderr,
            )
            return 2
        if section == "apps":
            name = executable_name(expand_env(payload["executable"]))
            if not name.endswith(".exe"):
                print(
                    f"Error: '{payload['executable']}' is not an .exe file.",
                    file=sys.stderr,
                )
                return 2
            if not policy.enforce_trusted_app_roots and not policy.is_trusted_app_path(
                payload["executable"]
            ):
                print(
                    "Warning: allowing an application outside the trusted locations "
                    "(recorded in the audit log of this command).",
                    file=sys.stderr,
                )
        if section == "files":
            reason = policy.is_blocked_file_target(payload["path"])
            if reason:
                print(f"Error: {reason}.", file=sys.stderr)
                return 2
        data[section].append(payload)
        existing_ids.add(payload["id"])

    return _write_and_validate(
        data, path, additions, policy=policy, quiet=bool(getattr(args, "json", False))
    )


def remove_entries(args: argparse.Namespace, path: Path) -> int:
    if not args.remove:
        return 0
    if not path.is_file():
        print(f"Error: registry not found: {path}", file=sys.stderr)
        return 2
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    removed: list[str] = []
    missing: list[str] = []
    for entry_id in args.remove:
        found = False
        for section in REGISTRY_SECTIONS:
            entries = [
                entry
                for entry in data.get(section, [])
                if not (isinstance(entry, dict) and entry.get("id") == entry_id)
            ]
            if len(entries) != len(data.get(section, [])):
                found = True
                data[section] = entries
        if not found:
            missing.append(entry_id)
        else:
            removed.append(entry_id)
    if missing:
        raise RegistryToolError(
            "no entry with id " + ", ".join(f"'{entry_id}'" for entry_id in missing)
            + " found"
        )
    payloads = [(section, {"id": entry_id, _FIELD_BY_SECTION[section]: ""})
                for section in REGISTRY_SECTIONS for entry_id in removed]
    return _write_and_validate(
        data, path, payloads, action="removed", policy=_policy_for(args),
        quiet=bool(getattr(args, "json", False)),
    )


def _write_and_validate(
    data: dict,
    path: Path,
    touched: list[tuple[str, dict]],
    action: str = "added",
    policy: SecurityPolicy | None = None,
    quiet: bool = False,
) -> int:
    """Validate the edited registry, then write it atomically."""
    try:
        registry = Registry.load_from_dict(data, path=path, policy=policy or effective_policy())
    except RegistryError as exc:
        print(f"Error: the edited registry would be invalid: {exc}", file=sys.stderr)
        return 2
    backup: Path | None = None
    if path.is_file():
        backup = path.with_suffix(path.suffix + ".bak")
        try:
            shutil.copyfile(path, backup)
        except OSError:
            backup = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        print(f"Error: cannot write {path}: {exc}", file=sys.stderr)
        return 2
    if not quiet:
        for section, payload in touched:
            target = payload.get(_FIELD_BY_SECTION[section], "")
            if target:
                print(f"{action}: {_describe_entry(section, payload)}")
        print(f"OK: {registry_summary(registry)} ({path})")
        if backup is not None:
            print(f"Backup: {backup}")
    return 0


def run(args: argparse.Namespace) -> int:
    """Entry point for ``minipcai registry``."""
    path, is_user = _target_registry_path(args)
    wrote = False
    try:
        adding = (
            args.add_app or args.add_file or args.add_folder
            or args.add_website or args.add_searcher
        )
        if adding:
            code = add_entries(args, path)
            if code != 0:
                return code
            wrote = True
        if args.remove:
            code = remove_entries(args, path)
            if code != 0:
                return code
            wrote = True
        policy = _policy_for(args)
        registry = Registry.load(path, policy=policy)
    except RegistryToolError as exc:
        print(f"Error: {exc}.", file=sys.stderr)
        return 2
    except RegistryError as exc:
        print(f"Invalid registry: {exc}", file=sys.stderr)
        return 2
    problems = registry.missing_targets()
    if args.json:
        print(
            json.dumps(
                {
                    "path": str(path),
                    "user_copy": is_user,
                    "summary": registry_summary(registry),
                    "stats": registry.stats(),
                    "problems": [
                        {"section": entry.section, "id": entry.id, "reason": reason}
                        for entry, reason in problems
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    print(f"OK: {registry_summary(registry)}")
    print(f"    file: {path}{' (per-user copy)' if is_user else ''}")
    searchable_ids = {folder.id for folder in registry.searchable_folders()}
    for entry in registry.entries:
        if entry.section in {"websites", "searchers"}:
            target = entry.url_template
        else:
            target = entry.path
        flags = " [searchable]" if entry.id in searchable_ids else ""
        print(f"  {entry.section:<9} {entry.id:<14} {target}{flags}")
        print(f"  {'':<9} aliases: {', '.join(entry.aliases)}")
    if problems:
        print()
        print(f"{len(problems)} entr(y/ies) are not usable on this machine:")
        for entry, reason in problems:
            print(f"  {entry.section}/{entry.id}: {reason}")
        print("Fix the paths, or remove the entries with --remove ID.")
    if not problems and args.check_files:
        print("\nAll targets exist on this machine.")
    if problems and args.check_files and not wrote:
        # Exit code 1: structurally valid, but not usable on this machine.
        return 1
    return 0


__all__ = ["run", "add_entries", "remove_entries", "RegistryToolError"]
