"""Command line interface for MiniPCAI.

Subcommands:

* ``ask``      - handle a single request (interactive confirmation, ``--yes``)
* ``chat``     - interactive text chat with clarifications and confirmations
* ``train``    - train the intent model (alias of ``minipcai-train``)
* ``registry`` - validate, list and *edit* the target registry
* ``doctor``   - check the whole installation and report problems
* ``audit``    - show recent audit records, verify the hash chain
* ``setup``    - create the per-user configuration and registry
* ``ui``       - launch the desktop UI
* ``version``  - print version information

Global options (also accepted per subcommand): ``--model``, ``--registry``,
``--audit``, ``--executor``, ``--language``, ``--yes``, ``--json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minipcai import __version__, doctor, i18n, paths
from minipcai.audit import SCHEMA_VERSION, verify_chain
from minipcai.config import EXECUTOR_MODES, LANGUAGES
from minipcai.service import (
    STATUS_CLARIFICATION_REQUIRED,
    AssistantService,
    build_service,
)
from minipcai.settings import Settings, load_settings, save_settings

_STATUS_PREFIX = {
    "ok": "[ok]",
    "rejected": "[rejected]",
    "error": "[error]",
    "confirmation_required": "[confirmation]",
    STATUS_CLARIFICATION_REQUIRED: "[question]",
}


# ---------------------------------------------------------------------------
# argument plumbing
# ---------------------------------------------------------------------------


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=None,
                        help="path to the trained model artifact")
    parser.add_argument("--registry", type=Path, default=None,
                        help="path to the target registry JSON")
    parser.add_argument("--audit", type=Path, default=None,
                        help="path to the audit log")
    parser.add_argument("--executor", choices=list(EXECUTOR_MODES), default=None,
                        help="executor mode (default: dry-run, or the configured value)")
    parser.add_argument("--language", choices=list(LANGUAGES), default=None,
                        help="language of the replies (default: configured value, else en)")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask for confirmation (for scripts)")
    parser.add_argument("--json", action="store_true", help="print machine readable JSON")
    parser.add_argument("--check-files", action="store_true",
                        help="verify that registry targets exist on this machine")
    parser.add_argument("--verbose", "-v", action="store_true", help="verbose logging")


def _settings_from_args(args: argparse.Namespace) -> Settings:
    try:
        settings = load_settings()
    except ValueError as exc:
        print(f"Warning: {exc}", file=sys.stderr)
        settings = Settings()
    if getattr(args, "executor", None):
        settings.executor_mode = args.executor
    if getattr(args, "language", None):
        settings.language = args.language
    if getattr(args, "yes", False):
        settings.auto_confirm = True
    if getattr(args, "model", None):
        settings.model_path = str(args.model)
    if getattr(args, "registry", None):
        settings.registry_path = str(args.registry)
    if getattr(args, "audit", None):
        settings.audit_path = str(args.audit)
    return settings


def _configure_logging(verbose: bool) -> None:
    import logging

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def _build_service(args: argparse.Namespace) -> AssistantService:
    settings = _settings_from_args(args)
    return build_service(
        settings,
        check_files=getattr(args, "check_files", False),
    )


def _print_result(result, language: str, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return
    prefix = _STATUS_PREFIX.get(result.status, "[?]")
    print(f"{prefix} {result.message}")
    if result.intent and result.confidence is not None and result.status != "ok":
        print(
            f"      intent: {result.intent} (confidence {result.confidence:.0%}, "
            f"reason: {result.reason or 'none'})"
        )


def _prompt(question: str) -> str:
    try:
        return input(question).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def _interactive() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _affirmative(text: str, language: str) -> bool:
    yes = i18n.t("confirmation.yes", language).lower()
    normalized = text.strip().lower()
    return normalized in {yes, "ja", "j", "yes", "y", "ok", "okay", "1"}


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        service = _build_service(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    language = service.language
    result = service.ask(args.text)

    if result.status == "confirmation_required":
        if args.yes or not _interactive():
            if not args.yes:
                print(
                    "Error: this action requires confirmation. Re-run with --yes to "
                    "confirm it automatically.",
                    file=sys.stderr,
                )
                _print_result(result, language, json_output=args.json)
                return 1
            result = service.confirm(True)
        else:
            print(f"[confirmation] {result.details.get('description', '')}")
            answer = _prompt("Proceed? [y/N] ")
            result = service.confirm(_affirmative(answer, language))

    if result.status == STATUS_CLARIFICATION_REQUIRED:
        if not _interactive():
            _print_result(result, language, json_output=args.json)
            return 1
        answer = _prompt(f"{result.message} ")
        result = service.ask(answer)

    _print_result(result, language, json_output=args.json)
    return 0 if result.status == "ok" else 1


def _cmd_chat(args: argparse.Namespace) -> int:
    try:
        service = _build_service(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    language = service.language
    print(
        f"MiniPCAI {__version__} - chat (executor: {service.assistant.executor.mode}, "
        f"language: {language}). Commands: /help, /timers, /cancel, /executor, /language, "
        "'exit'."
    )
    while True:
        text = _prompt("> ")
        if not text:
            return 0
        lowered = text.lower()
        if lowered in {"exit", "quit"}:
            return 0
        if lowered == "/help":
            print(
                "Ask in German, e.g. 'öffne notepad', 'wie viel ram ist frei', "
                "'stelle einen timer auf 5 minuten'.\n"
                "/timers - list active timers, /cancel - cancel all timers,\n"
                "/executor dry-run|windows - switch executor, /language en|de."
            )
            continue
        if lowered == "/timers":
            timers = service.active_timers()
            if not timers:
                print("No active timers.")
            for info in timers:
                print(
                    f"  {info['request_id']}: {info['remaining_seconds']}s left "
                    f"(of {info['duration_seconds']}s)"
                )
            continue
        if lowered == "/cancel":
            print(f"Cancelled {service.cancel_timers()} timer(s).")
            continue
        if lowered.startswith("/executor"):
            parts = lowered.split()
            if len(parts) == 2 and parts[1] in EXECUTOR_MODES:
                from minipcai.actions import DryRunExecutor, WindowsExecutor

                service.assistant.executor = (
                    WindowsExecutor(
                        audit=service.assistant.audit,
                        max_active_timers=service.assistant.policy.max_active_timers,
                    )
                    if parts[1] == "windows"
                    else DryRunExecutor()
                )
                print(f"Executor: {parts[1]}")
            else:
                print(f"Current executor: {service.assistant.executor.mode}")
            continue
        if lowered.startswith("/language"):
            parts = lowered.split()
            if len(parts) == 2 and parts[1] in LANGUAGES:
                service.language = parts[1]
                service.assistant.language = parts[1]
                print(f"Language: {parts[1]}")
            else:
                print(f"Current language: {service.language}")
            continue
        result = service.ask(text)
        _print_result(result, service.language)
        if result.status == "confirmation_required" and _interactive():
            answer = _prompt("Proceed? [y/N] ")
            _print_result(service.confirm(_affirmative(answer, service.language)), service.language)


def _cmd_train(args: argparse.Namespace) -> int:
    from minipcai.train import parse_gate, train

    try:
        train(
            dataset_path=args.dataset,
            models_dir=args.models_dir,
            seed=args.seed,
            registry_path=args.registry,
            grouped_eval=args.grouped_eval,
            promotion_gate=parse_gate(args.gate),
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


def _cmd_registry(args: argparse.Namespace) -> int:
    from minipcai import registry_tools

    return registry_tools.run(args)


def _cmd_doctor(args: argparse.Namespace) -> int:
    settings = _settings_from_args(args)
    code, report = doctor.run(
        settings=settings,
        check_files=not args.no_file_check,
        check_gui=not args.no_gui_check,
        json_output=args.json,
    )
    print(report)
    return code


def _cmd_audit(args: argparse.Namespace) -> int:
    """Print the most recent audit records (read-only)."""
    path = Path(args.audit) if args.audit else paths.default_audit_path()
    if not path.is_file():
        print(f"No audit log yet at {path}.")
        return 0
    if args.verify:
        ok, checked, bad_line = verify_chain(path)
        if ok:
            print(f"OK: {checked} record(s) verified, hash chain intact ({path}).")
            return 0
        print(
            f"FAILED: hash chain broken at line {bad_line} of {path} "
            "(the log was modified or truncated).",
            file=sys.stderr,
        )
        return 1
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        print(f"Error: cannot read audit log {path}: {exc}", file=sys.stderr)
        return 2
    records: list[dict] = []
    skipped = 0
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        if isinstance(record, dict):
            records.append(record)
    if not records:
        print(f"Audit log at {path} contains no readable records.")
        if skipped:
            print(f"({skipped} unreadable line(s) skipped)", file=sys.stderr)
        return 0
    for record in records[-args.limit :]:
        if args.json:
            print(json.dumps(record, ensure_ascii=False))
        else:
            print(_format_audit_record(record))
    if skipped:
        print(f"({skipped} unreadable line(s) skipped)", file=sys.stderr)
    return 0


def _format_audit_record(record: dict) -> str:
    event = record.get("event", "?")
    ts = str(record.get("ts", ""))[:19].replace("T", " ")
    request_id = record.get("request_id", "-")
    if event == "request_accepted":
        return (
            f"{ts}  {request_id}  accepted  {record.get('intent')} "
            f"(target: {record.get('target_kind')}/{record.get('target_id')}, "
            f"executor: {record.get('executor_mode')})"
        )
    if event == "request_result":
        return (
            f"{ts}  {request_id}  rejected  {record.get('reason')} "
            f"intent={record.get('intent')}"
        )
    if event == "action_result":
        status = "ok" if record.get("ok") else "failed"
        return (
            f"{ts}  {request_id}  action {status} ({record.get('executor_mode')}, "
            f"{record.get('duration_ms')} ms): {record.get('summary')}"
        )
    if event == "timer_elapsed":
        return f"{ts}  {request_id}  timer elapsed ({record.get('duration_seconds')} s)"
    if event == "confirmation":
        return (
            f"{ts}  {request_id}  confirmation {record.get('decision')} "
            f"({record.get('intent')}: {record.get('action')})"
        )
    if event == "clarification":
        return (
            f"{ts}  {request_id}  clarification {record.get('decision')} "
            f"({record.get('options')})"
        )
    return f"{ts}  {request_id}  {event}"


def _cmd_setup(args: argparse.Namespace) -> int:
    """Create the per-user registry and configuration (first run)."""
    settings = load_settings(use_environment=False)
    if args.language:
        settings.language = args.language
    if args.executor:
        settings.executor_mode = args.executor
    if args.private_audit:
        settings.store_text_in_audit = False
    if args.models_dir is not None:
        settings.models_dir = str(args.models_dir)

    state = paths.state_dir()
    state.mkdir(parents=True, exist_ok=True)
    registry_path = paths.user_registry_path()
    if registry_path.exists() and not args.force:
        print(f"Keeping the existing registry: {registry_path}")
    else:
        source = paths.seed_user_registry(overwrite=True)
        print(f"Registry: {registry_path} (seeded from {source})" if source != registry_path
              else f"Registry: {registry_path}")
    settings.registry_path = str(registry_path)
    config_path = save_settings(settings, paths.user_config_path())
    print(f"Configuration: {config_path}")

    if args.train:
        try:
            from minipcai.train import train

            train(dataset_path=paths.default_dataset_path(),
                  models_dir=settings.resolved_models_dir())
        except Exception as exc:  # noqa: BLE001 - CLI boundary
            print(f"Training failed: {exc}", file=sys.stderr)
            return 2
    checks = doctor.run_checks(settings=settings, check_files=not args.no_file_check)
    print()
    print(doctor.format_report(checks))
    # A fresh installation has no trained model yet - that is a next step, not a
    # failure, so `minipcai setup` stays usable in scripts.
    blocking = [
        check for check in checks
        if check.status == doctor.FAIL and check.name != "model"
    ]
    if any(check.name == "model" and check.status == doctor.FAIL for check in checks):
        print("\nNext step: 'minipcai train', then 'minipcai ui'.")
    else:
        print("\nNext steps: 'minipcai train' (if not trained yet), then 'minipcai ui'.")
    return 1 if blocking else 0


def _cmd_ui(args: argparse.Namespace) -> int:
    """Launch the desktop UI (``minipcai ui`` / ``minipcai gui``)."""
    from minipcai.ui import GuiUnavailable, run_ui

    settings = _settings_from_args(args)
    try:
        return run_ui(settings=settings)
    except GuiUnavailable as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minipcai",
        description="MiniPCAI: a small self-trained German-language Windows PC assistant.",
    )
    parser.add_argument("--version", action="version", version=f"minipcai {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="handle a single request")
    ask.add_argument("text", help="the request text (German)")
    _add_common_arguments(ask)
    ask.set_defaults(func=_cmd_ask)

    chat = subparsers.add_parser("chat", help="interactive chat (REPL)")
    _add_common_arguments(chat)
    chat.set_defaults(func=_cmd_chat)

    train = subparsers.add_parser("train", help="train the intent model")
    train.add_argument("--dataset", type=Path, default=paths.default_dataset_path())
    train.add_argument("--models-dir", type=Path, default=paths.default_models_dir())
    train.add_argument("--registry", type=Path, default=paths.packaged_registry_path())
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--grouped-eval", action="store_true",
                       help="also evaluate with near-duplicate-disjoint folds")
    train.add_argument("--gate", default=None,
                       help="fail when quality drops below name=value,name=value")
    train.set_defaults(func=_cmd_train)

    registry = subparsers.add_parser(
        "registry", help="validate, list and edit the target registry"
    )
    registry.add_argument("--registry", type=Path, default=None)
    registry.add_argument("--check-files", action="store_true",
                          help="verify that every target exists on this machine")
    registry.add_argument("--user", action="store_true",
                          help="edit the per-user registry instead of the packaged one")
    registry.add_argument("--add-app", metavar="ID=PATH", action="append", default=[],
                          help="add an application (repeatable)")
    registry.add_argument("--add-file", metavar="ID=PATH", action="append", default=[],
                          help="add a document (repeatable)")
    registry.add_argument("--add-folder", metavar="ID=PATH", action="append", default=[],
                          help="add a folder (repeatable)")
    registry.add_argument("--add-website", metavar="ID=URL", action="append", default=[],
                          help="add a website (repeatable)")
    registry.add_argument("--add-searcher", metavar="ID=TEMPLATE", action="append", default=[],
                          help="add a web search provider, template must contain {query}")
    registry.add_argument("--alias", action="append", default=[],
                          help="aliases for the added entry (repeatable)")
    registry.add_argument("--searchable", action="store_true",
                          help="mark an added folder as searchable for find_file")
    registry.add_argument("--remove", metavar="ID", action="append", default=[],
                          help="remove a registry entry by id (repeatable)")
    registry.add_argument("--allow-untrusted", action="store_true",
                          help="accept an application outside the trusted locations")
    registry.add_argument("--json", action="store_true")
    registry.set_defaults(func=_cmd_registry)

    doctor_parser = subparsers.add_parser("doctor", help="check the installation")
    doctor_parser.add_argument("--registry", type=Path, default=None)
    doctor_parser.add_argument("--model", type=Path, default=None)
    doctor_parser.add_argument("--audit", type=Path, default=None)
    doctor_parser.add_argument("--json", action="store_true")
    doctor_parser.add_argument("--no-file-check", action="store_true",
                               help="skip verifying that registry targets exist")
    doctor_parser.add_argument("--no-gui-check", action="store_true")
    doctor_parser.set_defaults(func=_cmd_doctor)

    audit = subparsers.add_parser("audit", help="show recent audit log entries")
    audit.add_argument("--audit", type=Path, default=None, help="path to the audit log")
    audit.add_argument("--limit", type=int, default=20,
                       help="how many recent records to show (default: 20)")
    audit.add_argument("--json", action="store_true",
                       help="print raw JSON records instead of formatted lines")
    audit.add_argument("--verify", action="store_true",
                       help="verify the hash chain of the whole log")
    audit.set_defaults(func=_cmd_audit)

    setup = subparsers.add_parser("setup", help="create the per-user configuration")
    setup.add_argument("--force", action="store_true",
                       help="overwrite an existing user registry with the default one")
    setup.add_argument("--train", action="store_true", help="train the model as well")
    setup.add_argument("--models-dir", type=Path, default=None,
                       help="where the trained model is written (default: per-user)")
    setup.add_argument("--language", choices=list(LANGUAGES), default=None)
    setup.add_argument("--executor", choices=list(EXECUTOR_MODES), default=None)
    setup.add_argument("--private-audit", action="store_true",
                       help="do not store request texts in the audit log")
    setup.add_argument("--no-file-check", action="store_true")
    setup.set_defaults(func=_cmd_setup)

    ui = subparsers.add_parser("ui", help="launch the desktop UI")
    _add_common_arguments(ui)
    ui.set_defaults(func=_cmd_ui)

    gui = subparsers.add_parser("gui", help="alias for 'ui' (launch the desktop UI)")
    _add_common_arguments(gui)
    gui.set_defaults(func=_cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(getattr(args, "verbose", False))
    return int(args.func(args))


__all__ = ["main", "build_parser", "SCHEMA_VERSION"]


if __name__ == "__main__":
    raise SystemExit(main())
