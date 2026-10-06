"""Command line interface for MiniPCAI.

Subcommands:

* ``ask``      - handle a single request and print the result
* ``chat``     - interactive text chat (REPL)
* ``train``    - train the intent model (alias of ``minipcai-train``)
* ``registry`` - validate and list the registry
* ``ui``       - launch the PySide6 desktop UI
* ``version``  - print version information
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from minipcai import __version__
from minipcai.config import (
    DEFAULT_AUDIT_PATH,
    DEFAULT_DATASET_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_REGISTRY_PATH,
    MODELS_DIR,
)
from minipcai.pipeline import build_assistant
from minipcai.registry import Registry, RegistryError, registry_summary


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH,
                        help="path to the trained model artifact")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH,
                        help="path to the target registry JSON")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH,
                        help="path to the audit log")
    parser.add_argument("--executor", choices=["dry-run", "windows"], default="dry-run",
                        help="executor mode (default: safe dry-run)")


def _build_assistant_from_args(args: argparse.Namespace):
    return build_assistant(
        model_path=args.model,
        registry_path=args.registry,
        audit_path=args.audit,
        executor_mode=args.executor,
    )


def _cmd_ask(args: argparse.Namespace) -> int:
    try:
        assistant = _build_assistant_from_args(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    result = assistant.handle(args.text)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    else:
        prefix = {"ok": "[ok]", "rejected": "[rejected]", "error": "[error]"}[result.status]
        print(f"{prefix} {result.message}")
        if result.intent:
            print(f"      intent: {result.intent} (confidence {result.confidence:.0%}, "
                  f"reason: {result.reason or 'none'})")
    return 0 if result.status == "ok" else 1


def _cmd_chat(args: argparse.Namespace) -> int:
    try:
        assistant = _build_assistant_from_args(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"MiniPCAI {__version__} - chat (executor: {args.executor}). "
          "Empty line or 'exit' to quit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not text or text.lower() in {"exit", "quit"}:
            return 0
        result = assistant.handle(text)
        prefix = {"ok": "[ok]", "rejected": "[rejected]", "error": "[error]"}[result.status]
        print(f"{prefix} {result.message}")


def _cmd_train(args: argparse.Namespace) -> int:
    from minipcai.train import main as train_main

    return train_main(
        ["--dataset", str(args.dataset), "--models-dir", str(args.models_dir),
         "--seed", str(args.seed)]
    )


def _cmd_registry(args: argparse.Namespace) -> int:
    try:
        registry = Registry.load(args.registry)
    except RegistryError as exc:
        print(f"Invalid registry: {exc}", file=sys.stderr)
        return 2
    print(f"OK: {registry_summary(registry)}")
    searchable_ids = {folder.id for folder in registry.searchable_folders()}
    for entry in registry.entries:
        target = entry.url if entry.section == "websites" else entry.path
        flags = " [searchable]" if entry.id in searchable_ids else ""
        print(f"  {entry.section:<9} {entry.id:<14} {target}{flags}")
        print(f"  {'':<9} aliases: {', '.join(entry.aliases)}")
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    from minipcai.ui import run_ui

    return run_ui(
        model_path=args.model,
        registry_path=args.registry,
        audit_path=args.audit,
        executor_mode=args.executor,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="minipcai",
        description="MiniPCAI: a small self-trained German-language Windows PC assistant.",
    )
    parser.add_argument("--version", action="version", version=f"minipcai {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="handle a single request")
    ask.add_argument("text", help="the request text (German)")
    ask.add_argument("--json", action="store_true", help="print the result as JSON")
    _add_common_arguments(ask)
    ask.set_defaults(func=_cmd_ask)

    chat = subparsers.add_parser("chat", help="interactive chat (REPL)")
    _add_common_arguments(chat)
    chat.set_defaults(func=_cmd_chat)

    train = subparsers.add_parser("train", help="train the intent model")
    train.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    train.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=_cmd_train)

    registry = subparsers.add_parser("registry", help="validate and list the registry")
    registry.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    registry.set_defaults(func=_cmd_registry)

    ui = subparsers.add_parser("ui", help="launch the PySide6 desktop UI")
    _add_common_arguments(ui)
    ui.set_defaults(func=_cmd_ui)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
