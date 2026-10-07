"""Run the golden hard set against a real installation (CI / local gate).

Usage::

    python scripts/check_golden_hard.py [--dataset tests/data/golden_hard.jsonl]
                                        [--registry minipcai/data/registry.json]
                                        [--model models/model.joblib]
                                        [--sweep] [--json]

Exit code 0 means: no wrong accepts, no wrong targets and an unnecessary-reject
rate at or below the roadmap gate (8 %). ``--sweep`` additionally expands the
registry into hundreds of phrasing/typo variants.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from minipcai import paths  # noqa: E402
from minipcai.golden import (  # noqa: E402
    MAX_UNNECESSARY_REJECT_RATE,
    evaluate_cases,
    format_report,
    load_cases,
    summarize,
)


def build_assistant(model_path: Path, registry_path: Path, audit_path: Path):
    from minipcai.actions import DryRunExecutor
    from minipcai.audit import AuditLogger
    from minipcai.config import Thresholds
    from minipcai.model import SklearnIntentClassifier
    from minipcai.pipeline import Assistant
    from minipcai.registry import Registry

    model = SklearnIntentClassifier.load(model_path)
    registry = Registry.load(registry_path)
    thresholds = model.metadata.get("thresholds", {})
    return Assistant(
        model=model,
        registry=registry,
        executor=DryRunExecutor(),
        audit=AuditLogger(audit_path),
        thresholds=Thresholds(
            min_confidence=float(thresholds.get("min_confidence", 0.4)),
            min_margin=float(thresholds.get("min_margin", 0.1)),
        ),
        language="de",
        auto_confirm=True,
    )


def sweep_cases(registry) -> list[dict]:
    """Same expansion as ``tests/test_golden.py::TestRegistrySweep``."""
    sys.path.insert(0, str(REPO / "tests"))
    from test_golden import TestRegistrySweep

    return TestRegistrySweep.build_cases(registry)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=REPO / "tests" / "data" / "golden_hard.jsonl")
    parser.add_argument("--registry", type=Path, default=paths.packaged_registry_path())
    parser.add_argument("--model", type=Path, default=paths.default_models_dir() / paths.MODEL_FILENAME)
    parser.add_argument("--audit", type=Path, default=Path("golden-check-audit.jsonl"))
    parser.add_argument("--sweep", action="store_true", help="also run the registry-driven sweep")
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = parser.parse_args(argv)

    if not args.model.is_file():
        print(f"Error: no trained model at {args.model}. Run 'minipcai train' first.",
              file=sys.stderr)
        return 2

    assistant = build_assistant(args.model, args.registry, args.audit)
    outcomes = evaluate_cases(assistant, load_cases(args.dataset))
    summary = summarize(outcomes)
    report = {**summary, "set": "curated"}

    if args.sweep:
        outcomes += evaluate_cases(assistant, sweep_cases(assistant.registry))
        summary = summarize(outcomes)
        report = {**summary, "set": "curated+sweep"}

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report(summary, outcomes))

    ok = (
        summary["wrong_accepts"] == 0
        and summary["wrong_targets"] == 0
        and summary["unnecessary_reject_rate"] <= MAX_UNNECESSARY_REJECT_RATE
    )
    if not ok:
        print("GATE FAILED", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
