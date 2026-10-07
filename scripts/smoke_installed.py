#!/usr/bin/env python3
"""Install a built wheel into a clean virtual environment and use it.

This is the end-to-end packaging test that the repository-level test suite
cannot give: it proves that an *installed* MiniPCAI can seed its per-user files,
train a model from the packaged dataset and answer a request -- all from a
working directory that contains no checkout.

    python scripts/smoke_installed.py [--wheel dist/x.whl] [--reuse-site-packages]

``--reuse-site-packages`` makes the temporary venv inherit the current
interpreter's packages (useful offline, e.g. in a sandbox without PyPI).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    print("$ " + " ".join(str(part) for part in command))
    return subprocess.run(
        [str(part) for part in command], cwd=str(cwd), env=env, text=True,
        capture_output=True, check=False,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, default=None,
                        help="wheel to install (default: newest in dist/)")
    parser.add_argument("--reuse-site-packages", action="store_true",
                        help="let the test venv see the current environment's packages")
    args = parser.parse_args(argv)

    wheel = args.wheel
    if wheel is None:
        candidates = sorted((REPO_ROOT / "dist").glob("*.whl"))
        if not candidates:
            print("FAIL: no wheel found in dist/ (run 'python -m build')", file=sys.stderr)
            return 2
        wheel = candidates[-1]
    if not wheel.is_file():
        print(f"FAIL: {wheel} does not exist", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="minipcai-smoke-") as tmp:
        tmp_path = Path(tmp)
        venv_dir = tmp_path / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=args.reuse_site_packages).create(
            venv_dir
        )
        python = _venv_python(venv_dir)

        env = dict(os.environ)
        # A clean HOME makes the "first run" path realistic and keeps the test
        # from touching the developer's own ~/.minipcai.
        env["HOME"] = str(tmp_path / "home")
        env["USERPROFILE"] = env["HOME"]
        env["LOCALAPPDATA"] = str(tmp_path / "home" / "AppData" / "Local")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env.pop("MINIPCAI_REGISTRY", None)
        env.pop("MINIPCAI_MODEL", None)

        install = _run([python, "-m", "pip", "install", "--no-deps", wheel], tmp_path, env)
        if install.returncode != 0:
            # Dependencies are provided by the interpreter (--reuse-site-packages)
            # or already present; a real install failure is reported below.
            print(install.stdout[-2000:])
            print(install.stderr[-2000:], file=sys.stderr)
            return 1

        workdir = tmp_path / "work"  # deliberately *not* inside the repository
        workdir.mkdir()

        def check(label: str, command: list[str], expect_ok: bool = True) -> bool:
            result = _run(command, workdir, env)
            ok = (result.returncode == 0) == expect_ok
            print(f"[{'ok' if ok else 'FAIL'}] {label}")
            if not ok:
                print(result.stdout[-3000:])
                print(result.stderr[-3000:], file=sys.stderr)
            return ok

        failures = 0

        probe = (
            "from minipcai import paths;"
            "reg = paths.packaged_registry_path();"
            "ds = paths.default_dataset_path();"
            "assert reg.is_file(), reg;"
            "assert ds.is_file(), ds;"
            "print('packaged:', reg.name, ds.name)"
        )
        if not check("packaged data is importable from a foreign directory",
                     [python, "-c", probe]):
            failures += 1

        if not check("first run seeds the per-user registry",
                     [python, "-m", "minipcai", "setup", "--no-file-check"]):
            failures += 1

        models_dir = tmp_path / "models"
        if not check("training works from the installed package",
                     [python, "-m", "minipcai", "train", "--models-dir", str(models_dir)]):
            failures += 1
        else:
            model = models_dir / "model.joblib"
            if not check("a trained model answers a request",
                         [python, "-m", "minipcai", "ask", "öffne notepad",
                          "--model", str(model), "--yes"]):
                failures += 1
            if not check("doctor is happy with the installed package",
                         [python, "-m", "minipcai", "doctor", "--no-gui-check",
                          "--model", str(model), "--no-file-check"]):
                failures += 1

        if failures:
            print(f"FAIL: {failures} packaging smoke step(s) failed", file=sys.stderr)
            return 1
        print("OK: the wheel works from a clean environment")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
