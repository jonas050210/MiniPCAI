"""Tests for the packaging contract (what a wheel must contain and expose).

The real wheel build runs in CI (``scripts/check_wheel_contents.py`` and
``scripts/smoke_installed.py``); these tests keep the *declaration* honest so
the defect "installed MiniPCAI ships no data" cannot come back unnoticed.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
PACKAGE_DIR = REPO_ROOT / "minipcai"


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


class TestProjectMetadata:
    def test_core_metadata(self, pyproject):
        project = pyproject["project"]
        assert project["name"] == "minipcai"
        assert project["requires-python"] >= ">=3.10"
        assert project["readme"] == "README.md"
        assert "MIT" in project["license"]["text"]

    def test_qt_is_optional(self, pyproject):
        project = pyproject["project"]
        assert not any("PySide6" in dep for dep in project["dependencies"])
        assert any("PySide6" in dep for dep in project["optional-dependencies"]["gui"])

    def test_console_scripts(self, pyproject):
        scripts = pyproject["project"]["scripts"]
        assert scripts["minipcai"] == "minipcai.cli:main"
        assert scripts["minipcai-train"] == "minipcai.train:main"

    def test_dev_extra_has_the_test_tooling(self, pyproject):
        dev = pyproject["project"]["optional-dependencies"]["dev"]
        assert any(dep.startswith("pytest") for dep in dev)
        assert any(dep.startswith("ruff") for dep in dev)

    def test_license_and_security_documents_exist(self):
        assert (REPO_ROOT / "LICENSE").is_file()
        text = (REPO_ROOT / "SECURITY.md").read_text(encoding="utf-8")
        assert "Threat model" in text and "Reporting a vulnerability" in text

    def test_ci_workflow_exists(self):
        workflow = REPO_ROOT / ".github" / "workflows" / "ci.yml"
        text = workflow.read_text(encoding="utf-8")
        assert "windows-latest" in text and "ubuntu-latest" in text
        assert "check_dataset_fingerprint.py" in text
        assert "smoke_installed.py" in text


class TestPackageData:
    def test_package_data_declares_registry_and_dataset(self, pyproject):
        package_data = pyproject["tool"]["setuptools"]["package-data"]
        assert "data/registry.json" in package_data["minipcai"]
        assert "data/intent_dataset.v*.jsonl" in package_data["minipcai"]

    def test_both_packages_are_declared(self, pyproject):
        packages = pyproject["tool"]["setuptools"]["packages"]
        assert "minipcai" in packages
        assert "minipcai.ui" in packages

    def test_data_lives_inside_the_package(self):
        assert (PACKAGE_DIR / "data" / "registry.json").is_file()
        datasets = list((PACKAGE_DIR / "data").glob("intent_dataset.v*.jsonl"))
        assert datasets, "the versioned dataset must live inside the package"
        # nothing in the repository root shadows it any more
        assert not (REPO_ROOT / "data").exists()

    def test_registry_file_is_valid_json_with_the_expected_sections(self):
        data = json.loads((PACKAGE_DIR / "data" / "registry.json").read_text(encoding="utf-8"))
        assert data["version"] >= 1
        assert set(data) >= {"apps", "files", "folders", "websites", "searchers"}

    def test_version_matches_pyproject(self, pyproject):
        import minipcai

        assert minipcai.__version__ == pyproject["project"]["version"]


class TestNoCheckoutDependency:
    def test_importing_from_another_directory_finds_data(self, tmp_path):
        """An installed package is used from unrelated working directories."""
        probe = (
            "import json;"
            "from minipcai import paths;"
            "from minipcai.registry import Registry;"
            "reg = paths.packaged_registry_path();"
            "ds = paths.default_dataset_path();"
            "assert reg.is_file(), reg;"
            "assert ds.is_file(), ds;"
            "Registry.load(reg);"
            "print(json.dumps({'registry': str(reg), 'dataset': ds.name}))"
        )
        # Windows needs the inherited environment (SYSTEMROOT etc.) for the
        # interpreter to start at all; the point of the test is that nothing
        # from the repository checkout is on the path by accident.
        env = dict(os.environ, PYTHONPATH=str(REPO_ROOT), PYTHONDONTWRITEBYTECODE="1")
        env.pop("PYTHONHOME", None)
        result = subprocess.run(
            [sys.executable, "-c", probe],
            cwd=str(tmp_path),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        assert payload["registry"].endswith("registry.json")
        assert payload["dataset"].startswith("intent_dataset.v")

    def test_scripts_referenced_by_ci_exist(self):
        """Whatever the workflow calls must exist (and stay callable)."""
        text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        referenced = set(re.findall(r"scripts/([\w.-]+\.(?:py|sh))", text))
        expected = {
            "check_dataset_fingerprint.py",
            "check_wheel_contents.py",
            "smoke_installed.py",
            "ci_pytest.sh",
        }
        assert expected <= referenced, referenced
        for name in sorted(referenced):
            assert (REPO_ROOT / "scripts" / name).is_file(), name

    def test_ci_reports_failures_as_annotations(self):
        """Broken tests must be readable without the raw (often unreachable) log."""
        text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
        assert "::error::" in (REPO_ROOT / "scripts" / "ci_pytest.sh").read_text(
            encoding="utf-8"
        )
        assert "ci_pytest.sh" in text
        for os_name in ("ubuntu-latest", "windows-latest"):
            assert os_name in text
