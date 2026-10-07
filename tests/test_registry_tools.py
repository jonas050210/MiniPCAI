"""Tests for ``minipcai registry`` (validating and editing the registry)."""

from __future__ import annotations

import json
from pathlib import Path

from minipcai.cli import main
from minipcai.registry import Registry


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class TestAddEntries:
    def _base(self, tmp_path: Path) -> Path:
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "apps": [], "files": [], "folders": [],
                        "websites": [], "searchers": []}),
            encoding="utf-8",
        )
        return path

    def test_add_app_with_aliases(self, capsys, tmp_path):
        path = self._base(tmp_path)
        app = tmp_path / "tool.exe"
        app.touch()
        code = main([
            "registry", "--registry", str(path),
            "--add-app", f"tool={app}",
            "--alias", "werkzeug", "--alias", "tool",
        ])
        assert code == 0
        payload = _read(path)
        entry = payload["apps"][0]
        assert entry["id"] == "tool"
        assert entry["executable"] == str(app)
        assert entry["aliases"] == ["werkzeug", "tool"]
        # the written file must be loadable again
        registry = Registry.load(path)
        assert registry.by_id("tool").path == str(app)

    def test_add_folder_with_searchable_flag(self, capsys, tmp_path):
        path = self._base(tmp_path)
        folder = tmp_path / "Documents"
        folder.mkdir()
        assert main([
            "registry", "--registry", str(path),
            "--add-folder", f"dokumente={folder}", "--searchable",
        ]) == 0
        assert _read(path)["folders"][0]["searchable"] is True

    def test_add_website_and_searcher(self, capsys, tmp_path):
        path = self._base(tmp_path)
        assert main([
            "registry", "--registry", str(path),
            "--add-website", "wiki=https://example.org",
            "--add-searcher", "example=https://example.org/search?q={query}",
        ]) == 0
        payload = _read(path)
        assert payload["websites"][0]["url"] == "https://example.org"
        assert "{query}" in payload["searchers"][0]["url_template"]

    def test_searcher_requires_placeholder(self, capsys, tmp_path):
        path = self._base(tmp_path)
        code = main([
            "registry", "--registry", str(path),
            "--add-searcher", "example=https://example.org/search",
        ])
        assert code == 2
        assert "placeholder" in capsys.readouterr().err

    def test_bad_section_targets_are_refused(self, capsys, tmp_path):
        path = self._base(tmp_path)
        script = tmp_path / "script.bat"
        script.touch()
        code = main(["registry", "--registry", str(path),
                     "--add-app", f"script={script}"])
        assert code == 2
        assert ".exe" in capsys.readouterr().err
        # nothing was written
        assert _read(path)["apps"] == []

    def test_blocked_extension_is_refused_for_files(self, capsys, tmp_path):
        path = self._base(tmp_path)
        shortcut = tmp_path / "start.lnk"
        shortcut.touch()
        code = main(["registry", "--registry", str(path),
                     "--add-file", f"start={shortcut}"])
        assert code == 2
        assert "execute code" in capsys.readouterr().err

    def test_untrusted_app_needs_the_flag(self, capsys, tmp_path):
        path = self._base(tmp_path)
        app = tmp_path / "tool.exe"
        app.touch()
        windows_style = r"C:\tools\tool.exe"
        code = main(["registry", "--registry", str(path),
                     "--add-app", f"tool={windows_style}"])
        assert code == 2
        assert "trusted" in capsys.readouterr().err
        code = main(["registry", "--registry", str(path),
                     "--add-app", f"tool={windows_style}", "--allow-untrusted"])
        assert code == 0
        assert _read(path)["apps"][0]["executable"] == windows_style

    def test_invalid_id_is_refused(self, capsys, tmp_path):
        path = self._base(tmp_path)
        app = tmp_path / "tool.exe"
        app.touch()
        assert main(["registry", "--registry", str(path),
                     "--add-app", f"bad id={app}"]) == 2

    def test_malformed_pair_is_refused(self, capsys, tmp_path):
        path = self._base(tmp_path)
        assert main(["registry", "--registry", str(path),
                     "--add-app", "no-equals-sign"]) == 2
        assert "ID=PATH" in capsys.readouterr().err

    def test_duplicate_id_is_refused(self, capsys, tmp_path):
        path = self._base(tmp_path)
        app = tmp_path / "tool.exe"
        app.touch()
        assert main(["registry", "--registry", str(path),
                     "--add-app", f"tool={app}"]) == 0
        capsys.readouterr()
        assert main(["registry", "--registry", str(path),
                     "--add-app", f"tool={app}"]) == 2


class TestRemoveEntries:
    def test_remove_by_id(self, capsys, tmp_path):
        app = tmp_path / "tool.exe"
        app.touch()
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "apps": [
                {"id": "tool", "aliases": ["tool"], "executable": str(app)},
                {"id": "other", "aliases": ["other"], "executable": str(app)},
            ]}),
            encoding="utf-8",
        )
        assert main(["registry", "--registry", str(path), "--remove", "tool"]) == 0
        assert [entry["id"] for entry in _read(path)["apps"]] == ["other"]

    def test_remove_unknown_id_fails(self, capsys, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"version": 1, "apps": []}), encoding="utf-8")
        assert main(["registry", "--registry", str(path), "--remove", "nope"]) == 2
        assert "no entry with id" in capsys.readouterr().err


class TestCheckFiles:
    def test_check_files_reports_missing_targets(self, capsys, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "files": [
                {"id": "ghost", "aliases": ["ghost"],
                 "path": str(tmp_path / "not-there.txt")},
            ]}),
            encoding="utf-8",
        )
        code = main(["registry", "--registry", str(path), "--check-files"])
        assert code == 1  # valid registry, unusable target: reported, not crashed
        out = capsys.readouterr().out
        assert "not usable on this machine" in out
        assert "ghost" in out

    def test_json_output(self, capsys, tmp_path):
        app = tmp_path / "tool.exe"
        app.touch()
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "apps": [
                {"id": "tool", "aliases": ["tool"], "executable": str(app)},
            ]}),
            encoding="utf-8",
        )
        code = main(["registry", "--registry", str(path), "--json", "--check-files"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["stats"]["apps"] == 1
        assert data["problems"] == []
        assert "registry v1" in data["summary"]


class TestUserRegistry:
    def test_add_to_user_registry(self, capsys, tmp_path, monkeypatch):
        from minipcai import paths

        monkeypatch.setattr(paths, "state_dir", lambda environ=None: tmp_path / "state")
        app = tmp_path / "tool.exe"
        app.touch()
        code = main(["registry", "--user", "--add-app", f"tool={app}", "--json"])
        assert code == 0
        data = json.loads(capsys.readouterr().out)
        assert data["user_copy"] is True
        assert (tmp_path / "state" / "registry.json").is_file()

    def test_default_registry_without_arguments(
        self, capsys, tmp_path, monkeypatch
    ):
        from minipcai import paths

        monkeypatch.setattr(paths, "state_dir", lambda environ=None: tmp_path / "state")
        monkeypatch.setattr(paths, "is_source_checkout", lambda: False)
        code = main(["registry", "--json"])
        assert code == 0
        assert json.loads(capsys.readouterr().out)["stats"]["apps"] > 0


class TestPolicyFlag:
    def test_allow_untrusted_flag_is_honoured(self, capsys, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "apps": [
                {"id": "tool", "aliases": ["tool"],
                 "executable": r"C:\tools\tool.exe"},
            ]}),
            encoding="utf-8",
        )
        assert main(["registry", "--registry", str(path)]) == 2
        assert main(["registry", "--registry", str(path), "--allow-untrusted"]) == 0

    def test_environment_escape_hatch(self, capsys, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIPCAI_ALLOW_UNTRUSTED_APPS", "1")
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({"version": 1, "apps": [
                {"id": "tool", "aliases": ["tool"],
                 "executable": r"C:\tools\tool.exe"},
            ]}),
            encoding="utf-8",
        )
        assert main(["registry", "--registry", str(path)]) == 0
