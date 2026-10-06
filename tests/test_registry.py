"""Tests for registry loading, validation and alias matching."""

from __future__ import annotations

import json

import pytest

from minipcai.registry import Registry, RegistryError, registry_summary


class TestLoading:
    def test_loads_valid_registry(self, registry_factory):
        registry = Registry.load(registry_factory())
        assert registry.version == 1
        assert len(registry.entries) == 10
        assert registry_summary(registry).startswith("registry v1")

    def test_missing_file(self, tmp_path):
        with pytest.raises(RegistryError, match="not found"):
            Registry.load(tmp_path / "nope.json")

    def test_invalid_json(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text("{oops", encoding="utf-8")
        with pytest.raises(RegistryError, match="not valid JSON"):
            Registry.load(path)

    def test_missing_version(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"apps": []}), encoding="utf-8")
        with pytest.raises(RegistryError, match="version"):
            Registry.load(path)

    def test_unknown_top_level_key(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"version": 1, "printers": []}), encoding="utf-8")
        with pytest.raises(RegistryError, match="unknown registry keys"):
            Registry.load(path)

    def test_unknown_entry_field(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "apps": [{"id": "a", "aliases": ["a"], "executable": "C:\\a.exe", "evil": 1}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="unknown fields"):
            Registry.load(path)

    def test_duplicate_alias_across_entries_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "apps": [
                    {"id": "a", "aliases": ["editor"], "executable": "C:\\a.exe"},
                    {"id": "b", "aliases": ["Editor"], "executable": "C:\\b.exe"},
                ],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="alias 'editor' is used by both"):
            Registry.load(path)

    def test_duplicate_id_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "apps": [{"id": "a", "aliases": ["x"], "executable": "C:\\a.exe"}],
                "files": [{"id": "a", "aliases": ["y"], "path": "C:\\a.txt"}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="duplicate entry id"):
            Registry.load(path)

    @pytest.mark.parametrize(
        "target",
        [
            "C:\\Windows\\..\\System32\\cmd.exe",   # traversal
            "notepad.exe",                          # relative
            "folder\\sub",                          # relative
        ],
    )
    def test_invalid_paths_rejected(self, tmp_path, target):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "files": [{"id": "f", "aliases": ["f"], "path": target}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError):
            Registry.load(path)

    def test_bad_url_scheme_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "websites": [{"id": "w", "aliases": ["w"], "url": "ftp://example.com"}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="http"):
            Registry.load(path)

    def test_searchable_only_on_folders(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "files": [{"id": "f", "aliases": ["f"], "path": "C:\\f.txt",
                           "searchable": True}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="only folders can be searchable"):
            Registry.load(path)

    def test_zero_version_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"version": 0}), encoding="utf-8")
        with pytest.raises(RegistryError, match="1 or higher"):
            Registry.load(path)

    def test_boolean_version_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps({"version": True}), encoding="utf-8")
        with pytest.raises(RegistryError, match="integer"):
            Registry.load(path)

    def test_non_boolean_searchable_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "folders": [{"id": "f", "aliases": ["f"], "path": "C:\\f",
                             "searchable": "yes"}],
            }),
            encoding="utf-8",
        )
        with pytest.raises(RegistryError, match="must be a boolean"):
            Registry.load(path)

    def test_env_prefixed_windows_path_accepted_without_env_var(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "files": [{"id": "f", "aliases": ["f"],
                           "path": "%SOME_UNDEFINED_VAR%\\Documents\\f.txt"}],
            }),
            encoding="utf-8",
        )
        registry = Registry.load(path)
        assert registry.by_id("f") is not None


class TestEnvironmentExpansion:
    def test_expands_known_variable(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MINIPCAI_TEST_HOME", str(tmp_path))
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps({
                "version": 1,
                "files": [{"id": "f", "aliases": ["f"],
                           "path": "%MINIPCAI_TEST_HOME%\\notes.txt"}],
            }),
            encoding="utf-8",
        )
        registry = Registry.load(path)
        assert registry.by_id("f").path == str(tmp_path) + "\\notes.txt"

    def test_unknown_variable_left_untouched(self, registry_factory):
        registry = Registry.load(registry_factory())
        # fixture paths are plain absolute paths; expansion is a no-op
        for entry in registry.section("apps"):
            assert "%" not in entry.path


class TestAliasMatching:
    def test_simple_match(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("öffne notepad", "apps")
        assert [m.entry.id for m in matches] == ["notepad"]

    def test_match_ignores_case_and_punctuation(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("Mach den Download-Ordner auf!", "folders")
        assert [m.entry.id for m in matches] == ["downloads"]

    def test_multiword_alias(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("öffne den download ordner", "folders")
        assert [m.entry.id for m in matches] == ["downloads"]
        assert matches[0].alias == "download ordner"

    def test_no_match(self, registry_factory):
        registry = Registry.load(registry_factory())
        assert registry.find_matches("starte gimp", "apps") == []

    def test_word_boundary_prevents_substring_matches(self, registry_factory):
        registry = Registry.load(registry_factory())
        assert registry.find_matches("öffne wikinger", "websites") == []

    def test_multiple_distinct_entries_match(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("öffne firefox und chrome", "apps")
        assert {m.entry.id for m in matches} == {"firefox", "chrome"}

    def test_same_entry_matched_via_two_aliases_counts_once(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("öffne wikipedia das wiki", "websites")
        assert [m.entry.id for m in matches] == ["wikipedia"]

    def test_longest_alias_wins_for_sorting(self, registry_factory):
        registry = Registry.load(registry_factory())
        matches = registry.find_matches("öffne den browser firefox", "apps")
        assert matches[0].alias == "firefox"


class TestSearchableFolders:
    def test_only_searchable_folders_listed(self, registry_factory):
        registry = Registry.load(registry_factory())
        ids = {entry.id for entry in registry.searchable_folders()}
        assert ids == {"downloads", "documents"}
