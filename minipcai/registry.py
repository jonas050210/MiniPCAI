"""The validated target registry: the ONLY source of paths, executables and URLs.

The AI never generates Windows paths. Every actionable target (application,
file, folder, website) must be present in this registry, which is maintained
and reviewed by hand. Loading validates the registry strictly so that broken,
ambiguous or unsafe entries are rejected at startup rather than at execution
time.

Supported entry fields per section:

* ``apps``:     ``id``, ``aliases``, ``executable`` (absolute Windows path)
* ``files``:    ``id``, ``aliases``, ``path`` (absolute path)
* ``folders``:  ``id``, ``aliases``, ``path`` (absolute path), optional
                ``searchable`` (bool, enables the folder for ``find_file``)
* ``websites``: ``id``, ``aliases``, ``url`` (http/https)

Paths may contain Windows environment variables such as ``%USERPROFILE%``;
they are expanded at load time.
"""

from __future__ import annotations

import json
import ntpath
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from minipcai.intents import REGISTRY_SECTIONS
from minipcai.textutils import normalize

_ENV_VAR_RE = re.compile(r"%([A-Za-z_][A-Za-z0-9_]*)%")
# A Windows path that begins with an environment variable, e.g.
# "%USERPROFILE%\Documents\notes.txt". Treated as absolute: the placeholder is
# expanded at runtime on the target machine.
_ENV_PREFIXED_PATH_RE = re.compile(r"^%[A-Za-z_][A-Za-z0-9_]*%[\\/]")

# Required target field per section.
_TARGET_FIELD = {
    "apps": "executable",
    "files": "path",
    "folders": "path",
    "websites": "url",
}


class RegistryError(ValueError):
    """Raised when the registry file is malformed, inconsistent or unsafe."""


def _expand_env(value: str) -> str:
    """Expand ``%VAR%`` placeholders; unknown variables are left untouched."""

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    return _ENV_VAR_RE.sub(replace, value)


def _is_absolute_path(value: str) -> bool:
    """Accept absolute Windows paths (drive letters, UNC, %VAR%-prefixed) and
    absolute POSIX paths."""
    if _ENV_PREFIXED_PATH_RE.match(value):
        return True
    return ntpath.isabs(value) or os.path.isabs(value)


def _has_no_traversal(value: str) -> bool:
    parts = re.split(r"[\\/]", value)
    return ".." not in parts


@dataclass(frozen=True)
class RegistryEntry:
    """One approved target from the registry."""

    section: str  # "apps" | "files" | "folders" | "websites"
    id: str
    aliases: tuple[str, ...]
    target: str  # executable path, file path, folder path or URL

    @property
    def path(self) -> str:
        """Expanded path (apps, files, folders)."""
        return _expand_env(self.target)

    @property
    def url(self) -> str:
        """The approved URL (websites)."""
        return self.target


@dataclass(frozen=True)
class Match:
    """An alias match of a registry entry inside user text."""

    entry: RegistryEntry
    alias: str

    @property
    def alias_length(self) -> int:
        return len(self.alias)


class Registry:
    """In-memory view of the validated registry."""

    def __init__(self, entries: tuple[RegistryEntry, ...], version: int,
                 searchable_ids: frozenset[str], path: Path | None = None):
        self._entries = entries
        self.version = version
        self._searchable_ids = searchable_ids
        self.path = path
        self._by_id = {entry.id: entry for entry in entries}

    # -- lookup -------------------------------------------------------------
    @property
    def entries(self) -> tuple[RegistryEntry, ...]:
        return self._entries

    def by_id(self, entry_id: str) -> RegistryEntry | None:
        return self._by_id.get(entry_id)

    def section(self, section: str) -> list[RegistryEntry]:
        return [entry for entry in self._entries if entry.section == section]

    def searchable_folders(self) -> list[RegistryEntry]:
        return [
            entry
            for entry in self._entries
            if entry.section == "folders" and entry.id in self._searchable_ids
        ]

    # -- alias matching -------------------------------------------------------
    def find_matches(self, text: str, section: str) -> list[Match]:
        """Find entries of ``section`` whose alias appears in ``text``.

        Matching is word-boundary based on the normalized text. Only the
        longest matching alias per entry counts; the result is sorted by alias
        length (longest first).
        """
        padded_text = f" {normalize(text)} "
        matches: list[Match] = []
        for entry in self.section(section):
            best_alias: str | None = None
            for alias in entry.aliases:
                normalized_alias = normalize(alias)
                if not normalized_alias:
                    continue
                if f" {normalized_alias} " in padded_text:
                    if best_alias is None or len(normalized_alias) > len(best_alias):
                        best_alias = normalized_alias
            if best_alias is not None:
                matches.append(Match(entry=entry, alias=best_alias))
        matches.sort(key=lambda match: match.alias_length, reverse=True)
        return matches

    def find_matches_across_sections(self, text: str) -> dict[str, list[Match]]:
        return {
            section: self.find_matches(text, section) for section in REGISTRY_SECTIONS
        }

    # -- loading --------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> Registry:
        path = Path(path)
        if not path.is_file():
            raise RegistryError(f"registry file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry is not valid JSON: {exc}") from exc
        return cls._from_dict(data, path)

    @classmethod
    def _from_dict(cls, data: dict, path: Path | None) -> Registry:
        if not isinstance(data, dict):
            raise RegistryError("registry root must be a JSON object")
        if not isinstance(data.get("version"), int):
            raise RegistryError("registry must declare an integer 'version'")
        unknown_keys = set(data) - {"version", *REGISTRY_SECTIONS}
        if unknown_keys:
            raise RegistryError(f"unknown registry keys: {sorted(unknown_keys)}")

        entries: list[RegistryEntry] = []
        searchable_ids: set[str] = set()
        seen_ids: set[str] = set()
        seen_aliases: dict[str, str] = {}  # normalized alias -> entry id

        for section in REGISTRY_SECTIONS:
            raw_entries = data.get(section, [])
            if not isinstance(raw_entries, list):
                raise RegistryError(f"registry section '{section}' must be a list")
            for index, raw in enumerate(raw_entries):
                entry = cls._validate_entry(section, raw, index)
                if entry.id in seen_ids:
                    raise RegistryError(f"duplicate entry id '{entry.id}'")
                seen_ids.add(entry.id)
                for alias in entry.aliases:
                    normalized = normalize(alias)
                    if not normalized:
                        raise RegistryError(
                            f"entry '{entry.id}' has an unusable alias {alias!r}"
                        )
                    if normalized in seen_aliases:
                        owner = seen_aliases[normalized]
                        raise RegistryError(
                            f"alias '{normalized}' is used by both '{owner}' and "
                            f"'{entry.id}'; aliases must be unique across the registry"
                        )
                    seen_aliases[normalized] = entry.id
                if raw.get("searchable") is True:
                    if section != "folders":
                        raise RegistryError(
                            f"entry '{entry.id}': only folders can be searchable"
                        )
                    searchable_ids.add(entry.id)
                entries.append(entry)

        return cls(
            entries=tuple(entries),
            version=data["version"],
            searchable_ids=frozenset(searchable_ids),
            path=path,
        )

    @classmethod
    def _validate_entry(cls, section: str, raw: object, index: int) -> RegistryEntry:
        where = f"{section}[{index}]"
        if not isinstance(raw, dict):
            raise RegistryError(f"{where}: entry must be a JSON object")
        unknown_keys = set(raw) - {"id", "aliases", "searchable", _TARGET_FIELD[section]}
        if unknown_keys:
            raise RegistryError(f"{where}: unknown fields {sorted(unknown_keys)}")
        entry_id = raw.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise RegistryError(f"{where}: 'id' must be a non-empty string")
        aliases = raw.get("aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
        ):
            raise RegistryError(f"{where}: 'aliases' must be a non-empty list of strings")
        target = raw.get(_TARGET_FIELD[section])
        if not isinstance(target, str) or not target.strip():
            raise RegistryError(
                f"{where}: '{_TARGET_FIELD[section]}' must be a non-empty string"
            )
        if section == "websites":
            parsed = urlparse(target)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise RegistryError(f"{where}: website URL must be absolute http(s): {target!r}")
        else:
            expanded = _expand_env(target)
            if not _is_absolute_path(expanded):
                raise RegistryError(
                    f"{where}: target path must be absolute: {target!r}"
                )
            if not _has_no_traversal(expanded):
                raise RegistryError(f"{where}: target path must not contain '..': {target!r}")
        return RegistryEntry(
            section=section, id=entry_id, aliases=tuple(aliases), target=target
        )


def registry_summary(registry: Registry) -> str:
    """Human-readable one-line summary for CLI/UI status output."""
    parts = [
        f"{len(registry.section(section))} {section}" for section in REGISTRY_SECTIONS
    ]
    return f"registry v{registry.version}: " + ", ".join(parts)
