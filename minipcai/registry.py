"""The validated target registry: the ONLY source of paths, executables and URLs.

The AI never generates Windows paths. Every actionable target (application,
file, folder, website, search endpoint) must be present in this registry, which
is maintained and reviewed by hand. Loading validates the registry strictly so
that broken, ambiguous or unsafe entries are rejected at startup rather than at
execution time:

* structural: globally unique ids, aliases unique *within a section* and
  compared folded (so ``"öffne"``/``"oeffne"`` cannot silently collide),
  absolute paths only, no ``..``, ``http(s)`` URLs only, ``searchable`` only on
  folders. The same alias may exist in two sections on purpose - the classifier
  decides which section is meant ("öffne google" opens the website, "google
  nach katzen" searches the web);
* policy: no blocked executable names, no shell-executable file extensions and
  (by default) applications only from trusted locations. Pass
  ``policy=...allow_untrusted_app_paths()`` to opt out deliberately;
* optional: ``check_files=True`` additionally verifies that every target exists
  on this machine and has the expected type (file/folder). That check is
  machine specific, so it is off by default and used by ``minipcai doctor`` and
  ``minipcai registry --check-files``.

Supported entry fields per section:

* ``apps``:     ``id``, ``aliases``, ``executable`` (absolute Windows path),
                optional ``sha256`` (pinned binary hash)
* ``files``:    ``id``, ``aliases``, ``path`` (absolute path)
* ``folders``:  ``id``, ``aliases``, ``path`` (absolute path), optional
                ``searchable`` (bool, enables the folder for ``find_file``)
* ``websites``: ``id``, ``aliases``, ``url`` (http/https)
* ``searchers``: ``id``, ``aliases``, ``url_template`` (http/https, must contain
                ``{query}``), optional ``searchable_from`` (folded keywords that
                hint at a web search)

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
from minipcai.policy import SecurityPolicy, default_policy, executable_name, expand_env
from minipcai.policy import extension_of as policy_extension_of
from minipcai.textutils import fold

_ENV_PREFIXED_PATH_RE = re.compile(r"^%[A-Za-z_][A-Za-z0-9_()]*%[\\/]")

# Required target field per section.
_TARGET_FIELD = {
    "apps": "executable",
    "files": "path",
    "folders": "path",
    "websites": "url",
    "searchers": "url_template",
}

_OPTIONAL_FIELDS = {
    "apps": frozenset({"sha256"}),
    "files": frozenset(),
    "folders": frozenset({"searchable"}),
    "websites": frozenset(),
    "searchers": frozenset(),
}


class RegistryError(ValueError):
    """Raised when the registry file is malformed, inconsistent or unsafe."""


def _is_absolute_path(value: str) -> bool:
    """Accept absolute Windows paths (drive letters, UNC, %VAR%-prefixed) and
    absolute POSIX paths."""
    if _ENV_PREFIXED_PATH_RE.match(value):
        return True
    return ntpath.isabs(value) or os.path.isabs(value)


_ENTRY_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_.\-]*$")


def _is_valid_entry_id(value: str) -> bool:
    """Entry ids are stable, lower-case, path-free identifiers."""
    return bool(_ENTRY_ID_RE.match(fold(value)))


def _has_no_traversal(value: str) -> bool:
    parts = re.split(r"[\\/]", value)
    return ".." not in parts


@dataclass(frozen=True)
class RegistryEntry:
    """One approved target from the registry."""

    section: str  # "apps" | "files" | "folders" | "websites" | "searchers"
    id: str
    aliases: tuple[str, ...]
    target: str  # executable path, file path, folder path, URL or URL template
    pinned_sha256: str = ""  # apps only, optional

    @property
    def path(self) -> str:
        """Expanded path (apps, files, folders)."""
        return expand_env(self.target)

    @property
    def url(self) -> str:
        """The approved URL (websites)."""
        return self.target

    @property
    def url_template(self) -> str:
        """The approved URL template (searchers)."""
        return self.target

    @property
    def extension(self) -> str:
        return policy_extension_of(self.target)

    def matches_hash_pin(self) -> bool:
        return bool(self.pinned_sha256)


@dataclass(frozen=True)
class Match:
    """An alias match of a registry entry inside user text."""

    entry: RegistryEntry
    alias: str

    @property
    def alias_length(self) -> int:
        return len(self.alias)


class Registry:
    """In-memory view of the validated registry, with a prebuilt alias index."""

    def __init__(
        self,
        entries: tuple[RegistryEntry, ...],
        version: int,
        searchable_ids: frozenset[str],
        path: Path | None = None,
        policy: SecurityPolicy | None = None,
    ):
        self._entries = entries
        self.version = version
        self._searchable_ids = searchable_ids
        self.path = path
        self.policy = policy or default_policy()
        self._by_id = {entry.id: entry for entry in entries}
        # section -> tuple of (folded alias, entry), longest alias first
        index: dict[str, list[tuple[str, RegistryEntry]]] = {
            section: [] for section in REGISTRY_SECTIONS
        }
        for entry in entries:
            for alias in entry.aliases:
                index[entry.section].append((fold(alias), entry))
        self._index = {
            section: tuple(sorted(pairs, key=lambda pair: len(pair[0]), reverse=True))
            for section, pairs in index.items()
        }
        self._folded_aliases: frozenset[str] = frozenset(
            alias for pairs in index.values() for alias, _ in pairs
        )

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

    def searchers(self) -> list[RegistryEntry]:
        return self.section("searchers")

    def folded_aliases(self) -> frozenset[str]:
        return self._folded_aliases

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(self._entries),
            "searchable_folders": len(self.searchable_folders()),
            **{section: len(self.section(section)) for section in REGISTRY_SECTIONS},
        }

    # -- alias matching -------------------------------------------------------
    def find_matches(self, text: str, section: str) -> list[Match]:
        """Find entries of ``section`` whose alias appears in ``text``.

        Matching is word-boundary based on the folded text (case/accent
        insensitive). Only the longest matching alias per entry counts; the
        result is sorted by alias length (longest first), which makes the most
        specific match win.
        """
        padded_text = f" {fold(text)} "
        matches: list[Match] = []
        seen_ids: set[str] = set()
        for alias, entry in self._index.get(section, ()):
            if entry.id in seen_ids or not alias:
                continue
            if f" {alias} " in padded_text:
                matches.append(Match(entry=entry, alias=alias))
                seen_ids.add(entry.id)
        matches.sort(key=lambda match: match.alias_length, reverse=True)
        return matches

    def find_matches_across_sections(self, text: str) -> dict[str, list[Match]]:
        return {section: self.find_matches(text, section) for section in REGISTRY_SECTIONS}

    # -- validation helpers ---------------------------------------------------
    def missing_targets(self) -> list[tuple[RegistryEntry, str]]:
        """Entries whose target does not exist (or has the wrong type).

        Returns ``(entry, reason)`` pairs; used by ``minipcai doctor`` and
        ``minipcai registry --check-files``. Machine specific by definition.
        """
        problems: list[tuple[RegistryEntry, str]] = []
        for entry in self._entries:
            if entry.section == "websites" or entry.section == "searchers":
                continue
            path = Path(entry.path)
            try:
                exists = path.exists()
            except OSError as exc:  # pragma: no cover - platform specific
                problems.append((entry, f"unreadable path ({exc})"))
                continue
            if not exists:
                problems.append((entry, "target does not exist on this machine"))
                continue
            if entry.section == "folders" and not path.is_dir():
                problems.append((entry, "target is not a directory"))
            if entry.section in {"apps", "files"} and path.is_dir():
                problems.append((entry, "target is a directory, expected a file"))
        return problems

    # -- loading --------------------------------------------------------------
    @classmethod
    def load_from_dict(
        cls,
        data: dict,
        path: Path | None = None,
        policy: SecurityPolicy | None = None,
        check_files: bool = False,
    ) -> Registry:
        """Validate an in-memory registry document (used by editors/tests)."""
        registry = cls._from_dict(data, path, policy or default_policy())
        if check_files:
            problems = registry.missing_targets()
            if problems:
                details = "; ".join(
                    f"{entry.section}/{entry.id}: {reason}" for entry, reason in problems
                )
                raise RegistryError(f"registry targets are not usable here: {details}")
        return registry

    @classmethod
    def load(
        cls,
        path: Path | str,
        policy: SecurityPolicy | None = None,
        check_files: bool = False,
    ) -> Registry:
        path = Path(path)
        if not path.is_file():
            raise RegistryError(f"registry file not found: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RegistryError(f"registry is not valid JSON: {exc}") from exc
        return cls.load_from_dict(data, path=path, policy=policy, check_files=check_files)

    @classmethod
    def _from_dict(
        cls, data: dict, path: Path | None, policy: SecurityPolicy
    ) -> Registry:
        if not isinstance(data, dict):
            raise RegistryError("registry root must be a JSON object")
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise RegistryError("registry must declare an integer 'version'")
        if version < 1:
            raise RegistryError("registry 'version' must be 1 or higher")
        unknown_keys = set(data) - {"version", *REGISTRY_SECTIONS}
        if unknown_keys:
            raise RegistryError(f"unknown registry keys: {sorted(unknown_keys)}")

        entries: list[RegistryEntry] = []
        searchable_ids: set[str] = set()
        seen_ids: set[str] = set()
        seen_aliases: dict[str, dict[str, str]] = {
            section: {} for section in REGISTRY_SECTIONS
        }  # section -> folded alias -> entry id

        for section in REGISTRY_SECTIONS:
            raw_entries = data.get(section, [])
            if not isinstance(raw_entries, list):
                raise RegistryError(f"registry section '{section}' must be a list")
            for index, raw in enumerate(raw_entries):
                entry = cls._validate_entry(section, raw, index, policy)
                if entry.id in seen_ids:
                    raise RegistryError(f"duplicate entry id '{entry.id}'")
                seen_ids.add(entry.id)
                for alias in entry.aliases:
                    normalized = fold(alias)
                    if not normalized:
                        raise RegistryError(
                            f"entry '{entry.id}' has an unusable alias {alias!r}"
                        )
                    section_aliases = seen_aliases[section]
                    if normalized in section_aliases:
                        owner = section_aliases[normalized]
                        raise RegistryError(
                            f"alias '{normalized}' is used by both '{owner}' and "
                            f"'{entry.id}'; aliases must be unique within a section"
                        )
                    section_aliases[normalized] = entry.id
                if "searchable" in raw and not isinstance(raw["searchable"], bool):
                    raise RegistryError(
                        f"entry '{entry.id}': 'searchable' must be a boolean"
                    )
                if raw.get("searchable") is True:
                    if section != "folders":
                        raise RegistryError(
                            f"entry '{entry.id}': only folders can be searchable"
                        )
                    searchable_ids.add(entry.id)
                entries.append(entry)

        return cls(
            entries=tuple(entries),
            version=version,
            searchable_ids=frozenset(searchable_ids),
            path=path,
            policy=policy,
        )

    @classmethod
    def _validate_entry(
        cls, section: str, raw: object, index: int, policy: SecurityPolicy
    ) -> RegistryEntry:
        where = f"{section}[{index}]"
        if not isinstance(raw, dict):
            raise RegistryError(f"{where}: entry must be a JSON object")
        # ``searchable`` is recognised in every section (so a misplaced flag
        # gets the specific "only folders can be searchable" error instead of a
        # generic "unknown fields" one) but only folders may set it.
        allowed_keys = (
            {"id", "aliases", _TARGET_FIELD[section], "searchable"}
            | _OPTIONAL_FIELDS[section]
        )
        unknown_keys = set(raw) - allowed_keys
        if unknown_keys:
            raise RegistryError(f"{where}: unknown fields {sorted(unknown_keys)}")
        entry_id = raw.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise RegistryError(f"{where}: 'id' must be a non-empty string")
        if not _is_valid_entry_id(entry_id):
            raise RegistryError(
                f"{where}: 'id' must start with a letter or digit and contain only "
                f"letters, digits, '_', '-' or '.' (got {entry_id!r})"
            )
        aliases = raw.get("aliases")
        if (
            not isinstance(aliases, list)
            or not aliases
            or not all(isinstance(alias, str) and alias.strip() for alias in aliases)
        ):
            raise RegistryError(f"{where}: 'aliases' must be a non-empty list of strings")
        field = _TARGET_FIELD[section]
        target = raw.get(field)
        if not isinstance(target, str) or not target.strip():
            raise RegistryError(f"{where}: '{field}' must be a non-empty string")

        pinned_sha256 = ""
        if section == "apps" and "sha256" in raw:
            value = raw["sha256"]
            if not isinstance(value, str) or len(value) != 64:
                raise RegistryError(
                    f"{where}: 'sha256' must be a 64 character hex digest"
                )
            try:
                int(value, 16)
            except ValueError as exc:
                raise RegistryError(
                    f"{where}: 'sha256' must be a 64 character hex digest"
                ) from exc
            pinned_sha256 = value.lower()

        if section in {"websites", "searchers"}:
            parsed = urlparse(target)
            if parsed.scheme not in ("http", "https") or not parsed.netloc:
                raise RegistryError(f"{where}: URL must be absolute http(s): {target!r}")
            if section == "searchers" and "{query}" not in target:
                raise RegistryError(
                    f"{where}: 'url_template' must contain the '{{query}}' placeholder"
                )
        else:
            expanded = expand_env(target)
            if not _is_absolute_path(expanded):
                raise RegistryError(f"{where}: target path must be absolute: {target!r}")
            if not _has_no_traversal(expanded):
                raise RegistryError(
                    f"{where}: target path must not contain '..': {target!r}"
                )
            policy_error = cls._static_policy_error(section, target, policy)
            if policy_error:
                raise RegistryError(f"{where}: {policy_error}")

        return RegistryEntry(
            section=section,
            id=entry_id,
            aliases=tuple(aliases),
            target=target,
            pinned_sha256=pinned_sha256,
        )

    @staticmethod
    def _static_policy_error(
        section: str, target: str, policy: SecurityPolicy
    ) -> str | None:
        """Load-time policy check; the same rules are re-checked at runtime."""
        if section == "apps":
            name = executable_name(expand_env(target))
            if not name.endswith(".exe"):
                return f"registered executable must be an .exe file: {target!r}"
            if name in policy.blocked_executables:
                return f"executable '{name}' is blocked by the security policy"
            if policy.enforce_trusted_app_roots and not policy.is_trusted_app_path(target):
                return (
                    f"application path '{target}' is outside the trusted application "
                    "locations. Move the application to Program Files/System32, add the "
                    "location to 'trusted_roots' in the configuration (or set "
                    "MINIPCAI_ALLOW_UNTRUSTED_APPS=1) if you really approved it"
                )
            return None
        if section == "files":
            reason = policy.is_blocked_file_target(target)
            if reason:
                return f"file target {target!r} is not openable safely: {reason}"
            return None
        if section == "folders":
            extension = policy_extension_of(expand_env(target))
            if extension in policy.blocked_file_extensions:
                return (
                    f"folder target {target!r} ends with the executable extension "
                    f"'{extension}' and is therefore refused"
                )
            return None
        return None


def registry_summary(registry: Registry) -> str:
    """Human-readable one-line summary for CLI/UI status output."""
    parts = [f"{len(registry.section(section))} {section}" for section in REGISTRY_SECTIONS]
    return f"registry v{registry.version}: " + ", ".join(parts)
