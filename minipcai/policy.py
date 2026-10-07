"""The central security policy: what may be executed, opened and confirmed.

MiniPCAI already refuses requests instead of guessing, resolves targets only
from the registry and never builds a command line from user text. This module
adds the *policy layer* on top of those invariants:

* which executable names are never launchable (shells, script hosts, LOLBins),
* which file extensions may never be opened through the shell
  (``.exe``, ``.lnk``, ``.bat``, ... - the "open a file" primitive would
  otherwise turn into "execute a program"),
* which folders count as trusted locations for approved applications
  (an allowlist of names is weaker than an allowlist of names *and* locations),
* which intents additionally require the user's explicit confirmation,
* hard limits for stateful actions (active timers).

Everything is data, so a deployment can tighten (or deliberately loosen) the
policy through configuration without touching code. The defaults are the
strict ones: an "installed" assistant is expected to run unmodified.
"""

from __future__ import annotations

import ntpath
import os
import re
from dataclasses import dataclass, replace

# ---------------------------------------------------------------------------
# Executables that must never be launchable through MiniPCAI, even if someone
# adds them to the registry. MiniPCAI has no use case for shells, script hosts
# and interpreter/loader helpers; opening one would defeat the whole security
# model because the assistant would then be a generic command runner.
# ---------------------------------------------------------------------------
BLOCKED_EXECUTABLES: frozenset[str] = frozenset(
    {
        # Windows command interpreters and script hosts
        "cmd.exe",
        "powershell.exe",
        "powershell_ise.exe",
        "pwsh.exe",
        "wscript.exe",
        "cscript.exe",
        "mshta.exe",
        # Troubleshooter/proxy execution helpers (known LOLBin abuse)
        "msdt.exe",
        "fodhelper.exe",
        "computerdefaults.exe",
        # Registration / control-panel / loader tooling
        "regedit.exe",
        "reg.exe",
        "regsvr32.exe",
        "rundll32.exe",
        "control.exe",
        "mmc.exe",
        "certutil.exe",
        "bitsadmin.exe",
        "installutil.exe",
        "regasm.exe",
        "regsvcs.exe",
        "msbuild.exe",
        "msiexec.exe",
        "schtasks.exe",
        "wmic.exe",
        "forfiles.exe",
        "pcalua.exe",
        "runas.exe",
        "wt.exe",
        # POSIX shells inside Windows
        "wsl.exe",
        "bash.exe",
        "sh.exe",
        "dash.exe",
        "zsh.exe",
        # Console host (never useful to launch directly)
        "conhost.exe",
        # Interpreters that would accept file arguments as code
        "python.exe",
        "pythonw.exe",
        "py.exe",
        "node.exe",
        "cscript",
        "explorer.exe",
    }
)

# ---------------------------------------------------------------------------
# Extensions that must never be opened through the shell, no matter which
# registry section they appear in. ``os.startfile`` on any of these executes
# code (directly, or indirectly through the shell association).
# ---------------------------------------------------------------------------
BLOCKED_FILE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".exe", ".com", ".scr", ".pif", ".cpl", ".msc",
        ".bat", ".cmd", ".btm",
        ".ps1", ".psm1", ".psd1", ".ps1xml",
        ".vbs", ".vbe", ".vb", ".vbscript",
        ".js", ".jse", ".wsf", ".wsh", ".ws", ".hta",
        ".lnk", ".url", ".website", ".job", ".sct", ".shb",
        ".reg", ".inf", ".ins", ".isp",
        ".msi", ".msp", ".mst", ".msu", ".cab",
        ".dll", ".ocx", ".sys", ".drv", ".efi",
        ".jar", ".class", ".jnlp",
        ".sh", ".bash", ".zsh", ".ksh", ".csh", ".run", ".bin",
        ".py", ".pyw", ".pyc", ".rb", ".pl", ".php", ".lua", ".tcl", ".ahk", ".au3",
        ".appref-ms", ".application", ".deploy", ".gadget", ".theme", ".themepack",
        ".iso", ".img", ".vhd", ".vhdx", ".wim",
    }
)

# Extensions that are typical for the "open a registered document" use case.
# Only enforced when a deployment opts into the allowlist mode.
DEFAULT_DOCUMENT_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".txt", ".md", ".rst", ".log", ".csv", ".tsv", ".json", ".xml", ".yaml", ".yml",
        ".ini", ".cfg", ".toml",
        ".pdf", ".doc", ".docx", ".odt", ".rtf", ".xls", ".xlsx", ".ods", ".ppt", ".pptx",
        ".odp", ".epub", ".mobi",
        ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".tif", ".tiff", ".svg",
        ".mp3", ".m4a", ".wav", ".flac", ".ogg", ".mp4", ".mkv", ".avi", ".mov", ".webm",
        ".zip", ".gz", ".7z", ".tar",
    }
)

# Windows-style locations that are considered safe homes for approved
# applications. Paths are compared after ``%VAR%`` expansion in a
# case-insensitive, separator-insensitive way. The placeholders deliberately
# survive on non-Windows systems, so a registry written for Windows is still
# validated identically on a development machine.
# Documented Windows default locations, used only when a placeholder above
# cannot be resolved on the current machine (e.g. a Linux CI runner validating
# a Windows registry).
_WINDOWS_LITERAL_FALLBACKS: dict[str, tuple[str, ...]] = {
    r"%ProgramFiles%": (r"C:\Program Files",),
    r"%ProgramFiles(x86)%": (r"C:\Program Files (x86)",),
    r"%SystemRoot%\System32": (r"C:\Windows\System32",),
    r"%SystemRoot%": (r"C:\Windows",),
    r"%ProgramData%": (r"C:\ProgramData",),
}

DEFAULT_TRUSTED_APP_ROOTS: tuple[str, ...] = (
    r"%ProgramFiles%",
    r"%ProgramFiles(x86)%",
    r"%SystemRoot%\System32",
    r"%SystemRoot%",
    r"%ProgramData%",
    r"%LOCALAPPDATA%\Programs",
    r"%LOCALAPPDATA%\Microsoft\WindowsApps",
)

# Intents that change the state of the machine in a way the user can notice or
# regret (closing an application can discard unsaved work) or that leave the
# local machine (web search). They require an explicit confirmation.
DEFAULT_CONFIRM_INTENTS: frozenset[str] = frozenset({"close_app", "web_search"})

_DEFAULT_MAX_ACTIVE_TIMERS = 32

_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")
_UNC_RE = re.compile(r"^\\\\[^\\/]+[\\/][^\\/]+")
_ENV_RE = re.compile(r"^%[A-Za-z_][A-Za-z0-9_()]*%[\\/]")


def is_windows_style_path(value: str) -> bool:
    """Heuristically decide whether ``value`` is a Windows path.

    Trusted-root enforcement only makes sense for Windows-shaped paths, so
    POSIX paths (development machines, tests) are left alone.
    """
    value = value.strip()
    return bool(_DRIVE_RE.match(value) or _UNC_RE.match(value) or _ENV_RE.match(value))


# A path *inside* a longer sentence: a drive letter, a UNC prefix, a
# ``%VAR%\`` prefix, a traversal segment or an absolute POSIX path with more
# than one separator. Bare fractions such as "2/3" or "3/4" must not match,
# otherwise calculator input would be refused.
_EMBEDDED_PATH_RE = re.compile(
    r"[A-Za-z]:[\\/]"                      # C:\ or C:/
    r"|\\\\[^\\/\s]+[\\/]"                 # \\server\share
    r"|%[A-Za-z_][A-Za-z0-9_()]*%[\\/]"    # %SystemRoot%\
    r"|(?:^|[\s\\/])\.\.(?:[\\/]|$|\s)"    # .. as a path segment
    r"|(?:^|\s)/(?:[^\s/]+/)+"             # /etc/passwd (2+ separators)
)


def looks_like_path(value: str) -> bool:
    """True when a request contains a filesystem path instead of a plain name.

    MiniPCAI resolves every target through the registry and never builds paths
    from user text, so a request that spells out a path is refused before the
    model even runs. The check is deliberately narrow: drive letters, UNC
    prefixes, traversal segments and absolute POSIX paths count, ordinary words
    with slashes (fractions) do not.
    """
    text = value.strip()
    if not text:
        return False
    if text.startswith(("/", "\\")):
        return True
    if _DRIVE_RE.match(text) or _ENV_RE.match(text):
        return True
    return bool(_EMBEDDED_PATH_RE.search(text))


def normalize_windows_path(value: str) -> str:
    """Case-fold and normalize separators of a Windows path for comparison."""
    return ntpath.normcase(ntpath.normpath(value.strip()))


def is_inside_path(path: str, root: str) -> bool:
    """True when ``path`` equals ``root`` or lives below it."""
    candidate = normalize_windows_path(path)
    parent = normalize_windows_path(root).rstrip("\\/")
    if not parent:
        return False
    return candidate == parent or candidate.startswith(parent + "\\")


def extension_of(path: str) -> str:
    """Lowercase file extension including the dot (``""`` when there is none)."""
    name = path.replace("/", "\\").rsplit("\\", 1)[-1]
    if "." not in name.strip("."):
        return ""
    return "." + name.rsplit(".", 1)[-1].lower()


def executable_name(path: str) -> str:
    """Lowercase base name of a path (``""`` when the path is empty)."""
    return path.replace("/", "\\").rsplit("\\", 1)[-1].strip().lower()


def expand_env(value: str) -> str:
    """Expand ``%VAR%`` placeholders; unknown variables are left untouched."""

    def replace(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), match.group(0))

    return re.sub(r"%([A-Za-z_][A-Za-z0-9_()]*)%", replace, value)


@dataclass(frozen=True)
class SecurityPolicy:
    """A complete, immutable description of the security-relevant policy."""

    blocked_executables: frozenset[str] = BLOCKED_EXECUTABLES
    blocked_file_extensions: frozenset[str] = BLOCKED_FILE_EXTENSIONS
    # ``None`` disables the allowlist and keeps the (smaller) denylist mode.
    allowed_file_extensions: frozenset[str] | None = None
    trusted_app_roots: tuple[str, ...] = DEFAULT_TRUSTED_APP_ROOTS
    enforce_trusted_app_roots: bool = True
    require_file_extension: bool = True
    confirm_intents: frozenset[str] = DEFAULT_CONFIRM_INTENTS
    #: Seconds a pending confirmation stays valid (stale confirmations must not
    #: be able to trigger an action much later).
    confirmation_timeout: float = 120.0
    max_active_timers: int = _DEFAULT_MAX_ACTIVE_TIMERS
    # Optional integrity pinning: ``{"<expanded path>": "<sha256>"}``.
    executable_hashes: tuple[tuple[str, str], ...] = ()

    # -- derived views -------------------------------------------------------
    def trusted_roots_expanded(self) -> tuple[str, ...]:
        """Trusted prefixes with environment variables resolved.

        On machines where a placeholder cannot be resolved (a Linux CI box
        validating a Windows registry) well-known literal locations are used
        instead, so validation behaves the same everywhere.
        """
        prefixes: list[str] = []
        for root in self.trusted_app_roots:
            expanded = expand_env(root)
            prefixes.append(expanded)
            if "%" in expanded:
                # Unresolvable placeholder: keep the placeholder form and add
                # the documented Windows default location.
                prefixes.append(root)
                for literal in _WINDOWS_LITERAL_FALLBACKS.get(root, ()):
                    prefixes.append(literal)
        return tuple(prefixes)

    def is_trusted_app_path(self, path: str) -> bool:
        """Whether ``path`` (Windows-shaped) lives in a trusted app location."""
        if not is_windows_style_path(path):
            # POSIX paths are only used on development machines; the Windows
            # executor refuses to act there anyway.
            return True
        expanded = expand_env(path)
        return any(
            is_inside_path(expanded, root) for root in self.trusted_roots_expanded()
        )

    def is_blocked_file_target(self, path: str) -> str | None:
        """Return a human-readable reason when ``path`` must not be opened."""
        extension = extension_of(expand_env(path))
        if extension and extension in self.blocked_file_extensions:
            return f"the file type '{extension}' can execute code"
        if not extension and self.require_file_extension:
            return "the file has no extension, so its type cannot be verified"
        allowed = self.allowed_file_extensions
        if allowed is not None and extension and extension not in allowed:
            return f"the file type '{extension}' is not in the allowed document types"
        return None

    def hash_for(self, path: str) -> str | None:
        expanded = expand_env(path)
        for pinned_path, digest in self.executable_hashes:
            if normalize_windows_path(expand_env(pinned_path)) == normalize_windows_path(
                expanded
            ):
                return digest
        return None

    # -- tweaks --------------------------------------------------------------
    def allow_untrusted_app_paths(self) -> SecurityPolicy:
        """Return a copy that does not enforce the trusted-root allowlist."""
        return replace(self, enforce_trusted_app_roots=False)

    def with_confirmations(self, intents: frozenset[str]) -> SecurityPolicy:
        return replace(self, confirm_intents=intents)

    def without_confirmations(self) -> SecurityPolicy:
        """Return a copy that performs every action without asking."""
        return replace(self, confirm_intents=frozenset())

    def with_confirmation_timeout(self, seconds: float) -> SecurityPolicy:
        return replace(self, confirmation_timeout=max(0.0, float(seconds)))


DEFAULT_POLICY = SecurityPolicy()


def default_policy() -> SecurityPolicy:
    """The strict, built-in policy."""
    return DEFAULT_POLICY


def policy_from_env(environ: dict[str, str] | None = None) -> SecurityPolicy:
    """Build the policy from environment overrides (documented escape hatches).

    * ``MINIPCAI_ALLOW_UNTRUSTED_APPS=1`` disables the trusted-root allowlist.
    * ``MINIPCAI_NO_CONFIRM=1`` disables the confirmation prompts
      (intended for scripted, non-interactive use; the audit log still records
      every request).
    * ``MINIPCAI_EXTRA_TRUSTED_ROOTS`` is a ``;``-separated list of additional
      trusted application roots.
    """
    env = os.environ if environ is None else environ
    policy = DEFAULT_POLICY
    if env.get("MINIPCAI_ALLOW_UNTRUSTED_APPS", "").strip() in {"1", "true", "yes"}:
        policy = policy.allow_untrusted_app_paths()
    if env.get("MINIPCAI_NO_CONFIRM", "").strip() in {"1", "true", "yes"}:
        policy = policy.without_confirmations()
    extra = env.get("MINIPCAI_EXTRA_TRUSTED_ROOTS", "").strip()
    if extra:
        roots = tuple(part.strip() for part in extra.split(";") if part.strip())
        policy = replace(policy, trusted_app_roots=policy.trusted_app_roots + roots)
    return policy
