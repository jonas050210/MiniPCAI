"""Tests for the security policy (blocklists, trusted roots, escape hatches)."""

from __future__ import annotations

import pytest

from minipcai.policy import (
    BLOCKED_EXECUTABLES,
    BLOCKED_FILE_EXTENSIONS,
    DEFAULT_POLICY,
    SecurityPolicy,
    default_policy,
    executable_name,
    expand_env,
    extension_of,
    is_inside_path,
    is_windows_style_path,
    looks_like_path,
    normalize_windows_path,
    policy_from_env,
)


class TestPathHelpers:
    def test_is_windows_style_path(self):
        assert is_windows_style_path(r"C:\Windows\System32\notepad.exe")
        assert is_windows_style_path(r"\\server\share\tool.exe")
        assert is_windows_style_path(r"%ProgramFiles%\Tool\tool.exe")
        assert not is_windows_style_path("/usr/bin/notepad")
        assert not is_windows_style_path("notepad")

    def test_is_inside_path(self):
        assert is_inside_path(r"C:\Program Files\Tool\t.exe", r"C:\Program Files")
        assert is_inside_path(r"C:\Program Files", r"C:\Program Files\\")
        assert not is_inside_path(r"C:\Program Files (x86)\Tool\t.exe", r"C:\Program Files")
        assert not is_inside_path(r"C:\ProgramFiles\Tool\t.exe", r"C:\Program Files")

    def test_normalize_windows_path(self):
        assert normalize_windows_path(r"C:\Windows\..\Windows\System32\A.EXE") == (
            r"c:\windows\system32\a.exe"
        )

    def test_extension_and_executable_name(self):
        assert extension_of(r"C:\a\b\Tool.EXE") == ".exe"
        assert extension_of("notes") == ""
        assert extension_of(r"C:\a\.hidden") == ""
        assert executable_name(r"C:\Dir\Tool.Exe") == "tool.exe"
        assert executable_name("") == ""

    def test_expand_env_leaves_unknown_placeholders(self):
        assert expand_env(r"%NOT_A_REAL_VAR%\x") == r"%NOT_A_REAL_VAR%\x"


class TestBlocklists:
    def test_blocked_executables_cover_shells_and_interpreters(self):
        for name in ("cmd.exe", "powershell.exe", "wscript.exe", "python.exe", "node.exe"):
            assert name in BLOCKED_EXECUTABLES

    def test_blocked_extensions_cover_executables_and_shortcuts(self):
        for extension in (".exe", ".lnk", ".bat", ".ps1", ".vbs", ".msi", ".jar"):
            assert extension in BLOCKED_FILE_EXTENSIONS

    def test_blocked_file_target_reports_reason(self):
        policy = default_policy()
        assert "can execute code" in policy.is_blocked_file_target(r"C:\x\tool.exe")
        assert "can execute code" in policy.is_blocked_file_target(r"C:\x\shortcut.lnk")
        assert policy.is_blocked_file_target(r"C:\x\notes.txt") is None

    def test_extensionless_file_is_refused_by_default(self):
        policy = default_policy()
        reason = policy.is_blocked_file_target(r"C:\x\startup")
        assert reason is not None and "no extension" in reason
        lenient = SecurityPolicy(require_file_extension=False)
        assert lenient.is_blocked_file_target(r"C:\x\startup") is None

    def test_extension_allowlist_mode(self):
        policy = SecurityPolicy(allowed_file_extensions=frozenset({".txt"}))
        assert policy.is_blocked_file_target(r"C:\x\notes.txt") is None
        assert "not in the allowed document types" in policy.is_blocked_file_target(
            r"C:\x\photo.png"
        )

    def test_blocked_or_extensionless_targets_stay_refused_for_any_registry(self):
        # Defense in depth: the policy level refuses them regardless of the
        # registry the file came from.
        policy = default_policy()
        for target in (r"C:\x\evil.exe", r"C:\x\evil.lnk", r"C:\x\evil.bat"):
            assert policy.is_blocked_file_target(target) is not None


class TestTrustedRoots:
    def test_default_trusted_roots_include_program_files_and_system32(self):
        policy = default_policy()
        assert policy.is_trusted_app_path(r"C:\Program Files\Tool\tool.exe")
        assert policy.is_trusted_app_path(r"C:\Windows\System32\notepad.exe")
        assert not policy.is_trusted_app_path(r"C:\Users\me\Downloads\tool.exe")
        assert not policy.is_trusted_app_path(r"C:\tools\tool.exe")

    def test_posix_paths_are_not_judged(self, tmp_path):
        policy = default_policy()
        assert policy.is_trusted_app_path(str(tmp_path / "app.exe"))

    def test_allow_untrusted_copy(self):
        policy = default_policy()
        lenient = policy.allow_untrusted_app_paths()
        assert lenient.enforce_trusted_app_roots is False
        # deriving a copy must not mutate the original
        assert policy.enforce_trusted_app_roots is True

    def test_extra_roots_are_honoured(self):
        policy = SecurityPolicy(trusted_app_roots=(r"C:\tools",))
        assert policy.is_trusted_app_path(r"C:\tools\app.exe")
        assert not policy.is_trusted_app_path(r"C:\other\app.exe")

    def test_confirmation_intents(self):
        policy = default_policy()
        assert {"close_app", "web_search"} <= set(policy.confirm_intents)
        assert policy.without_confirmations().confirm_intents == frozenset()
        assert policy.with_confirmations(frozenset({"open_app"})).confirm_intents == (
            frozenset({"open_app"})
        )

    def test_executable_hash_pinning_lookup(self):
        policy = SecurityPolicy(
            executable_hashes=((r"C:\Program Files\Tool\tool.exe", "ab" * 32),)
        )
        assert policy.hash_for(r"c:\program files\tool\TOOL.EXE") == "ab" * 32
        assert policy.hash_for(r"C:\Program Files\Tool\other.exe") is None


class TestPolicyFromEnv:
    def test_defaults(self):
        policy = policy_from_env({})
        assert policy is DEFAULT_POLICY

    @pytest.mark.parametrize("value", ["1", "true", "yes"])
    def test_allow_untrusted(self, value, tmp_path):
        from minipcai.registry import Registry

        policy = policy_from_env({"MINIPCAI_ALLOW_UNTRUSTED_APPS": value})
        assert policy.enforce_trusted_app_roots is False
        # ``is_trusted_app_path`` still answers the *trust* question; the
        # enforcement flag is what decides whether it is applied.
        assert policy.is_trusted_app_path(r"C:\tools\tool.exe") is False
        path = tmp_path / "registry.json"
        path.write_text(
            '{"version": 1, "apps": [{"id": "tool", "aliases": ["tool"],'
            ' "executable": "C:\\\\tools\\\\tool.exe"}]}',
            encoding="utf-8",
        )
        assert Registry.load(path, policy=policy).by_id("tool").target.endswith("tool.exe")

    def test_no_confirm(self):
        policy = policy_from_env({"MINIPCAI_NO_CONFIRM": "1"})
        assert policy.confirm_intents == frozenset()

    def test_extra_roots(self):
        policy = policy_from_env({"MINIPCAI_EXTRA_TRUSTED_ROOTS": r"C:\tools; C:\opt"})
        assert policy.is_trusted_app_path(r"C:\tools\a.exe")
        assert policy.is_trusted_app_path(r"C:\opt\b.exe")

    def test_unknown_function_default_policy_is_strict(self):
        policy = default_policy()
        assert policy.enforce_trusted_app_roots is True
        assert policy.allowed_file_extensions is None


class TestLooksLikePath:
    @pytest.mark.parametrize(
        "text",
        [
            r"öffne C:\Users\hacker\secrets.txt",
            "öffne /etc/passwd",
            "finde ../../etc/passwd",
            r"öffne \\server\share\x.txt",
            r"starte %SystemRoot%\System32\cmd.exe",
        ],
    )
    def test_paths_are_detected(self, text):
        assert looks_like_path(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "öffne notepad",
            "was ist 1/2",
            "rechne 3/4 aus",
            "öffne die downloads",
            "suche im internet nach 2/3",
            "",
        ],
    )
    def test_plain_requests_are_not_paths(self, text):
        assert looks_like_path(text) is False
