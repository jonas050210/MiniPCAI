# Security policy

MiniPCAI can open and close programs, open files and folders, open web pages
and run web searches on a user's machine. That makes it a small but real
attack surface, so the security model is documented here **before** the
feature list: every guarantee below is enforced by code and by tests, not by
convention.

## Reporting a vulnerability

Please open a **private** security advisory on GitHub
(`Security` → `Report a vulnerability`) instead of a public issue, or contact
the maintainers directly if you cannot use advisories. Include the version,
your platform, the exact request text and, if possible, the audit log excerpt
(`minipcai audit --limit 20 --json`).

Please do not include real secrets; MiniPCAI does not store request texts when
privacy mode is on (`store_text_in_audit = false`).

## Security model

### 1. The registry is the only source of targets

The intent model maps a sentence to an *intent*, never to a path. A target is
resolved by matching the sentence against the aliases of the registry
(`minipcai/data/registry.json` or the per-user copy in
`%LOCALAPPDATA%\MiniPCAI\registry.json`). Consequences:

* A request that spells out a path (`öffne C:\Users\me\secret.txt`,
  `finde ../../etc/passwd`, `/etc/passwd`) is refused *before* it reaches the
  model — no path is ever assembled from user text.
* `find_file` searches only inside folders the user marked `"searchable"`.
* A registry entry is only usable if it survives validation *and* the runtime
  validator (`SecurityValidator`) proves the plan's entry is an unmodified copy
  of a validated registry entry (provenance check).

### 2. No shell, no command strings

Requests are matched against intents and resolved through the registry; nothing
is ever handed to a shell (`subprocess` runs with an argument *list*, never
`shell=True`, and there is no `eval`/`exec` in the code base - a static
source-scan test enforces both). On top of that, a request containing shell
syntax (`;`, `` ` ``, `|`, `&&`, `||`, `$(`, `${}`, newlines) is refused as
`unsafe_request` before the classifier runs, so "öffne notepad; rm -rf /" is
not interpreted as a valid action.

### 3. Blocklists that survive a hand-edited registry

Even if somebody edits the registry by hand, these are refused at load time and
again at execution time:

* interpreters and shells (`cmd.exe`, `powershell.exe`, `wscript.exe`,
  `python.exe`, `node.exe`, `wsl.exe`, `explorer.exe`, …),
* file types that execute code (`.exe`, `.lnk`, `.bat`, `.ps1`, `.vbs`, `.js`,
  `.msi`, `.jar`, `.dll`, `…`) — an `open_file` target must be a document type,
* executable-less names (the file type must be verifiable),
* relative paths, `..` traversal, non-`http(s)` URLs, non-`.exe` applications.

### 4. Trusted application locations

Applications must live under a trusted root (`%ProgramFiles%`,
`%ProgramFiles(x86)%`, `%SystemRoot%\System32`, `%SystemRoot%`, `%ProgramData%`,
`%LOCALAPPDATA%\Programs`, `%LOCALAPPDATA%\Microsoft\WindowsApps`) unless the
deployment explicitly opts out with `MINIPCAI_ALLOW_UNTRUSTED_APPS=1` (or
`MINIPCAI_EXTRA_TRUSTED_ROOTS`, or `--allow-untrusted`). The escape hatches are
honoured consistently: wherever a component has to pick a policy on its own
(``Registry.load``, the request guard, ``Settings.to_policy``) it uses the
environment-aware policy, so a path approved in the configuration is approved
everywhere. On machines where a placeholder cannot be expanded (a Linux CI box
validating a Windows registry) well-documented literal locations are used
instead of skipping the check.

Optional integrity pinning: `"sha256": "…"` on an app entry, or an
`executable_hashes` map in the policy, pins the binary that is allowed to run.

### 5. Confirmation before state-changing actions

Closing an application can discard unsaved work and a web search leaves the
machine, so both ask first (`confirmation_intents`, default
`close_app, web_search`). Confirmations

* are single use, bound to a request id,
* expire after `confirmation_timeout` seconds (default 120),
* are recorded in the audit log (`required` / `approved` / `declined`),
* can be skipped for scripts with `--yes` or `MINIPCAI_NO_CONFIRM=1` — never by
  accident, never partially.

### 6. Fail-closed audit log

Every accepted request is written to the audit log *before* it is executed. If
that write fails, the action is not executed and the request is rejected with
`audit_unavailable`. The log is append-only JSONL, rotates at a size limit and
carries a SHA-256 hash chain (`prev`/`hash`) so deleting or editing a record is
detectable with `minipcai audit --verify`. Privacy mode stores a hash and the
length of the request instead of its text.

### 7. Executors

`dry-run` (the default) performs no side effects and only records plans.
`windows` is the real executor. It uses `subprocess.Popen([exe], shell=False)`
for applications and `os.startfile()` for documents/folders/URLs. There is no
`shell=True`, no `eval`, no `exec`, no string-built command line anywhere in the
package — a static test scans the sources for these patterns, and a fuzz-style
test asserts that no user text ever appears in a plan.

## Threat model

In scope: a malicious or careless *request text* (prompt injection, path
injection, command injection, unicode tricks), a hand-edited or hostile
registry file, a tampered audit log, a model artifact that does not match the
dataset it claims.

Out of scope: an attacker who can already write to the MiniPCAI installation or
run code as the user (they do not need MiniPCAI), and a compromised dependency
(as with any Python package — pin your environment).

## What MiniPCAI deliberately does not do

* It never deletes, moves or renames files.
* It never types into other applications, clicks or automates the desktop.
* It never sends anything except an explicitly requested web search to the
  internet. There is no telemetry, no update check, no cloud model.
* It never executes a target that is not in the registry — including
  "helpful" guesses such as opening a file that looks like a path.

## Hardening checklist for a deployment

```text
minipcai setup --private-audit      # no request texts on disk
minipcai doctor --check-files       # registry/model/audit/policy self-check
minipcai registry --check-files     # which entries are unusable here
minipcai audit --verify             # hash chain intact?
```

Keep `executor_mode = "dry-run"` (the default) until the registry fits your
machine, then switch with `--executor windows`.
