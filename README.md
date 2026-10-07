# MiniPCAI

A small, self-trained assistant for **safe, everyday Windows PC actions**, controlled by
**German** natural-language requests. MiniPCAI is deliberately minimal: no LLM, no cloud,
no shell — just a lightweight local intent classifier, a hand-curated target registry and
a strictly validated execution path.

```
German request
   │
   ▼
Path guard                            ── a request that spells out a path is refused here
   │
   ▼
Intent classifier (TF-IDF + LogisticRegression, CPU-friendly)
   │
   ▼
Confidence / ambiguity gates          ── rejects unclear or out-of-scope requests
   │
   ▼
Clarification / confirmation          ── "Meintest du 'notizen'?" / "bestätige schließen"
   │
   ▼
Logical target resolution             ── aliases matched against the registry ONLY;
   │                                     paths/URLs never come from the model
   ▼
Security validation                   ── defense-in-depth re-check of every plan
   │
   ▼
Fail-closed audit checkpoint          ── no audit log ⇒ no execution
   │
   ▼
Executor (dry-run by default, or Windows)
```

## Supported actions

| Intent        | German example                        | What it does                                   |
|---------------|---------------------------------------|------------------------------------------------|
| `open_app`    | "öffne notepad"                       | Launches a registered application              |
| `close_app`   | "beende firefox"                      | Terminates a registered application (asks first)|
| `open_url`    | "öffne wikipedia"                     | Opens a registered website                     |
| `open_file`   | "öffne die notizen"                   | Opens a registered file                        |
| `open_folder` | "öffne die downloads"                 | Opens a registered folder                      |
| `find_file`   | "finde die datei rechnung"            | Searches inside searchable registered folders  |
| `web_search`  | "suche im internet nach katzen"       | Opens a search on a registered search provider |
| `sys_cpu`     | "wie hoch ist die cpu auslastung"     | Reports CPU usage                              |
| `sys_ram`     | "wie viel arbeitsspeicher ist frei"   | Reports RAM usage                              |
| `sys_disk`    | "wie voll ist die festplatte"         | Reports disk usage                             |
| `sys_summary` | "wie läuft der pc"                    | Combined CPU/RAM/disk report                   |
| `calc`        | "was ist 12*4"                        | Evaluates a basic arithmetic expression        |
| `timer`       | "stelle einen timer auf 10 minuten"   | Starts a countdown timer (capped, cancellable)  |
| `unknown`     | "wie wird das wetter morgen"          | Explained refusal                              |

Replies are **German by default** (`--language en`, `MINIPCAI_LANGUAGE=en` or
`language = "en"` in `config.toml` switch to English).

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate                # Windows
pip install -e ".[dev,gui]"           # omit [gui] to skip the desktop UI

minipcai setup                        # per-user registry + config (~/.minipcai or %LOCALAPPDATA%\MiniPCAI)
minipcai train                        # trains models/model.joblib from the packaged dataset
minipcai doctor                       # self-check: registry, model, audit, policy, GUI
minipcai ask "öffne notepad"          # dry-run: describes what it would do
minipcai ask "schließe firefox" --yes # confirmation is required, --yes is the script escape hatch
minipcai ask "öffne notepad" --executor windows   # real execution (Windows only)
minipcai chat                         # interactive REPL
minipcai ui                           # desktop UI (PySide6, optional extra)
```

Day-to-day commands:

```bash
minipcai registry --check-files         # which entries are usable on this machine
minipcai registry --add-app paint="C:\Program Files\Paint\paint.exe" --alias malprogramm
minipcai registry --add-searcher ddg="https://duckduckgo.com/?q={query}" --alias suche
minipcai registry --remove paint
minipcai registry --json                # machine-readable
minipcai audit --limit 50 --json        # what happened
minipcai audit --verify                 # hash chain intact?
minipcai --version
```

## Quality gates

Beyond the unit tests, MiniPCAI ships a *golden hard set* of realistic German
sentences (typos, umlaut variants, colloquial phrasings, out-of-scope requests,
injection attempts) plus a registry-driven sweep that derives hundreds of
phrasings from the registry itself:

```bash
python scripts/check_golden_hard.py --sweep
```

The gate fails if anything out of scope or wrongly targeted is ever executed
(**0 wrong accepts** - a safety invariant) or if more than **8 %** of in-scope
requests are rejected that the assistant should have understood. Typos are
allowed to ask back ("Meintest du 'malprogramm'?") - that counts as handled,
not as a rejection. The same numbers are asserted in `tests/test_golden.py`,
so a regression shows up in the normal test run.

## Security model

The full model (threat model, reporting process, hardening checklist) lives in
[`SECURITY.md`](SECURITY.md). The short version:

1. **The registry is the only source of targets.** The classifier decides *what kind* of
   action is wanted; every executable, file, folder, URL and search provider comes from
   `minipcai/data/registry.json` (or the per-user copy). Path-like requests
   (`öffne C:\...`, `finde ../../etc/passwd`) are refused before the model even runs.
2. **Blocklists survive a hand-edited registry.** Shells and interpreters
   (`cmd.exe`, `powershell.exe`, `python.exe`, `node.exe`, `wsl.exe`, …) and file types
   that execute code (`.exe`, `.lnk`, `.bat`, `.ps1`, `.vbs`, `.msi`, `.jar`, …) are
   refused at load time *and* again at execution time. `open_file` targets must be
   document types with a verifiable extension.
3. **Applications come from trusted locations** (`%ProgramFiles%`, `%SystemRoot%\System32`,
   `%LOCALAPPDATA%\Programs`, …) unless the deployment opts out explicitly. Entries may
   pin a SHA-256 hash.
4. **State-changing actions ask first.** `close_app` and `web_search` require an
   explicit confirmation that is single-use, expires after 120 s and is audited.
5. **Fail-closed audit log.** Accepted requests are written before execution; a broken
   audit log means no execution. JSONL + rotation + SHA-256 hash chain
   (`minipcai audit --verify`) + optional privacy mode (hash instead of text).
6. **No arbitrary execution.** No `shell=True`, no `eval`/`exec`, no string-built command
   lines; applications start with `subprocess.Popen([exe], shell=False)`, files/folders via
   `os.startfile`, URLs via the browser. A static test scans the sources for the
   forbidden patterns.
7. **Rejection over guessing.** Unknown, low-confidence, ambiguous or unmatched requests
   are rejected with a readable explanation; ambiguous targets are resolved through a
   clarification question instead of a guess.

## Desktop UI

The desktop window is a thin shell around the same audited pipeline the CLI uses:

```bash
pip install "minipcai[gui]"     # PySide6 is an optional extra
minipcai ui                     # or: minipcai gui
```

* **Never blocks.** Requests and confirmations run in a worker thread; the send
  button turns into *Stop* while a request is in flight and cancels it.
* **Plan cards.** Every accepted request shows intent, target, parameters,
  executor and an explicit *dry-run* badge, so it is obvious what would happen.
* **Confirmations and clarifications are dialogs.** Closing an application or
  starting a web search asks first; ambiguous targets ("öffne notizen") offer a
  pick list, and the picked answer is re-sent for you.
* **Sidebar** with three tabs: running timers (cancel one or all), the registry
  editor (add/remove entries - validated with the same checks as the runtime,
  so a blocked executable cannot sneak in) and the audit viewer (last 100
  events, plus *verify hash chain*).
* **Settings dialog** (Ctrl+,): executor mode, reply language, theme, privacy
  option `store_text_in_audit` and the global hotkey.
* **History and autocomplete** over previous requests and registry aliases, a
  system-tray icon, dark mode (`Ansicht → Design`) and a global hotkey
  (default `Ctrl+Alt+M`, window shortcut outside Windows).

Everything is localized: the UI is **German by default** and follows
`Settings.language`, so `minipcai ui --language en` (or `MINIPCAI_LANGUAGE=en`)
switches the whole window to English. When PySide6 is missing, `minipcai ui`
prints how to install it instead of raising an `ImportError`.

## Installation layouts

| | Development (checkout) | Installed (wheel) |
|---|---|---|
| Registry | `minipcai/data/registry.json` (in the package) | packaged copy, seeded to `%LOCALAPPDATA%\MiniPCAI\registry.json` on first run |
| Dataset | packaged `minipcai/data/intent_dataset.v*.jsonl` | packaged (shipped in the wheel) |
| Models | `models/` in the checkout | `~/.minipcai/models` |
| Audit log | `~/.minipcai/audit.jsonl` | `%LOCALAPPDATA%\MiniPCAI\audit.jsonl` |
| Config | `~/.minipcai/config.toml` | same, created by `minipcai setup` |

Settings precedence: **command line > environment (`MINIPCAI_*`) > `config.toml` > defaults**.
Documented environment variables: `MINIPCAI_REGISTRY`, `MINIPCAI_MODEL`, `MINIPCAI_MODELS_DIR`,
`MINIPCAI_AUDIT`, `MINIPCAI_LANGUAGE`, `MINIPCAI_EXECUTOR`, `MINIPCAI_NO_CONFIRM`,
`MINIPCAI_PRIVACY_AUDIT`, `MINIPCAI_ALLOW_UNTRUSTED_APPS`, `MINIPCAI_EXTRA_TRUSTED_ROOTS`.

## Training, evaluation and artifacts

* **Dataset**: `minipcai/data/intent_dataset.v3.jsonl` — 2053 versioned German examples
  across four categories: `normal`, `typo`, `unknown` (unsupported/out-of-domain) and
  `ambiguous` (trained towards `unknown`, resolved at runtime by a clarification question).
  It is generated from the registry: `python scripts/generate_dataset.py` synthesises
  templates from the registered aliases (deterministic seed) and *drops* every example
  whose target the registry does not contain (140 dropped for v3). The version is part of
  the file name and the loader always picks the highest `intent_dataset.v*.jsonl` it ships.
* **Model**: TF-IDF word (1–2 grams) + character (2–4 grams) features with multinomial
  `LogisticRegression` (`C=10`); trains in a few seconds on CPU. `models/metadata.json`
  records the dataset version, example count and **SHA-256** of the training file, and the
  loader verifies that fingerprint — a model trained on another dataset is refused.
* **Calibration**: `minipcai train` searches a grid of `(min_confidence, min_margin)`
  pairs on the validation split. The hard rule is *no wrongly accepted labelled request*;
  within a 2 % budget for accepted out-of-scope examples it maximises accepted-correct
  decisions and ties towards the stricter gates. The chosen row, the full table and the
  budget are recorded in `models/metrics.json`.
* **Current results** (seed 42, committed artifacts): validation accuracy **0.9675** /
  macro-F1 **0.9723**, test accuracy **0.9870** / macro-F1 **0.9891**,
  `accepted_wrong_in_scope = 0` (the safety-relevant number), 2 out-of-scope accepts out
  of 308 test examples, chosen thresholds (0.4, 0.1). A near-duplicate-disjoint grouped
  holdout (`minipcai train --grouped-eval`) reports accuracy 0.9615 — the dataset's
  typo variants no longer inflate the headline number.
* **Registry agreement**: training reports how many dataset targets resolve against the
  registry (`692/692` for v3 — every targeted example is executable).
* **Held-out phrasings**: `tests/test_pipeline.py::TestGeneralizationToNewPhrasings`
  exercises 20 in-scope phrasings that do not appear verbatim in the dataset plus
  out-of-scope requests, and a 46-case realistic probe set covers spoken-style phrasing
  ("hätte gern firefox offen") in `tests/test_user_acceptance.py`.
* **Artifacts in git**: `models/metadata.json`, `models/metrics.json` (+ the packaged
  dataset). The binary `model.joblib` is not committed — run `minipcai train`. It is a
  joblib pickle: only load model artifacts you trained yourself.

## The registry

```json
{
  "version": 1,
  "apps":      [{"id": "notepad", "aliases": ["notepad", "editor"],
                 "executable": "C:\\Windows\\System32\\notepad.exe"}],
  "files":     [{"id": "notes", "aliases": ["notizen"],
                 "path": "%USERPROFILE%\\Documents\\notes.txt"}],
  "folders":   [{"id": "downloads", "aliases": ["downloads"],
                 "path": "%USERPROFILE%\\Downloads", "searchable": true}],
  "websites":  [{"id": "wikipedia", "aliases": ["wikipedia", "wiki"],
                 "url": "https://www.wikipedia.org"}],
  "searchers": [{"id": "google_search", "aliases": ["google", "im internet"],
                 "url_template": "https://www.google.com/search?q={query}"}]
}
```

Ids are unique across the registry; aliases are unique **within a section** (so `google`
may be both a website and a search provider — the *intent* decides which one is meant).
`%ENVVAR%` placeholders are expanded at load time. `searchable` marks folders that
`find_file` may scan; `sha256` optionally pins an application binary. `minipcai registry`
validates and edits the file with exactly the same rules as the runtime.

## Project layout

```
minipcai/
  data/                   registry.json + intent_dataset.v3.jsonl (shipped in the wheel)
  intents.py              label definitions and intent → section mapping
  config.py               paths, limits, thresholds, Config
  paths.py                packaged vs. per-user layout, dataset discovery
  policy.py               SecurityPolicy: blocklists, trusted roots, confirmations
  textutils.py            normalization, German math/duration/query parsing
  dataset.py              dataset loading + validation
  registry.py             registry loading, validation, alias matching
  model.py                IntentModel protocol + sklearn implementation (+ fingerprint)
  calc_engine.py          safe AST-whitelist arithmetic
  targets.py              logical target resolution → ActionPlan
  security.py             defense-in-depth plan validation (provenance, blocked, pinned)
  actions.py              DryRunExecutor + WindowsExecutor (+ timers)
  audit.py                append-only JSONL audit log, rotation, hash chain, privacy mode
  pipeline.py             the Assistant orchestration (gates, confirmations, i18n)
  service.py              sessions: clarifications, confirmations, async, timer view
  settings.py             TOML/env/CLI settings with validation
  i18n.py                 German/English catalogue for every user-facing message
  doctor.py               installation self-check
  registry_tools.py       `minipcai registry` implementation
  evaluate.py             grouped (near-duplicate-disjoint) evaluation, resolvability
  train.py                training, calibration, evaluation
  cli.py                  ask / chat / train / registry / doctor / audit / setup / ui
  ui/app.py               PySide6 desktop UI
scripts/
  generate_dataset.py     registry-driven dataset generator
  check_dataset_fingerprint.py   CI: dataset hash vs. model metadata
  check_wheel_contents.py        CI: the artifacts ship registry + dataset
  smoke_installed.py             CI: install the wheel in a clean venv and use it
tests/                    pytest suite (incl. unsafe-input, source-scan, packaging)
```

### Swapping the model

The pipeline depends only on the small `IntentModel` interface (`labels` +
`predict(text) -> Prediction`). Implement it with any backend and pass it to `Assistant`;
nothing else needs to change. The metadata contract (`labels`, `thresholds`, `dataset`
fingerprint) is what keeps an unknown model from silently weakening the gates.

## Development

```bash
pytest -q                                   # full suite (Qt tests skip without a display)
ruff check .                                # lint (CI runs this)
minipcai train --grouped-eval               # retrain + near-duplicate-disjoint evaluation
python scripts/check_dataset_fingerprint.py # dataset/model consistency
python -m build                             # wheel + sdist
python scripts/check_wheel_contents.py      # the artifacts must ship the data
python scripts/smoke_installed.py           # install the wheel in a clean venv and use it
```

CI (`.github/workflows/ci.yml`) runs ruff, the test suite on Linux and Windows
(Python 3.10 + 3.12, Qt tests under `xvfb` on Linux), the dataset fingerprint check and
the packaging smoke test.

## Known limitations

* Only the listed intents; compound requests ("öffne X und Y") are rejected as ambiguous.
* Target recognition is alias-based: typos are rejected with a "did you mean" question
  rather than fuzzy-executed; the clarification answer resolves to a registry id.
* Ambiguity detection relies on calibrated confidence/margin gates; out-of-scope phrases
  that pass the gates are still refused during target resolution or answered read-only.
* The calculator supports basic arithmetic only (no functions); results beyond 1000 digits
  are refused.
* `close_app` matches the registered executable path only; it cannot close applications
  that were started from a different location, and it asks for confirmation first.
* The classifier is a bag-of-words model: it generalizes to unseen German phrasings but has
  no understanding of negation or long-range context.
* The registry is maintained by hand on purpose; there is no auto-discovery of installed
  software.
* The desktop UI needs the `[gui]` extra and a Qt platform plugin (headless CI uses
  `xvfb`); the CLI is fully functional without it.

## License

MIT — see [`LICENSE`](LICENSE). Security policy: [`SECURITY.md`](SECURITY.md).
