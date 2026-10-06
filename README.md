# MiniPCAI

A small, self-trained assistant for **safe, everyday Windows PC actions**, controlled by
**German** natural-language requests. MiniPCAI is deliberately minimal: no LLM, no cloud,
no shell — just a lightweight local intent classifier, a hand-curated target registry and
a strictly validated execution path.

```
User text (German)
   │
   ▼
Intent classifier (trained TF-IDF + LogisticRegression, CPU-friendly)
   │
   ▼
Confidence / ambiguity gates          ── rejects unclear or out-of-scope requests
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
Executor (dry-run or Windows)
   │
   ▼
Audit log (JSONL) + English reply
```

## Supported actions (MVP)

| Intent        | German example                        | What it does                                   |
|---------------|---------------------------------------|------------------------------------------------|
| `open_app`    | "öffne notepad"                       | Launches a registered application              |
| `close_app`   | "beende firefox"                      | Terminates a registered application            |
| `open_url`    | "öffne wikipedia"                     | Opens a registered website                     |
| `open_file`   | "öffne die notizen"                   | Opens a registered file                        |
| `open_folder` | "öffne die downloads"                 | Opens a registered folder                      |
| `find_file`   | "finde die datei rechnung"            | Searches inside searchable registered folders  |
| `sys_cpu`     | "wie hoch ist die cpu auslastung"     | Reports CPU usage                              |
| `sys_ram`     | "wie viel arbeitsspeicher ist frei"   | Reports RAM usage                              |
| `sys_disk`    | "wie voll ist die festplatte"         | Reports disk usage                             |
| `sys_summary` | "wie läuft der pc"                    | Combined CPU/RAM/disk report                   |
| `calc`        | "was ist 12*4"                        | Evaluates a basic arithmetic expression        |
| `timer`       | "stelle einen timer auf 10 minuten"   | Starts a countdown timer                       |

Everything the assistant replies is in **English**; it **understands** German.

## Security model

1. **The AI never invents paths.** Intent classification only decides *what kind* of
   action is wanted. Every executable, file, folder and URL comes exclusively from
   `data/registry.json`, a small file you curate and review by hand.
2. **Strict registry validation at load time**: unique ids, globally unique aliases,
   absolute paths only (no `..`), `http(s)` URLs only, `searchable` only on folders.
3. **No arbitrary execution**: no `CMD`, no `PowerShell`, no `shell=True`, no
   `os.system`, no `eval`/`exec`, no string commands. Applications are launched with
   `subprocess.Popen([exe], shell=False)`; files/folders via `os.startfile`;
   URLs via `webbrowser`. A test statically scans the source for forbidden patterns.
4. **Blocked executables**: shells and script hosts (`cmd.exe`, `powershell.exe`,
   `wscript.exe`, ...) are rejected by the security validator even if someone adds
   them to the registry by accident.
5. **Process termination by full path match only**: `close_app` only terminates
   processes whose executable path is exactly the registered one.
6. **Fail-closed auditing**: an accepted request is written to the append-only audit
   log *before* execution; if that write fails, the action is not executed.
7. **Rejection over guessing**: unknown, low-confidence, ambiguous, unmatched or
   invalid requests are always rejected with an explanatory English message.
   The safe arithmetic evaluator uses an AST whitelist with hard limits
   (no names/calls, bounded literals and exponents), so calculator input cannot
   become code.

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e ".[dev]"       # use pip install -e . if you don't need tests/lint
```

## Usage

```bash
# 1. Train the intent model (creates models/model.joblib, metadata.json, metrics.json)
minipcai-train

# 2. Try it (dry-run by default - describes actions without performing them)
minipcai ask "öffne notepad"
minipcai ask "was ist 12*4"
minipcai ask "öffne die downloads" --json

# 3. Interactive chat
minipcai chat

# 4. Desktop UI (PySide6)
minipcai ui

# 5. Real execution on Windows (opt-in)
minipcai ask "öffne notepad" --executor windows

# Housekeeping
minipcai registry              # validate and list the registry
minipcai --version
```

In the UI you can switch between the dry-run and the Windows executor via the
*Executor* menu. Timer results appear in the chat when the timer elapses.

## Training, evaluation and artifacts

* Dataset: `data/intent_dataset.v1.jsonl` — 841 versioned German examples in four
  categories: `normal`, `typo`, `unknown` (unsupported/out-of-domain) and `ambiguous`
  (must trigger a clarification, trained towards the `unknown` label).
  The dataset is reproducible: `python scripts/generate_dataset.py` (deterministic seed).
* Model: TF-IDF word (1–2 grams) + character (2–4 grams) features with multinomial
  `LogisticRegression` — trains in a few seconds on CPU.
* Thresholds (`min_confidence`, `min_margin`) are calibrated on the validation split
  under safety constraints (no wrongly accepted request, at most one unknown accept).
* Committed artifacts: `models/metadata.json` (model, dataset hash, thresholds,
  environment) and `models/metrics.json` (accuracy, macro-F1, per-class P/R/F1,
  confusion matrix, per-category gate decisions). The binary `model.joblib` is not
  committed — run `minipcai-train` after cloning.

## The registry (`data/registry.json`)

```json
{
  "version": 1,
  "apps":     [{"id": "notepad", "aliases": ["notepad", "editor"],
                "executable": "C:\\Windows\\System32\\notepad.exe"}],
  "files":    [{"id": "notes", "aliases": ["notizen"],
                "path": "%USERPROFILE%\\Documents\\notes.txt"}],
  "folders":  [{"id": "downloads", "aliases": ["downloads"],
                "path": "%USERPROFILE%\\Downloads", "searchable": true}],
  "websites": [{"id": "wikipedia", "aliases": ["wikipedia", "wiki"],
                "url": "https://www.wikipedia.org"}]
}
```

Aliases must be unique across the whole registry. `%ENVVAR%` placeholders are expanded
at load time. Mark folders as `searchable` to include them in `find_file`.

## Project layout

```
data/                     dataset + registry (versioned, in git)
models/                   metadata.json + metrics.json in git; model.joblib is local
minipcai/
  intents.py              label definitions
  config.py               paths, limits, thresholds
  textutils.py            normalization, German math/duration/search-term parsing
  dataset.py              dataset loading + validation
  registry.py             registry loading, validation, alias matching
  model.py                IntentModel protocol + sklearn implementation
  calc_engine.py          safe AST-whitelist arithmetic
  targets.py              logical target resolution -> ActionPlan
  security.py             defense-in-depth plan validation
  actions.py              DryRunExecutor + WindowsExecutor
  audit.py                append-only JSONL audit log
  pipeline.py             the Assistant orchestration
  train.py                training, calibration, evaluation
  cli.py                  ask / chat / train / registry / ui
  ui/app.py               PySide6 chat window
scripts/generate_dataset.py   reproducible dataset generator
tests/                    pytest suite (incl. unsafe-input and source-scan tests)
```

### Swapping the model

The pipeline depends only on the tiny `IntentModel` interface (`labels` + `predict(text)
-> Prediction`). Implement it with any backend and pass it to `Assistant`; nothing else
needs to change.

## Development

```bash
pytest                 # full test suite (UI tests skip when no Qt platform is available)
ruff check .           # lint
minipcai-train         # retrain after dataset changes
```

## Known limitations (MVP)

* Only the 12 listed intents; no compound requests ("open X and Y" is rejected as ambiguous).
* Target recognition is alias-based: typos in target names are rejected (with a
  "did you mean" hint) rather than fuzzy-executed.
* Ambiguity detection relies on calibrated confidence/margin gates; a few rare
  target-less phrases pass the gates and are then safely rejected during target
  resolution.
* The calculator supports basic arithmetic only (no functions, no percent-of).
* `close_app` terminates without confirmation and only matches the registered
  executable path; it cannot close apps that were started outside their registered
  location.
* Audit log has no rotation.
* No auto-update of the registry; it is maintained by hand on purpose.

## License

MIT
