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
   `subprocess.Popen([exe], shell=False, cwd=<app folder>)`; files/folders via
   `os.startfile`; URLs via `webbrowser`. A test statically scans the source for
   forbidden patterns.
4. **Blocked executables**: shells, script hosts, registry tooling and known
   LOLBins (`cmd.exe`, `powershell.exe`, `wscript.exe`, `msdt.exe`, `fodhelper.exe`,
   `rundll32.exe`, `wsl.exe`, ...) are rejected by the security validator even if
   someone adds them to the registry by accident.
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
minipcai audit                 # show the most recent audit log entries
minipcai audit --limit 50 --json
minipcai --version
```

In the UI you can switch between the dry-run and the Windows executor via the
*Executor* menu (on a non-Windows machine the switch explains that OS actions will
be unavailable), clear the chat with *Chat → Clear chat* (Ctrl+L), and see the audit
log location in the status bar. Timer results appear in the chat when the timer
elapses; if the audit log is not writable, the UI says so at startup because accepted
requests are then refused.

## Training, evaluation and artifacts

* Dataset: `data/intent_dataset.v2.jsonl` — 1470 versioned German examples in four
  categories: `normal`, `typo`, `unknown` (unsupported/out-of-domain) and `ambiguous`
  (must trigger a clarification, trained towards the `unknown` label). The dataset is
  reproducible: `python scripts/generate_dataset.py` (deterministic seed). The version
  is part of the file name; `minipcai` automatically uses the highest
  `intent_dataset.v*.jsonl` it finds in `data/`, so publishing a new dataset version
  needs no code change.
* Model: TF-IDF word (1–2 grams) + character (2–4 grams) features with multinomial
  `LogisticRegression` (`C=10`) — trains in a few seconds on CPU. Both vectorizers
  share the `normalize_for_model` preprocessor, so casing/punctuation variants map to
  the same features while arithmetic operators survive for calculator requests.
* Thresholds (`min_confidence`, `min_margin`) are calibrated on the validation split.
  Grid pairs that meet the preferred safety bounds (no wrongly accepted in-scope
  request, at most one unknown/ambiguous accept) are preferred, taking the pair with
  the most accepted-and-correct decisions. When no grid pair meets those bounds —
  which is the case for the v2 validation split, where the lowest reachable number of
  accepted unknown/ambiguous examples is 4 — selection falls back to the penalty
  score, which heavily weights wrong and out-of-scope accepts. The chosen row,
  including its `accepted_unknown` count, is recorded in `models/metrics.json`
  (`threshold_calibration`); for the committed artifacts the chosen thresholds are
  (0.5, 0.3) with 4 unknown/ambiguous accepts out of 220 validation examples.
* Current results (seed 42, 70/15/15 split): validation accuracy 0.941 /
  macro-F1 0.951, test accuracy 0.923 / macro-F1 0.940. The safety-relevant number is
  `accepted_wrong_in_scope`: **0** — no request with a valid label was accepted with
  the wrong intent on either split. The few out-of-scope requests that pass the gates
  are either refused during target resolution or answered with read-only information.
* Held-out generalization is checked by
  `tests/test_pipeline.py::TestGeneralizationToNewPhrasings`: 20 German in-scope
  phrasings that do not appear verbatim in the dataset (enforced by a test) plus 2
  out-of-scope requests; all 22 pass
  (`pytest tests/test_pipeline.py -k Generalization`). The set is a smoke test, not a
  statistical estimate: it is small, and two of its in-scope cases are within edit
  distance 1 of a dataset entry.
* Known evaluation caveat: the split is stratified by label, not by near-duplicate
  cluster, so about 30% of the test examples (66/221) are within Damerau-Levenshtein
  distance 2 of a training example (mostly generated typo variants of the same normal
  example). The headline test accuracy is therefore optimistic; the safety metrics are
  reported separately per split.
* Committed artifacts: `models/metadata.json` (model, dataset hash, thresholds,
  environment, safety summary) and `models/metrics.json` (accuracy, macro-F1,
  per-class P/R/F1, confusion matrix, per-category gate decisions, safety summary).
  The binary `model.joblib` is not committed — run `minipcai-train` after cloning.
  It is a pickle produced by joblib: only load model artifacts you trained yourself.

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
  train.py                training, calibration, evaluation (incl. safety summary)
  cli.py                  ask / chat / train / registry / audit / ui
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
  resolution or answered with read-only information (system status, calculation).
  A handful of near-miss requests ("bearbeite die notizen") can still be classified
  as an open action; because the assistant can only *open* registered targets, the
  worst case is opening the wrong registered item, never modifying or deleting one.
* The calculator supports basic arithmetic only (no functions, no percent-of);
  results with more than 1000 digits are refused.
* `close_app` terminates without confirmation and only matches the registered
  executable path; it cannot close apps that were started outside their registered
  location.
* Audit log has no rotation; the UI tests need a Qt platform plugin and are skipped
  on machines without one.
* The classifier is a bag-of-words model: it generalizes to unseen German phrasings
  but has no understanding of negation or long-range context.
* No auto-update of the registry; it is maintained by hand on purpose.

## License

MIT
