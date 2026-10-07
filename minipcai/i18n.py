"""Localization of everything the user reads.

MiniPCAI *understands* German and can *answer* in German or English. The
default is German because MiniPCAI is built for German-speaking users;
``--language en`` (or ``language = "en"`` in the configuration) switches all
user-facing text to English.

Design: message catalogs are plain dictionaries keyed by a stable identifier.
Result objects (``AssistantResult``, ``ActionResult``, errors) always carry the
English text *and* a key plus fields, so a frontend can render the same event
in any supported language without re-classifying anything.
"""

from __future__ import annotations

from typing import Any

# MiniPCAI speaks German by default; English is one flag away
# (--language en, MINIPCAI_LANGUAGE=en or language = "en" in config.toml).
DEFAULT_LANGUAGE = "de"
SUPPORTED_LANGUAGES: tuple[str, ...] = ("en", "de")

_DE = {
    # -- input validation --------------------------------------------------
    "invalid.empty": "Bitte gib eine Anfrage ein.",
    "invalid.too_long": "Die Anfrage ist zu lang (maximal {limit} Zeichen).",
    "invalid.not_text": "Bitte gib eine Anfrage als Text ein.",
    # -- classification ----------------------------------------------------
    "unknown.request": (
        "Das kann ich leider nicht. {capabilities}"
    ),
    "unknown.capabilities": (
        "Ich kann registrierte Programme öffnen und schließen, registrierte Dateien, "
        "Ordner und Webseiten öffnen, Dateien finden, im Web suchen, Systeminfos "
        "anzeigen, rechnen und Timer stellen."
    ),
    "low_confidence": (
        "Ich bin mir bei dieser Anfrage nicht sicher genug ({confidence}). "
        "Kannst du sie eindeutiger formulieren?"
    ),
    "ambiguous_intent": (
        "Diese Anfrage ist mehrdeutig zwischen '{top}' ({top_confidence}) und "
        "'{second}' ({second_confidence}). Bitte sei genauer."
    ),
    "internal.model": "Das Intent-Modell konnte diese Anfrage nicht verarbeiten.",
    "internal.prediction": "Das Intent-Modell hat keine verwertbare Vorhersage geliefert.",
    # -- target resolution --------------------------------------------------
    "target.not_found": (
        "Zu dieser Anfrage passt kein freigegebenes Ziel. Es können nur Ziele aus "
        "der Registry verwendet werden."
    ),
    "target.not_found.suggestion": " Meintest du '{suggestion}'?",
    "target.ambiguous": (
        "Es passen mehrere Einträge ({names}). Bitte nenne genau einen."
    ),
    "target.mismatch": (
        "'{entry}' ist ein registriertes {kind}, aber die Anfrage klingt nach einer "
        "anderen Aktion. Bitte formuliere sie neu und nenne das {kind} ausdrücklich."
    ),
    "target.mismatch.generic": (
        "Das genannte Ziel passt nicht zu dieser Aktion. Bitte formuliere die Anfrage "
        "neu und nenne das Ziel ausdrücklich."
    ),
    # -- parameters ---------------------------------------------------------
    "parameter.calc": "Kein Rechenausdruck gefunden. Beispiel: 'was ist 12*4'.",
    "parameter.timer": "Keine Dauer gefunden. Beispiel: 'stelle einen timer auf 10 minuten'.",
    "parameter.search": "Kein Suchbegriff gefunden. Beispiel: 'finde die datei rechnung'.",
    "parameter.no_searchable_folders": (
        "In der Registry ist kein durchsuchbarer Ordner konfiguriert."
    ),
    "parameter.web_query": "Keine Suchanfrage gefunden. Beispiel: 'suche im internet nach katzen'.",
    "parameter.no_searcher": (
        "In der Registry ist keine Suchmaschine konfiguriert."
    ),
    # -- security -----------------------------------------------------------
    "unsafe_request": "Diese Anfrage wurde aus Sicherheitsgründen abgelehnt.",
    "untrusted_target": (
        "Dieses Ziel liegt nicht an einem freigegebenen Ort, deshalb führe ich es nicht aus."
    ),
    "audit_unavailable": (
        "Das Audit-Log ist gerade nicht beschreibbar, deshalb führe ich aus "
        "Sicherheitsgründen nichts aus."
    ),
    "executor_unavailable": "Diese Aktion ist hier nicht verfügbar: {detail}",
    "execution_error": "Die Aktion ist fehlgeschlagen: {detail}",
    # -- confirmation -------------------------------------------------------
    "confirmation.required": (
        "Ich brauche deine Bestätigung: {action} Antworte mit 'ja' oder 'nein'."
    ),
    "confirmation.declined": "Okay, ich habe nichts gemacht.",
    "confirmation.not_pending": "Es gibt gerade nichts zu bestätigen.",
    "confirmation.yes": "ja",
    "confirmation.no": "nein",
    # -- clarification ------------------------------------------------------
    "clarification.prompt": "Meintest du {options}? Bitte antworte mit der Nummer.",
    "clarification.not_pending": "Es gibt gerade keine offene Rückfrage.",
    "clarification.invalid_choice": "Bitte antworte mit einer Zahl zwischen 1 und {count}.",
    # -- action summaries (structured results) ------------------------------
    "action.open_app.ok": "Programm '{id}' geöffnet.",
    "action.open_app.failed": "'{id}' konnte nicht gestartet werden: {detail}.",
    "action.open_app.missing": "Programm nicht gefunden: {path}.",
    "action.close_app.ok": "{count} Prozess(e) von '{id}' beendet.",
    "action.close_app.none": "Von '{id}' lief kein Prozess.",
    "action.close_app.unresolved": "Der registrierte Pfad konnte nicht aufgelöst werden: {path}.",
    "action.open_url.ok": "Webseite '{id}' im Browser geöffnet.",
    "action.open_url.no_browser": "Es ist kein Browser verfügbar.",
    "action.open_file.ok": "Datei '{id}' geöffnet.",
    "action.open_folder.ok": "Ordner '{id}' geöffnet.",
    "action.open_path.missing": "Pfad nicht gefunden: {path}.",
    "action.open_folder.not_a_folder": "Das Ziel ist kein Ordner: {path}.",
    "action.find_file.results": "{count} Datei(en) zu '{term}' gefunden:\n{listing}",
    "action.find_file.none": "Keine Dateien zu '{term}' in den freigegebenen Ordnern gefunden.",
    "action.find_file.truncated": (
        " (Suche wurde nach {visited} Einträgen bzw. {seconds} s abgebrochen; "
        "das Ergebnis ist möglicherweise unvollständig.)"
    ),
    "action.web_search.ok": "Websuche nach '{query}' gestartet ({url}).",
    "action.web_search.no_browser": "Es ist kein Browser verfügbar.",
    "action.sys_cpu": "CPU-Auslastung: {usage}% ({cores} logische Kerne).",
    "action.sys_ram": (
        "RAM: {used} von {total} belegt ({percent}%), {available} verfügbar."
    ),
    "action.sys_disk": (
        "Datenträger ({root}): {used} von {total} belegt ({percent}%), {free} frei."
    ),
    "action.sys_summary": "Systemübersicht:\n{body}",
    "action.calc": "{expression} = {result}",
    "action.timer.started": "Timer gestartet: {human}.",
    "action.timer.limit": "Es laufen bereits {max} Timer; bitte warte oder beende einen.",
    "action.timer.cancelled": "Timer beendet: {count}.",
    "action.timer.not_found": "Ich habe keinen passenden Timer gefunden.",
    "action.dry_run.prefix": "[Testlauf] ",
    "action.dry_run.open_app": "Würde Programm '{id}' öffnen ({path}).",
    "action.dry_run.close_app": "Würde Prozesse von '{id}' beenden ({path}).",
    "action.dry_run.open_url": "Würde Webseite '{id}' öffnen ({url}).",
    "action.dry_run.open_file": "Würde Datei '{id}' öffnen ({path}).",
    "action.dry_run.open_folder": "Würde Ordner '{id}' öffnen ({path}).",
    "action.dry_run.find_file": "Würde nach '{term}' in {count} freigegebenen Ordner(n) suchen.",
    "action.dry_run.web_search": "Würde im Web nach '{query}' suchen ({url}).",
    "action.dry_run.sys": "Würde Systeminformationen abfragen.",
    "action.dry_run.timer": "Würde einen Timer über {seconds} Sekunden starten.",
    "action.calc.dry_run": "{expression} = {result} (keine Nebenwirkungen).",
    # -- misc ---------------------------------------------------------------
    "timer.elapsed": "Timer abgelaufen ({seconds} s).",
    "search.none_found": "Keine Treffer.",
    # -- desktop UI -----------------------------------------------------------
    "ui.welcome": (
        "Willkommen bei MiniPCAI. Frag auf Deutsch, zum Beispiel 'öffne notepad', "
        "'wie viel ram ist frei' oder 'stelle einen timer auf 5 minuten'. "
        "{registry} | Datensatz v{dataset}."
    ),
    "ui.you": "Du",
    "ui.assistant": "MiniPCAI",
    "ui.send": "Senden",
    "ui.stop": "Stopp",
    "ui.placeholder": "Anfrage auf Deutsch ...",
    "ui.chat.cleared": "Chat geleert.",
    "ui.audit.warning": (
        "Warnung: Das Audit-Log ist nicht beschreibbar ({detail}). Angenommene "
        "Anfragen werden abgelehnt, bis das behoben ist."
    ),
    "ui.plan.title": "Plan",
    "ui.plan.target": "Ziel: {target}",
    "ui.plan.parameters": "Parameter: {parameters}",
    "ui.plan.dry_run": "Dry-Run: es wird nichts ausgeführt",
    "ui.plan.executor": "Executor: {mode}",
    "ui.meta": "{intent} | Konfidenz {confidence} | Grund: {reason}",
    "ui.status.bar": (
        "Modell: {labels} Absichten | Executor: {mode} | {registry} | Audit: {audit}"
    ),
    "ui.executor.switched_windows": (
        "Windows-Executor aktiv: Anfragen führen jetzt echte Aktionen aus."
    ),
    "ui.executor.switched_dry_run": (
        "Dry-Run aktiv: es werden keine echten Aktionen ausgeführt."
    ),
    "ui.executor.non_windows": (
        "Hinweis: Das ist kein Windows-Rechner, daher melden Aktionen, die das "
        "Betriebssystem brauchen, einen Fehler. Informative Aktionen funktionieren weiter."
    ),
    "ui.confirm.title": "Bestätigung nötig",
    "ui.confirm.approved": "Bestätigt - die Aktion wird ausgeführt.",
    "ui.confirm.declined": "Abgebrochen - es wird nichts ausgeführt.",
    "ui.confirm.question": "{action} Wirklich ausführen?",
    "ui.confirm.yes": "Ja",
    "ui.confirm.no": "Nein",
    "ui.menu.file": "&Datei",
    "ui.menu.quit": "Beenden",
    "ui.menu.view": "&Ansicht",
    "ui.menu.theme_system": "Design: System",
    "ui.menu.theme": "Design",
    "ui.executor.dry_run": "Dry-Run (keine echten Aktionen)",
    "ui.executor.windows": "Windows (echte Aktionen)",
    "ui.menu.theme_light": "Design: Hell",
    "ui.menu.theme_dark": "Design: Dunkel",
    "ui.menu.panels": "Seitenleiste",
    "ui.menu.executor": "&Ausführung",
    "ui.menu.chat": "&Chat",
    "ui.menu.clear_chat": "Chat leeren",
    "ui.menu.settings": "Einstellungen",
    "ui.menu.registry_editor": "Registry-Editor",
    "ui.menu.timer_panel": "Timer",
    "ui.menu.audit_viewer": "Audit-Log",
    "ui.menu.help": "&Hilfe",
    "ui.menu.about": "Über MiniPCAI",
    "ui.about.text": (
        "MiniPCAI {version}\n\nEin kleiner, selbst trainierter Assistent, der "
        "deutsche Anfragen versteht und sichere, registry-basierte Aktionen "
        "ausführt.\n\nZiele stammen ausschließlich aus der geprüften Registry; "
        "keine Shell, keine beliebigen Befehle, keine erfundenen Pfade. Jede "
        "Anfrage landet im Audit-Log."
    ),
    "ui.timer.title": "Timer",
    "ui.timer.remaining": "Restzeit",
    "ui.timer.id": "Anfrage",
    "ui.timer.duration": "Dauer",
    "ui.timer.cancel_selected": "Ausgewählten abbrechen",
    "ui.timer.cancel_all": "Alle abbrechen",
    "ui.timer.empty": "Keine laufenden Timer.",
    "ui.timer.cancelled": "{count} Timer abgebrochen.",
    "ui.timer.finished": "Timer abgelaufen ({seconds} Sekunden).",
    "ui.registry.title": "Registry-Editor",
    "ui.registry.reload": "Neu laden",
    "ui.registry.save": "Speichern",
    "ui.registry.add": "Eintrag hinzufügen",
    "ui.registry.remove": "Ausgewählte löschen",
    "ui.registry.path": "Datei: {path}",
    "ui.registry.saved": "Registry gespeichert: {path}",
    "ui.registry.save_failed": "Registry wurde nicht gespeichert: {detail}",
    "ui.registry.add_prompt": "ID=Pfad (bei Webseiten ID=URL):",
    "ui.registry.section_prompt": "Bereich (apps, files, folders, websites, searchers):",
    "ui.registry.confirm_remove": "Eintrag '{id}' wirklich entfernen?",
    "ui.registry.user_hint": "Bearbeitet wird {path}",
    "ui.settings.title": "Einstellungen",
    "ui.settings.executor": "Executor",
    "ui.settings.language": "Sprache",
    "ui.settings.theme": "Design",
    "ui.settings.privacy": "Anfragetexte nicht im Audit-Log speichern",
    "ui.settings.confirmations": "Bestätigung für: {intents}",
    "ui.settings.hotkey": "Globaler Hotkey",
    "ui.settings.saved": "Einstellungen gespeichert: {path}",
    "ui.settings.save_failed": "Einstellungen konnten nicht gespeichert werden: {detail}",
    "ui.audit.title": "Audit-Log",
    "ui.audit.reload": "Neu laden",
    "ui.audit.verify": "Hash-Kette prüfen",
    "ui.audit.empty": "Keine Einträge.",
    "ui.audit.verified": "OK: {count} Einträge, Hash-Kette intakt.",
    "ui.audit.broken": "FEHLER: Hash-Kette gebrochen in Zeile {line}.",
    "ui.audit.unreadable": "{count} unlesbare Zeile(n) übersprungen.",
    "ui.audit.records": "{count} Eintrag/Einträge",
    "ui.reply.status": "{status}: {message}",
    "ui.error": "Fehler: {detail}",
    "ui.hotkey.hint": "Hotkey {hotkey} ist registriert.",
    "ui.hotkey.unavailable": (
        "Der globale Hotkey {hotkey} ist hier nicht verfügbar; nutze das "
        "Tray-Menü oder das Fenster."
    ),

}

_EN = {
    "invalid.empty": "Please enter a request.",
    "invalid.too_long": "Request is too long (maximum {limit} characters).",
    "invalid.not_text": "Please enter your request as text.",
    "unknown.request": "Sorry, I don't know how to do that. {capabilities}",
    "unknown.capabilities": (
        "I can open and close registered apps, open registered files, folders and "
        "websites, find files, search the web, report system information, calculate "
        "and set timers."
    ),
    "low_confidence": (
        "I'm not confident enough about this request ({confidence}). "
        "Could you rephrase it more explicitly?"
    ),
    "ambiguous_intent": (
        "This request is ambiguous between '{top}' ({top_confidence}) and "
        "'{second}' ({second_confidence}). Please be more specific."
    ),
    "internal.model": "The intent model failed to process this request.",
    "internal.prediction": "The intent model returned no usable prediction.",
    "target.not_found": (
        "No approved target matches this request. Only targets from the registry can "
        "be used."
    ),
    "target.not_found.suggestion": " Did you mean '{suggestion}'?",
    "target.ambiguous": "Multiple entries match this request ({names}). Please name exactly one.",
    "target.mismatch": (
        "'{entry}' is a registered {kind}, but this request sounds like a different "
        "action. Please rephrase and mention the {kind} explicitly."
    ),
    "target.mismatch.generic": (
        "That target does not match this action. Please rephrase and name the target "
        "explicitly."
    ),
    "parameter.calc": "No arithmetic expression found. Example: 'was ist 12*4'.",
    "parameter.timer": "No duration found. Example: 'stelle einen timer auf 10 minuten'.",
    "parameter.search": "No search term found. Example: 'finde die datei rechnung'.",
    "parameter.no_searchable_folders": "No searchable folders are configured in the registry.",
    "parameter.web_query": "No search query found. Example: 'suche im internet nach katzen'.",
    "parameter.no_searcher": "No search provider is configured in the registry.",
    "unsafe_request": "That request was refused for safety reasons.",
    "untrusted_target": (
        "That target is not in a trusted location, so I will not act on it."
    ),
    "audit_unavailable": (
        "The audit log is currently unavailable, so I refuse to execute anything for "
        "safety reasons."
    ),
    "executor_unavailable": "This action is not available here: {detail}",
    "execution_error": "The action failed: {detail}",
    "confirmation.required": (
        "I need your confirmation: {action} Answer with 'yes' or 'no'."
    ),
    "confirmation.declined": "Okay, I did not do that.",
    "confirmation.not_pending": "There is nothing to confirm right now.",
    "confirmation.yes": "yes",
    "confirmation.no": "no",
    "clarification.prompt": "Did you mean {options}? Please answer with the number.",
    "clarification.not_pending": "There is no open question right now.",
    "clarification.invalid_choice": "Please answer with a number between 1 and {count}.",
    "action.open_app.ok": "Opened app '{id}'.",
    "action.open_app.failed": "Could not start '{id}': {detail}.",
    "action.open_app.missing": "Application not found: {path}.",
    "action.close_app.ok": "Terminated {count} process(es) of '{id}'.",
    "action.close_app.none": "No running process of '{id}' found.",
    "action.close_app.unresolved": "Could not resolve the registered path: {path}.",
    "action.open_url.ok": "Opened website '{id}' in the browser.",
    "action.open_url.no_browser": "No web browser available.",
    "action.open_file.ok": "Opened file '{id}'.",
    "action.open_folder.ok": "Opened folder '{id}'.",
    "action.open_path.missing": "Path not found: {path}.",
    "action.open_folder.not_a_folder": "That target is not a folder: {path}.",
    "action.find_file.results": "Found {count} file(s) matching '{term}':\n{listing}",
    "action.find_file.none": "No files matching '{term}' found in the approved folders.",
    "action.find_file.truncated": (
        " (Search stopped after {visited} entries / {seconds} s; the result may be "
        "incomplete.)"
    ),
    "action.web_search.ok": "Started a web search for '{query}' ({url}).",
    "action.web_search.no_browser": "No web browser available.",
    "action.sys_cpu": "CPU usage: {usage}% ({cores} logical cores).",
    "action.sys_ram": "RAM: {used} of {total} used ({percent}%), {available} available.",
    "action.sys_disk": "Disk ({root}): {used} of {total} used ({percent}%), {free} free.",
    "action.sys_summary": "System summary:\n{body}",
    "action.calc": "{expression} = {result}",
    "action.timer.started": "Timer started: {human}.",
    "action.timer.limit": "{max} timers are already running; please wait or stop one.",
    "action.timer.cancelled": "Stopped {count} timer(s).",
    "action.timer.not_found": "I could not find a matching timer.",
    "action.dry_run.prefix": "[dry-run] ",
    "action.dry_run.open_app": "Would open app '{id}' ({path}).",
    "action.dry_run.close_app": "Would terminate processes of '{id}' ({path}).",
    "action.dry_run.open_url": "Would open website '{id}' ({url}).",
    "action.dry_run.open_file": "Would open file '{id}' ({path}).",
    "action.dry_run.open_folder": "Would open folder '{id}' ({path}).",
    "action.dry_run.find_file": "Would search for '{term}' in {count} approved folder(s).",
    "action.dry_run.web_search": "Would search the web for '{query}' ({url}).",
    "action.dry_run.sys": "Would query system information.",
    "action.dry_run.timer": "Would start a timer of {seconds} seconds.",
    "action.calc.dry_run": "Calculated {expression} = {result} (no side effects).",
    "timer.elapsed": "Timer finished ({seconds} seconds).",
    "search.none_found": "No matches.",
    # -- desktop UI -----------------------------------------------------------
    "ui.welcome": (
        "Welcome to MiniPCAI. Ask in German, e.g. 'öffne notepad', "
        "'wie viel ram ist frei' or 'stelle einen timer auf 5 minuten'. "
        "{registry} | dataset v{dataset}."
    ),
    "ui.you": 'You',
    "ui.assistant": 'MiniPCAI',
    "ui.send": 'Send',
    "ui.stop": 'Stop',
    "ui.placeholder": 'Request in German ...',
    "ui.chat.cleared": 'Chat cleared.',
    "ui.audit.warning": (
        "Warning: the audit log is not writable ({detail}). "
        "Accepted requests will be refused until this is fixed."
    ),
    "ui.plan.title": 'Plan',
    "ui.plan.target": 'target: {target}',
    "ui.plan.parameters": 'parameters: {parameters}',
    "ui.plan.dry_run": 'dry-run: nothing will be executed',
    "ui.plan.executor": 'executor: {mode}',
    "ui.meta": '{intent} | confidence {confidence} | reason: {reason}',
    "ui.status.bar": (
        "Model: {labels} intents | Executor: {mode} | {registry} | Audit: {audit}"
    ),
    "ui.executor.switched_windows": (
        "Switched to the Windows executor: requests will now perform real actions."
    ),
    "ui.executor.switched_dry_run": (
        "Switched to the dry-run executor: no real actions will be performed."
    ),
    "ui.executor.non_windows": (
        "Note: this is not a Windows machine, so actions that need the operating "
        "system will be reported as unavailable. Informational actions still work."
    ),
    "ui.confirm.title": 'Confirmation required',
    "ui.confirm.approved": 'Confirmed - the action will run.',
    "ui.confirm.declined": 'Cancelled - nothing will be executed.',
    "ui.confirm.question": '{action} Really execute it?',
    "ui.confirm.yes": 'Yes',
    "ui.confirm.no": 'No',
    "ui.menu.file": '&File',
    "ui.menu.quit": 'Quit',
    "ui.menu.view": '&View',
    "ui.menu.theme_system": 'Theme: system',
    'ui.menu.theme': 'Theme',
    'ui.executor.dry_run': 'Dry run (no real actions)',
    'ui.executor.windows': 'Windows (real actions)',
    "ui.menu.theme_light": 'Theme: light',
    "ui.menu.theme_dark": 'Theme: dark',
    "ui.menu.panels": 'Sidebar',
    "ui.menu.executor": '&Executor',
    "ui.menu.chat": '&Chat',
    "ui.menu.clear_chat": '&Clear chat',
    "ui.menu.settings": 'Settings',
    "ui.menu.registry_editor": 'Registry editor',
    "ui.menu.timer_panel": 'Timers',
    "ui.menu.audit_viewer": 'Audit log',
    "ui.menu.help": '&Help',
    "ui.menu.about": '&About MiniPCAI',
    "ui.about.text": (
        "MiniPCAI {version}\\n\\nA small, self-trained assistant that understands "
        "German requests and performs safe, registry-based actions.\\n\\n"
        "Targets come only from a validated registry; no shell, no arbitrary "
        "commands, no invented paths. Every request is written to an audit log."
    ),
    "ui.timer.title": 'Timers',
    "ui.timer.remaining": 'Remaining',
    "ui.timer.id": 'Request',
    "ui.timer.duration": 'Duration',
    "ui.timer.cancel_selected": 'Cancel selected',
    "ui.timer.cancel_all": 'Cancel all',
    "ui.timer.empty": 'No timers running.',
    "ui.timer.cancelled": 'Cancelled {count} timer(s).',
    "ui.timer.finished": 'Timer finished ({seconds} seconds).',
    "ui.registry.title": 'Registry editor',
    "ui.registry.reload": 'Reload',
    "ui.registry.save": 'Save',
    "ui.registry.add": 'Add entry',
    "ui.registry.remove": 'Remove selected',
    "ui.registry.path": 'file: {path}',
    "ui.registry.saved": 'Registry saved: {path}',
    "ui.registry.save_failed": 'Registry was not saved: {detail}',
    "ui.registry.add_prompt": 'ID=path (websites: ID=URL):',
    "ui.registry.section_prompt": 'Section (apps, files, folders, websites, searchers):',
    "ui.registry.confirm_remove": "Really remove the entry '{id}'?",
    "ui.registry.user_hint": 'Editing {path}',
    "ui.settings.title": 'Settings',
    "ui.settings.executor": 'Executor',
    "ui.settings.language": 'Language',
    "ui.settings.theme": 'Theme',
    "ui.settings.privacy": 'Do not store request texts in the audit log',
    "ui.settings.confirmations": 'Ask for confirmation for: {intents}',
    "ui.settings.hotkey": 'Global hotkey',
    "ui.settings.saved": 'Settings saved: {path}',
    "ui.settings.save_failed": 'Could not save the settings: {detail}',
    "ui.audit.title": 'Audit log',
    "ui.audit.reload": 'Reload',
    "ui.audit.verify": 'Verify hash chain',
    "ui.audit.empty": 'No records.',
    "ui.audit.verified": 'OK: {count} records, hash chain intact.',
    "ui.audit.broken": 'FAILED: hash chain broken at line {line}.',
    "ui.audit.unreadable": '{count} unreadable line(s) skipped.',
    "ui.audit.records": '{count} record(s)',
    "ui.reply.status": '{status}: {message}',
    "ui.error": 'Error: {detail}',
    "ui.hotkey.hint": 'Hotkey {hotkey} is registered.',
    "ui.hotkey.unavailable": (
        "The global hotkey {hotkey} is not available here; "
        "use the tray menu or the window instead."
    ),

}

_CATALOGS: dict[str, dict[str, str]] = {"en": _EN, "de": _DE}

#: Languages the UI and CLI accept (``--language``).
LANGUAGES: tuple[str, ...] = ("en", "de")


def normalize_language(language: str | None) -> str:
    """Return a supported language code, falling back to the default."""
    if not language:
        return DEFAULT_LANGUAGE
    code = language.strip().lower().replace("_", "-")
    if code in _CATALOGS:
        return code
    code = code.split("-", 1)[0]
    return code if code in _CATALOGS else DEFAULT_LANGUAGE


def t(key: str, language: str | None = None, **fields: Any) -> str:
    """Translate ``key`` into ``language``; falls back to English, then the key.

    Formatting errors are tolerated on purpose: a missing field must never
    crash a user-facing reply.
    """
    code = normalize_language(language)
    template = _CATALOGS[code].get(key) or _EN.get(key)
    if template is None:
        return key
    try:
        return template.format(**fields)
    except (KeyError, IndexError, ValueError):
        return template


def has_key(key: str, language: str | None = None) -> bool:
    code = normalize_language(language)
    return key in _CATALOGS[code] or key in _EN


def _fill(template: str, fields: dict[str, Any]) -> str:
    """Format ``template`` with ``fields``, leaving unknown placeholders be."""
    import string

    try:
        return string.Formatter().vformat(template, (), _SafeMapping(fields))
    except (ValueError, IndexError):
        return template


class _SafeMapping(dict):
    """Mapping that returns the original placeholder for unknown keys."""

    def __missing__(self, key: str) -> str:  # noqa: D105 - documented by context
        return "{" + key + "}"


def render(
    key: str, language: str | None, fallback: str | None = None, **fields: Any
) -> str:
    """Translate ``key``; use ``fallback`` when no catalog entry exists.

    Placeholders in the fallback are filled as well, so callers can pass an
    English sentence with ``{fields}`` and still get a readable message.
    """
    if not key or not has_key(key, language):
        template = fallback or fields.get("detail") or key
        return _fill(str(template), fields)
    return t(key, language, **fields)
