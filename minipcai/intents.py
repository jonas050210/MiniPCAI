"""Intent label definitions shared across training, pipeline and UI."""

from __future__ import annotations

# The supported actions. Adding one here requires dataset examples (see
# scripts/generate_dataset.py) and, for new registry-backed intents, a section
# in the registry schema.
ACTION_INTENTS: tuple[str, ...] = (
    "open_app",
    "close_app",
    "open_url",
    "open_file",
    "open_folder",
    "find_file",
    "web_search",
    "sys_cpu",
    "sys_ram",
    "sys_disk",
    "sys_summary",
    "calc",
    "timer",
)

# Catch-all label for unknown, out-of-domain or ambiguous requests.
UNKNOWN_LABEL = "unknown"

ALL_LABELS: tuple[str, ...] = ACTION_INTENTS + (UNKNOWN_LABEL,)

# Short English descriptions used by the CLI help, the UI help panel and the
# "what can you do" answer.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "open_app": "Open a registered application",
    "close_app": "Close a registered application (asks before acting)",
    "open_url": "Open a registered website in the browser",
    "open_file": "Open a registered document",
    "open_folder": "Open a registered folder",
    "find_file": "Search for files by name inside searchable registered folders",
    "web_search": "Search the web through a registered search provider (asks before acting)",
    "sys_cpu": "Report CPU usage",
    "sys_ram": "Report RAM usage",
    "sys_disk": "Report disk usage",
    "sys_summary": "Report a combined system summary",
    "calc": "Evaluate a basic arithmetic expression",
    "timer": "Start a countdown timer",
    "unknown": "Request is unknown, unsupported or ambiguous",
}

# Which registry section provides the logical target for an intent. Intents
# with a free parameter (find_file, web_search, calc, timer) are resolved by
# dedicated parsers instead.
TARGETED_INTENTS: dict[str, str] = {
    "open_app": "apps",
    "close_app": "apps",
    "open_url": "websites",
    "open_file": "files",
    "open_folder": "folders",
}

PARAMETERIZED_INTENTS: tuple[str, ...] = ("find_file", "web_search", "calc", "timer")

REGISTRY_SECTIONS: tuple[str, ...] = ("apps", "files", "folders", "websites", "searchers")

# Human-readable capability overview used in the "unknown request" reply and by
# the help panel.
CAPABILITY_OVERVIEW: str = (
    "I can open and close registered apps, open registered files, folders and "
    "websites, find files, search the web, report system information, calculate "
    "and set timers."
)
