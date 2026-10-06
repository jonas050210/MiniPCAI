"""Intent label definitions shared across training, pipeline and UI."""

from __future__ import annotations

# The 12 supported MVP actions.
ACTION_INTENTS: tuple[str, ...] = (
    "open_app",
    "close_app",
    "open_url",
    "open_file",
    "open_folder",
    "find_file",
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

# Short English descriptions used by the CLI/UI help output.
INTENT_DESCRIPTIONS: dict[str, str] = {
    "open_app": "Open a registered application",
    "close_app": "Close a registered application",
    "open_url": "Open a registered website in the browser",
    "open_file": "Open a registered file",
    "open_folder": "Open a registered folder",
    "find_file": "Search for files by name inside searchable registered folders",
    "sys_cpu": "Report CPU usage",
    "sys_ram": "Report RAM usage",
    "sys_disk": "Report disk usage",
    "sys_summary": "Report a combined system summary",
    "calc": "Evaluate a basic arithmetic expression",
    "timer": "Start a countdown timer",
    "unknown": "Request is unknown, unsupported or ambiguous",
}

# Which registry section provides the logical target for an intent.
TARGETED_INTENTS: dict[str, str] = {
    "open_app": "apps",
    "close_app": "apps",
    "open_url": "websites",
    "open_file": "files",
    "open_folder": "folders",
}

REGISTRY_SECTIONS: tuple[str, ...] = ("apps", "files", "folders", "websites")

# Intents that need no target resolution at all.
NO_TARGET_INTENTS: tuple[str, ...] = (
    "find_file",
    "sys_cpu",
    "sys_ram",
    "sys_disk",
    "sys_summary",
    "calc",
    "timer",
)
