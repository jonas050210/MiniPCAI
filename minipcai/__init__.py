"""MiniPCAI: a small, self-trained German-language assistant for safe Windows PC actions.

The package implements the core pipeline

    user text -> intent classifier -> confidence/ambiguity gates
              -> logical target resolution (registry only)
              -> security validation -> executor -> audit log

and is deliberately modular: the intent model, the registry, the executor and
the audit sink are all replaceable components behind small interfaces.
"""

from minipcai.config import Config, Thresholds, default_config
from minipcai.pipeline import Assistant, AssistantResult

__version__ = "0.1.0"

__all__ = [
    "Assistant",
    "AssistantResult",
    "Config",
    "Thresholds",
    "default_config",
    "__version__",
]
