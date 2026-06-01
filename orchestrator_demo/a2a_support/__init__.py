"""A2A adapter and local remote-compatible wrapper support."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_LAZY_EXPORTS = {
    "DEFAULT_LOCAL_REMOTE_AGENT_IDS": (
        "orchestrator_demo.a2a_support.local_remote_wrapper"
    ),
    "LocalRemoteAgentWrapper": "orchestrator_demo.a2a_support.local_remote_wrapper",
    "RemoteA2AAgentAdapter": "orchestrator_demo.a2a_support.remote_agent_adapter",
    "UserActionPayload": "orchestrator_demo.a2a_support.remote_agent_adapter",
    "build_default_local_remote_wrappers": (
        "orchestrator_demo.a2a_support.local_remote_wrapper"
    ),
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY_EXPORTS})


__all__ = [
    "DEFAULT_LOCAL_REMOTE_AGENT_IDS",
    "LocalRemoteAgentWrapper",
    "RemoteA2AAgentAdapter",
    "UserActionPayload",
    "build_default_local_remote_wrappers",
]
