"""Orchestration, planning, approval, routing, and graph execution support."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def __getattr__(name: str) -> Any:
    """Lazily re-export ADK loader entry points from the agent module."""

    if name in {"app", "root_agent"}:
        agent_module = import_module(f"{__name__}.agent")
        return getattr(agent_module, name)
    raise AttributeError(name)


__all__ = ["app", "root_agent"]
