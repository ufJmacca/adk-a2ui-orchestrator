"""Orchestrator-facing interface for remote-compatible specialist agents."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse, UserAction


UserActionPayload = UserAction | Mapping[str, Any]


@runtime_checkable
class RemoteA2AAgentAdapter(Protocol):
    """Async interface shared by future remote A2A agents and local wrappers."""

    @property
    def agent_id(self) -> str:
        """Stable specialist id used for deterministic routing."""

    async def run(self, request: SpecialistRequest) -> SpecialistResponse:
        """Execute a specialist request through the adapter boundary."""

    async def handle_user_action(
        self,
        user_action: UserActionPayload,
    ) -> SpecialistResponse:
        """Route an A2UI userAction to the owning specialist."""


__all__ = ["RemoteA2AAgentAdapter", "UserActionPayload"]
