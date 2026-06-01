"""Agent-facing facade for the orchestrator application service."""

from __future__ import annotations

from typing import Any

from orchestrator_demo.orchestrator.service import (
    OrchestratorRequestResult,
    OrchestratorService,
    OrchestratorUserActionResult,
)


class OrchestratorAgent:
    """Small facade matching the demo agent's request and userAction surfaces."""

    def __init__(self, service: OrchestratorService | None = None) -> None:
        self._service = service or OrchestratorService()

    async def handle_request(self, user_input: str) -> OrchestratorRequestResult:
        """Handle a natural-language request through the orchestrator service."""

        return await self._service.handle_user_request(user_input)

    async def handle_user_action(
        self,
        user_action: Any,
    ) -> OrchestratorUserActionResult:
        """Handle a structured A2UI userAction through deterministic routing."""

        return await self._service.handle_user_action(user_action)


__all__ = ["OrchestratorAgent"]
