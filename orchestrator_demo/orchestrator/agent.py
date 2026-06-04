"""Agent-facing facade and ADK Dev UI wrapper for the orchestrator service."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, is_dataclass
from typing import Any

from google.adk.agents import Agent
from google.adk.tools import FunctionTool
from pydantic import BaseModel

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.app.bootstrap_llm import build_litellm_model
from orchestrator_demo.intent.classifier import LiteLlmIntentClassifier
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


class AdkOrchestratorAdapter:
    """Expose orchestrator request and approval operations as ADK tools."""

    def __init__(self, agent: OrchestratorAgent | None = None) -> None:
        self._agent = agent or OrchestratorAgent()

    def tools(self) -> list[Any]:
        """Return ADK function tools backed by one stateful orchestrator agent."""

        return [
            FunctionTool(self.submit_orchestrator_request),
            FunctionTool(self.approve_orchestrator_plan),
            FunctionTool(self.reject_orchestrator_plan),
        ]

    async def submit_orchestrator_request(self, user_input: str) -> dict[str, Any]:
        """Submit a natural-language RM request to the orchestrator."""

        result = await self._agent.handle_request(user_input)
        return _request_result_payload(result)

    async def approve_orchestrator_plan(
        self,
        plan_id: str,
        surface_id: str,
        approved_step_ids: list[str],
        edited_plan_version: int = 1,
    ) -> dict[str, Any]:
        """Approve a pending A2UI plan and execute the approved workflow."""

        result = await self._agent.handle_user_action(
            {
                "userAction": {
                    "type": "approve_plan",
                    "surfaceId": surface_id,
                    "payload": {
                        "planId": plan_id,
                        "editedPlanVersion": edited_plan_version,
                        "approvedStepIds": approved_step_ids,
                    },
                }
            }
        )
        return _user_action_result_payload(result)

    async def reject_orchestrator_plan(
        self,
        plan_id: str,
        surface_id: str,
        reason: str,
        edited_plan_version: int = 1,
    ) -> dict[str, Any]:
        """Reject a pending A2UI plan without executing specialists."""

        result = await self._agent.handle_user_action(
            {
                "userAction": {
                    "type": "reject_plan",
                    "surfaceId": surface_id,
                    "payload": {
                        "planId": plan_id,
                        "editedPlanVersion": edited_plan_version,
                        "reason": reason,
                    },
                }
            }
        )
        return _user_action_result_payload(result)


def build_root_agent(
    *,
    adapter: AdkOrchestratorAdapter | None = None,
    model: Any | None = None,
) -> Agent:
    """Build the ADK loader-compatible root agent for ``adk web``."""

    resolved_model = model if model is not None else build_litellm_model()
    resolved_adapter = adapter or AdkOrchestratorAdapter(
        agent=_runtime_orchestrator_agent(resolved_model)
    )

    return Agent(
        name="orchestrator",
        model=resolved_model,
        description=(
            "ADK Dev UI wrapper around the local ADK/A2UI business banking "
            "orchestrator demo."
        ),
        instruction=(
            "Use the tools to submit relationship-manager requests to the "
            "orchestrator. Return concise JSON-oriented summaries. If a request "
            "returns path plan_required, tell the user the planId, surfaceId, "
            "step ids, and that they can call the approval or rejection tool. "
            "This ADK Web surface does not render A2UI components; it exposes "
            "the orchestrator through tools for debugging."
        ),
        tools=resolved_adapter.tools(),
    )


_ROOT_AGENT: Agent | None = None


def __getattr__(name: str) -> Any:
    """Lazily expose ``root_agent`` for Google ADK's agent loader."""

    if name != "root_agent":
        raise AttributeError(name)

    global _ROOT_AGENT
    if _ROOT_AGENT is None:
        _ROOT_AGENT = build_root_agent()
    return _ROOT_AGENT


def _runtime_orchestrator_agent(model: Any) -> OrchestratorAgent:
    return OrchestratorAgent(
        service=OrchestratorService(
            intent_classifier=_intent_classifier_for_model(model),
        )
    )


def _intent_classifier_for_model(model: Any) -> LiteLlmIntentClassifier:
    if isinstance(model, str):
        return LiteLlmIntentClassifier(model_name=model)
    return LiteLlmIntentClassifier(model=model)


def _request_result_payload(result: OrchestratorRequestResult) -> dict[str, Any]:
    return {
        "path": result.path,
        "decision": _jsonable(result.decision),
        "approvalPlan": _jsonable(result.approval_plan),
        "approvalResult": _approval_result_payload(result.approval_result),
        "specialistResponses": _jsonable(result.specialist_responses),
        "a2uiParts": _data_parts_payload(result.a2ui_parts),
        "statusEvents": _jsonable(result.status_events),
        "artifacts": _jsonable(result.final_artifacts),
    }


def _user_action_result_payload(
    result: OrchestratorUserActionResult,
) -> dict[str, Any]:
    return {
        "status": result.status,
        "approvalResult": _approval_result_payload(result.approval_result),
        "surfaceRouteResult": _jsonable(result.surface_route_result),
        "specialistResponses": _jsonable(result.specialist_responses),
        "a2uiParts": _data_parts_payload(result.a2ui_parts),
        "statusEvents": _jsonable(result.status_events),
        "artifacts": _jsonable(result.final_artifacts),
    }


def _approval_result_payload(result: Any) -> dict[str, Any] | None:
    if result is None:
        return None
    return {
        "status": result.status,
        "planId": result.plan_id,
        "planVersion": result.plan_version,
        "approvedPlan": _jsonable(result.approved_plan),
        "reason": result.rejection_reason,
        "graphCreated": result.graph_created,
        "specialistsCalled": result.specialists_called,
    }


def _data_parts_payload(parts: Sequence[DataPart]) -> list[dict[str, Any]]:
    return [part.model_dump(by_alias=True, mode="json") for part in parts]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_jsonable(child) for child in value]
    return str(value)


__all__ = [
    "AdkOrchestratorAdapter",
    "OrchestratorAgent",
    "build_root_agent",
]
