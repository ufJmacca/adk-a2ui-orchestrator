"""Per-request routing state and specialist-call guardrails."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from orchestrator_demo.contracts import (
    IntentSuggestion,
    LlmIntentAssessment,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
)


SpecialistCallable = Callable[[SpecialistRequest], Awaitable[SpecialistResponse]]


class SpecialistPreApprovalError(RuntimeError):
    """Raised when a specialist call would bypass required approval state."""


@dataclass
class RequestContext:
    """Structured state produced by routing a new user request."""

    user_input: str
    slm_suggestion: IntentSuggestion
    llm_assessment: LlmIntentAssessment
    decision: RoutingDecision
    approved_plan_id: str | None = None

    @property
    def has_structured_approval(self) -> bool:
        return self.approved_plan_id is not None

    def mark_plan_approved(self, plan_id: str) -> None:
        self.approved_plan_id = plan_id

    def require_specialist_call_allowed(self, request: SpecialistRequest) -> None:
        if self.decision.path == "direct":
            if request.agent_id != self.decision.selected_agent:
                raise SpecialistPreApprovalError(
                    "direct route specialist call must target selected agent "
                    f"{self.decision.selected_agent!r}, got {request.agent_id!r}"
                )
            return

        if self.decision.path == "plan_required":
            if not self.has_structured_approval:
                raise SpecialistPreApprovalError(
                    "complex route requires structured approval before specialist calls"
                )
            if request.plan_id != self.approved_plan_id:
                raise SpecialistPreApprovalError(
                    "complex route specialist call must include the approved plan_id "
                    f"{self.approved_plan_id!r}, got {request.plan_id!r}"
                )
            return

        raise SpecialistPreApprovalError(
            "specialist calls are not allowed while clarification is required"
        )


async def call_specialist_with_guard(
    context: RequestContext,
    request: SpecialistRequest,
    specialist: SpecialistCallable,
) -> SpecialistResponse:
    """Apply request guardrails before invoking a specialist."""

    context.require_specialist_call_allowed(request)
    return await specialist(request)


__all__ = [
    "RequestContext",
    "SpecialistCallable",
    "SpecialistPreApprovalError",
    "call_specialist_with_guard",
]
