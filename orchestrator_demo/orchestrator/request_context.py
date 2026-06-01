"""Per-request routing state and specialist-call guardrails."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from orchestrator_demo.contracts import (
    ExecutionPlan,
    IntentSuggestion,
    LlmIntentAssessment,
    PlanStep,
    RoutingDecision,
    SpecialistRequest,
    SpecialistResponse,
)


SpecialistCallable = Callable[[SpecialistRequest], Awaitable[SpecialistResponse]]


class SpecialistPreApprovalError(RuntimeError):
    """Raised when a specialist call would bypass required approval state."""


class PlanApprovalStateError(RuntimeError):
    """Raised when approval state would be mutated after being frozen."""


@dataclass(frozen=True)
class _ApprovedStepPayload:
    agent_id: str
    user_input: str
    context: dict[str, Any]


def _new_plan_scope_id() -> str:
    return uuid4().hex[:12]


def _scope_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    token = token.strip("_-")
    return token or "unknown"


def _plan_snapshot(plan: ExecutionPlan) -> dict[str, Any]:
    return plan.model_dump(mode="json")


@dataclass
class RequestContext:
    """Structured state produced by routing a new user request."""

    user_input: str
    slm_suggestion: IntentSuggestion
    llm_assessment: LlmIntentAssessment
    decision: RoutingDecision
    plan_scope_id: str = field(default_factory=_new_plan_scope_id)
    draft_plan_id: str | None = field(default=None, init=False)
    draft_approval_surface_id: str | None = field(default=None, init=False)
    _draft_plan_snapshot: dict[str, Any] | None = field(
        default=None,
        init=False,
        repr=False,
    )
    approved_plan_id: str | None = None
    approved_plan_step_agents: dict[str, dict[str, str]] = field(default_factory=dict)
    approved_plan_step_payloads: dict[str, dict[str, _ApprovedStepPayload]] = field(
        default_factory=dict
    )

    @property
    def has_structured_approval(self) -> bool:
        return (
            self.approved_plan_id is not None
            and self.approved_plan_id in self.approved_plan_step_payloads
        )

    def record_draft_plan(self, plan: ExecutionPlan) -> None:
        self._require_plan_matches_request_scope(plan)
        if self.approved_plan_id is not None and plan.plan_id != self.approved_plan_id:
            raise PlanApprovalStateError(
                "approved plan state is immutable after structured approval"
            )

        self.draft_plan_id = plan.plan_id
        self.draft_approval_surface_id = plan.approval_surface_id
        self._draft_plan_snapshot = _plan_snapshot(plan)

    def mark_plan_approved(self, plan: ExecutionPlan) -> None:
        approved_step_agents = {step.step_id: step.agent_id for step in plan.steps}
        approved_step_payloads = {
            step.step_id: _ApprovedStepPayload(
                agent_id=step.agent_id,
                user_input=step.instruction,
                context=_approved_step_context(plan, step),
            )
            for step in plan.steps
        }

        if self.approved_plan_id is not None:
            if (
                self.approved_plan_id == plan.plan_id
                and self.approved_plan_step_agents.get(plan.plan_id)
                == approved_step_agents
                and self.approved_plan_step_payloads.get(plan.plan_id)
                == approved_step_payloads
            ):
                return
            raise PlanApprovalStateError(
                "approved plan state is immutable after structured approval"
            )

        self._require_plan_matches_current_draft(plan)
        self.approved_plan_id = plan.plan_id
        self.approved_plan_step_agents[plan.plan_id] = approved_step_agents
        self.approved_plan_step_payloads[plan.plan_id] = approved_step_payloads

    def _require_plan_matches_current_draft(self, plan: ExecutionPlan) -> None:
        self._require_plan_matches_request_scope(plan)
        if (
            self.draft_plan_id != plan.plan_id
            or self.draft_approval_surface_id != plan.approval_surface_id
            or self._draft_plan_snapshot != _plan_snapshot(plan)
        ):
            raise PlanApprovalStateError(
                "approved plan must match the current draft for this request"
            )

    def _require_plan_matches_request_scope(self, plan: ExecutionPlan) -> None:
        if self.decision.path != "plan_required":
            raise PlanApprovalStateError(
                "only plan_required routes can approve execution plans"
            )
        if plan.objective != self.user_input:
            raise PlanApprovalStateError(
                "approved plan objective must match the routed request"
            )

        expected_scope_suffix = f"_{_scope_token(self.plan_scope_id)}"
        if not plan.plan_id.endswith(expected_scope_suffix):
            raise PlanApprovalStateError(
                "approved plan must match the request plan_scope_id"
            )

        expected_surface_id = f"surface_{plan.plan_id}"
        if plan.approval_surface_id != expected_surface_id:
            raise PlanApprovalStateError(
                "approved plan surface must match the request-scoped plan_id"
            )

    def require_specialist_call_allowed(self, request: SpecialistRequest) -> None:
        if self.decision.path == "direct":
            if request.agent_id != self.decision.selected_agent:
                raise SpecialistPreApprovalError(
                    "direct route specialist call must target selected agent "
                    f"{self.decision.selected_agent!r}, got {request.agent_id!r}"
                )
            if request.user_input != self.user_input:
                raise SpecialistPreApprovalError(
                    "direct route specialist call user_input must match the "
                    "original routed user_input"
                )
            if request.context != {}:
                raise SpecialistPreApprovalError(
                    "direct route specialist call context must match the "
                    "original routed context"
                )
            if request.plan_id is not None or request.step_id is not None:
                raise SpecialistPreApprovalError(
                    "direct route specialist call must not include plan_id or step_id"
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
            approved_plan_id = request.plan_id
            if approved_plan_id is None:
                raise SpecialistPreApprovalError(
                    "complex route specialist call must include the approved plan_id"
                )
            approved_step_payloads = self.approved_plan_step_payloads.get(
                approved_plan_id
            )
            if approved_step_payloads is None:
                raise SpecialistPreApprovalError(
                    "complex route specialist call must target a stored approved plan"
                )
            if request.step_id is None:
                raise SpecialistPreApprovalError(
                    "complex route specialist call must include an approved step_id"
                )
            approved_payload = approved_step_payloads.get(request.step_id)
            if approved_payload is None:
                raise SpecialistPreApprovalError(
                    "complex route specialist call must target an approved step_id "
                    f"for plan {request.plan_id!r}, got {request.step_id!r}"
                )
            if request.agent_id != approved_payload.agent_id:
                raise SpecialistPreApprovalError(
                    "complex route specialist call agent_id must match the approved "
                    f"agent for step {request.step_id!r}: expected "
                    f"{approved_payload.agent_id!r}, got {request.agent_id!r}"
                )
            if request.user_input != approved_payload.user_input:
                raise SpecialistPreApprovalError(
                    "complex route specialist call user_input must match the "
                    f"approved instruction for step {request.step_id!r}"
                )
            if request.context != approved_payload.context:
                raise SpecialistPreApprovalError(
                    "complex route specialist call context must match the approved "
                    f"context for step {request.step_id!r}"
                )
            return

        raise SpecialistPreApprovalError(
            "specialist calls are not allowed while clarification is required"
        )


def _approved_step_context(
    plan: ExecutionPlan,
    step: PlanStep,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "objective": plan.objective,
        "planVersion": plan.plan_version,
        "expectedOutput": step.expected_output,
        "dataSourceCategories": list(step.data_source_categories),
        "dependsOn": list(step.depends_on),
    }
    if step.parallel_group is not None:
        context["parallelGroup"] = step.parallel_group

    return context


async def call_specialist_with_guard(
    context: RequestContext,
    request: SpecialistRequest,
    specialist: SpecialistCallable,
) -> SpecialistResponse:
    """Apply request guardrails before invoking a specialist."""

    context.require_specialist_call_allowed(request)
    return await specialist(request)


__all__ = [
    "PlanApprovalStateError",
    "RequestContext",
    "SpecialistCallable",
    "SpecialistPreApprovalError",
    "call_specialist_with_guard",
]
