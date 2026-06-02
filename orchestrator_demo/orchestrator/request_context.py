"""Per-request routing state and specialist-call guardrails."""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any
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

if TYPE_CHECKING:
    from orchestrator_demo.orchestrator.approval_state import ApprovalRecord


SpecialistCallable = Callable[[SpecialistRequest], Awaitable[SpecialistResponse]]


class SpecialistPreApprovalError(RuntimeError):
    """Raised when a specialist call would bypass required approval state."""


class PlanApprovalStateError(RuntimeError):
    """Raised when approval state would be mutated after being frozen."""


_RUNTIME_DEPENDENCY_CONTEXT_KEYS = frozenset({"dependencyOutputs", "stepResults"})


@dataclass(frozen=True)
class _ApprovedStepPayload:
    agent_id: str
    user_input: str
    context: Mapping[str, Any]


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
    _approved_plan_step_agents: dict[str, Mapping[str, str]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _approved_plan_step_payloads: dict[
        str,
        Mapping[str, _ApprovedStepPayload],
    ] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    @property
    def approved_plan_step_agents(self) -> Mapping[str, Mapping[str, str]]:
        return MappingProxyType(self._approved_plan_step_agents)

    @property
    def approved_plan_step_payloads(
        self,
    ) -> Mapping[str, Mapping[str, _ApprovedStepPayload]]:
        return MappingProxyType(self._approved_plan_step_payloads)

    @property
    def has_structured_approval(self) -> bool:
        return (
            self.approved_plan_id is not None
            and self.approved_plan_id in self._approved_plan_step_payloads
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

    def mark_plan_approved(
        self,
        plan: ExecutionPlan,
        *,
        approval_record: ApprovalRecord | None = None,
    ) -> None:
        approved_step_agents = MappingProxyType(
            {step.step_id: step.agent_id for step in plan.steps}
        )
        approved_step_payloads = MappingProxyType(
            {
                step.step_id: _ApprovedStepPayload(
                    agent_id=step.agent_id,
                    user_input=step.instruction,
                    context=_freeze_approval_value(
                        _approved_step_context(plan, step)
                    ),
                )
                for step in plan.steps
            }
        )

        if self.approved_plan_id is not None:
            if (
                self.approved_plan_id == plan.plan_id
                and self._approved_plan_step_agents.get(plan.plan_id)
                == approved_step_agents
                and self._approved_plan_step_payloads.get(plan.plan_id)
                == approved_step_payloads
            ):
                return
            raise PlanApprovalStateError(
                "approved plan state is immutable after structured approval"
            )

        self._require_plan_matches_current_draft(
            plan,
            approval_record=approval_record,
        )
        self.approved_plan_id = plan.plan_id
        self._approved_plan_step_agents[plan.plan_id] = approved_step_agents
        self._approved_plan_step_payloads[plan.plan_id] = approved_step_payloads

    def _require_plan_matches_current_draft(
        self,
        plan: ExecutionPlan,
        *,
        approval_record: ApprovalRecord | None,
    ) -> None:
        self._require_plan_matches_request_scope(plan)
        if (
            self.draft_plan_id != plan.plan_id
            or self.draft_approval_surface_id != plan.approval_surface_id
        ):
            raise PlanApprovalStateError(
                "approved plan must match the current draft for this request"
            )

        if self._draft_plan_snapshot == _plan_snapshot(plan):
            return

        if _approval_record_matches_approved_plan(approval_record, plan):
            return

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
            approved_step_payloads = self._approved_plan_step_payloads.get(
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
            if not _approved_context_allows_request_context(
                request.context,
                approved_payload.context,
            ):
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


def _approval_record_matches_approved_plan(
    approval_record: ApprovalRecord | None,
    plan: ExecutionPlan,
) -> bool:
    if approval_record is None or approval_record.status != "approved":
        return False
    if approval_record.approved_version != plan.plan_version:
        return False

    approved_plan = approval_record.approved_plan
    if approved_plan is None:
        return False

    return _plan_snapshot(approved_plan) == _plan_snapshot(plan)


def _approved_context_allows_request_context(
    request_context: Mapping[str, Any],
    approved_context: Mapping[str, Any],
) -> bool:
    approved_keys = set(approved_context)
    request_keys = set(request_context)
    runtime_keys = request_keys - approved_keys
    if not runtime_keys.issubset(_RUNTIME_DEPENDENCY_CONTEXT_KEYS):
        return False
    if not approved_keys.issubset(request_keys):
        return False

    static_request_context = {
        key: request_context[key]
        for key in approved_keys
    }
    if _freeze_approval_value(static_request_context) != approved_context:
        return False

    depends_on = approved_context.get("dependsOn", ())
    if not isinstance(depends_on, tuple) or any(
        not isinstance(step_id, str) for step_id in depends_on
    ):
        return False

    allowed_dependency_ids = set(depends_on)
    if allowed_dependency_ids and not runtime_keys:
        return False

    for key in runtime_keys:
        if not _runtime_dependency_outputs_allowed(
            request_context[key],
            allowed_dependency_ids,
        ):
            return False

    return True


def _runtime_dependency_outputs_allowed(
    value: Any,
    allowed_dependency_ids: set[str],
) -> bool:
    if not isinstance(value, Mapping):
        return False

    dependency_ids: set[str] = set()
    for step_id in value:
        if not isinstance(step_id, str):
            return False
        dependency_ids.add(step_id)

    return dependency_ids == allowed_dependency_ids


def _freeze_approval_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_approval_value(child_value)
                for key, child_value in value.items()
            }
        )
    if isinstance(value, list | tuple):
        return tuple(_freeze_approval_value(item) for item in value)
    if isinstance(value, set | frozenset):
        return frozenset(_freeze_approval_value(item) for item in value)
    return value


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
