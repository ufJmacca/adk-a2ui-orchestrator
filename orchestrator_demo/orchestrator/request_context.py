"""Per-request routing state and specialist-call guardrails."""

from __future__ import annotations

import re
from collections.abc import Mapping
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
from orchestrator_demo.orchestrator.specialist_invocation import (
    SpecialistCallable,
    SpecialistLike,
    invoke_specialist,
)

if TYPE_CHECKING:
    from orchestrator_demo.orchestrator.approval_state import ApprovalRecord


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
    dependency_output_alternates: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )


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
    _specialist_surface_owners: dict[str, str] = field(
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
    def specialist_surface_owners(self) -> Mapping[str, str]:
        return MappingProxyType(self._specialist_surface_owners)

    @property
    def has_structured_approval(self) -> bool:
        return (
            self.approved_plan_id is not None
            and self.approved_plan_id in self._approved_plan_step_payloads
        )

    def specialist_owner_for_surface(self, surface_id: str) -> str | None:
        return self._specialist_surface_owners.get(surface_id)

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
                    dependency_output_alternates=MappingProxyType(
                        _conditional_default_dependency_alternates(plan, step)
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

    def reset_plan_approval(self, plan_id: str) -> None:
        """Clear approval guard state after graph execution fails before commit."""

        if self.approved_plan_id != plan_id:
            return

        self.approved_plan_id = None
        self._approved_plan_step_agents.pop(plan_id, None)
        self._approved_plan_step_payloads.pop(plan_id, None)

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
                approved_payload.dependency_output_alternates,
            ):
                raise SpecialistPreApprovalError(
                    "complex route specialist call context must match the approved "
                    f"context for step {request.step_id!r}"
                )
            return

        raise SpecialistPreApprovalError(
            "specialist calls are not allowed while clarification is required"
        )

    def record_specialist_surface_owner(
        self,
        *,
        surface_id: str | None,
        agent_id: str,
    ) -> None:
        if surface_id is None:
            return

        existing_owner = self._specialist_surface_owners.get(surface_id)
        if existing_owner is not None and existing_owner != agent_id:
            raise SpecialistPreApprovalError(
                "specialist surface owner conflict for "
                f"{surface_id!r}: expected {existing_owner!r}, got {agent_id!r}"
            )

        self._specialist_surface_owners[surface_id] = agent_id

    def export_snapshot(self) -> dict[str, Any]:
        """Return JSON-safe request guardrail state for ADK session storage."""

        return {
            "userInput": self.user_input,
            "slmSuggestion": self.slm_suggestion.model_dump(mode="json"),
            "llmAssessment": self.llm_assessment.model_dump(mode="json"),
            "decision": self.decision.model_dump(mode="json"),
            "planScopeId": self.plan_scope_id,
            "draftPlanId": self.draft_plan_id,
            "draftApprovalSurfaceId": self.draft_approval_surface_id,
            "draftPlanSnapshot": _json_safe(self._draft_plan_snapshot),
            "approvedPlanId": self.approved_plan_id,
            "approvedPlanStepAgents": _json_safe(self._approved_plan_step_agents),
            "approvedPlanStepPayloads": _approved_payloads_to_snapshot(
                self._approved_plan_step_payloads
            ),
            "specialistSurfaceOwners": dict(self._specialist_surface_owners),
        }

    @classmethod
    def restore_snapshot(cls, snapshot: Mapping[str, Any]) -> "RequestContext":
        """Restore request guardrail state from a JSON-safe snapshot."""

        context = cls(
            user_input=_required_snapshot_string(snapshot, "userInput"),
            slm_suggestion=IntentSuggestion.model_validate(
                _required_snapshot_mapping(snapshot, "slmSuggestion")
            ),
            llm_assessment=LlmIntentAssessment.model_validate(
                _required_snapshot_mapping(snapshot, "llmAssessment")
            ),
            decision=RoutingDecision.model_validate(
                _required_snapshot_mapping(snapshot, "decision")
            ),
            plan_scope_id=_required_snapshot_string(snapshot, "planScopeId"),
        )
        context.draft_plan_id = _optional_snapshot_string(snapshot.get("draftPlanId"))
        context.draft_approval_surface_id = _optional_snapshot_string(
            snapshot.get("draftApprovalSurfaceId")
        )
        draft_plan_snapshot = snapshot.get("draftPlanSnapshot")
        context._draft_plan_snapshot = (
            dict(draft_plan_snapshot)
            if isinstance(draft_plan_snapshot, Mapping)
            else None
        )
        context.approved_plan_id = _optional_snapshot_string(
            snapshot.get("approvedPlanId")
        )
        context._approved_plan_step_agents = _restore_step_agents(
            snapshot.get("approvedPlanStepAgents")
        )
        context._approved_plan_step_payloads = _restore_step_payloads(
            snapshot.get("approvedPlanStepPayloads")
        )
        context._specialist_surface_owners = _restore_string_map(
            snapshot.get("specialistSurfaceOwners")
        )
        return context


def _approved_payloads_to_snapshot(
    payloads_by_plan_id: Mapping[str, Mapping[str, _ApprovedStepPayload]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        plan_id: {
            step_id: {
                "agentId": payload.agent_id,
                "userInput": payload.user_input,
                "context": _json_safe(payload.context),
                "dependencyOutputAlternates": {
                    dependency_id: list(alternates)
                    for dependency_id, alternates in (
                        payload.dependency_output_alternates.items()
                    )
                },
            }
            for step_id, payload in step_payloads.items()
        }
        for plan_id, step_payloads in payloads_by_plan_id.items()
    }


def _restore_step_agents(candidate: Any) -> dict[str, Mapping[str, str]]:
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise PlanApprovalStateError("approvedPlanStepAgents must be an object")

    return {
        str(plan_id): MappingProxyType(_restore_string_map(step_agents))
        for plan_id, step_agents in candidate.items()
    }


def _restore_step_payloads(
    candidate: Any,
) -> dict[str, Mapping[str, _ApprovedStepPayload]]:
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise PlanApprovalStateError("approvedPlanStepPayloads must be an object")

    payloads_by_plan_id: dict[str, Mapping[str, _ApprovedStepPayload]] = {}
    for plan_id, step_payloads in candidate.items():
        if not isinstance(step_payloads, Mapping):
            raise PlanApprovalStateError(
                "approvedPlanStepPayloads plan entries must be objects"
            )
        payloads_by_plan_id[str(plan_id)] = MappingProxyType(
            {
                str(step_id): _restore_step_payload(payload)
                for step_id, payload in step_payloads.items()
            }
        )
    return payloads_by_plan_id


def _restore_step_payload(candidate: Any) -> _ApprovedStepPayload:
    if not isinstance(candidate, Mapping):
        raise PlanApprovalStateError("approved step payload must be an object")

    context = candidate.get("context")
    if not isinstance(context, Mapping):
        raise PlanApprovalStateError("approved step payload context must be an object")

    dependency_output_alternates = candidate.get("dependencyOutputAlternates", {})
    if not isinstance(dependency_output_alternates, Mapping):
        raise PlanApprovalStateError(
            "approved step dependency alternates must be an object"
        )

    return _ApprovedStepPayload(
        agent_id=_required_snapshot_string(candidate, "agentId"),
        user_input=_required_snapshot_string(candidate, "userInput"),
        context=_freeze_approval_value(context),
        dependency_output_alternates=MappingProxyType(
            {
                str(dependency_id): tuple(_restore_string_sequence(alternates))
                for dependency_id, alternates in dependency_output_alternates.items()
            }
        ),
    )


def _restore_string_map(candidate: Any) -> dict[str, str]:
    if candidate is None:
        return {}
    if not isinstance(candidate, Mapping):
        raise PlanApprovalStateError("snapshot field must be an object")
    restored: dict[str, str] = {}
    for key, value in candidate.items():
        if not isinstance(value, str):
            raise PlanApprovalStateError("snapshot map values must be strings")
        restored[str(key)] = value
    return restored


def _restore_string_sequence(candidate: Any) -> list[str]:
    if not isinstance(candidate, list):
        raise PlanApprovalStateError("snapshot sequence must be a list")
    if not all(isinstance(value, str) for value in candidate):
        raise PlanApprovalStateError("snapshot sequence values must be strings")
    return list(candidate)


def _required_snapshot_mapping(
    snapshot: Mapping[str, Any],
    field_name: str,
) -> Mapping[str, Any]:
    value = snapshot.get(field_name)
    if not isinstance(value, Mapping):
        raise PlanApprovalStateError(f"{field_name} must be an object")
    return value


def _required_snapshot_string(
    snapshot: Mapping[str, Any],
    field_name: str,
) -> str:
    value = snapshot.get(field_name)
    if not isinstance(value, str):
        raise PlanApprovalStateError(f"{field_name} must be a string")
    return value


def _optional_snapshot_string(candidate: Any) -> str | None:
    if candidate is None:
        return None
    if isinstance(candidate, str):
        return candidate
    raise PlanApprovalStateError("snapshot field must be a string or null")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(child) for child in value]
    if isinstance(value, set | frozenset):
        return sorted(_json_safe(child) for child in value)
    return value


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
    dependency_output_alternates: Mapping[str, tuple[str, ...]],
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
            dependency_output_alternates,
        ):
            return False

    return True


def _runtime_dependency_outputs_allowed(
    value: Any,
    allowed_dependency_ids: set[str],
    dependency_output_alternates: Mapping[str, tuple[str, ...]],
) -> bool:
    if not isinstance(value, Mapping):
        return False

    dependency_ids: set[str] = set()
    for step_id in value:
        if not isinstance(step_id, str):
            return False
        dependency_ids.add(step_id)

    allowed_dependency_sets = {frozenset(allowed_dependency_ids)}
    for dependency_id, alternates in dependency_output_alternates.items():
        if dependency_id not in allowed_dependency_ids:
            continue
        alternate_ids = set(allowed_dependency_ids)
        alternate_ids.remove(dependency_id)
        alternate_ids.update(alternates)
        allowed_dependency_sets.add(frozenset(alternate_ids))

    return frozenset(dependency_ids) in allowed_dependency_sets


def _conditional_default_dependency_alternates(
    plan: ExecutionPlan,
    step: PlanStep,
) -> dict[str, tuple[str, ...]]:
    if step.agent_id != "synthesis":
        return {}

    alternates: dict[str, tuple[str, ...]] = {}
    for candidate in plan.steps:
        if candidate.agent_id != "data_quality" or len(candidate.depends_on) != 1:
            continue
        if candidate.step_id in step.depends_on:
            alternates[candidate.step_id] = tuple(candidate.depends_on)
    return alternates


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


def _freeze_static_runtime_context(
    runtime_context: Mapping[str, Any],
    approved_context: Mapping[str, Any],
) -> Any:
    static_context = dict(runtime_context)
    if (
        "upstream" not in approved_context
        and _approved_context_has_dependencies(approved_context)
    ):
        static_context.pop("upstream", None)

    return _freeze_approval_value(static_context)


def _approved_context_has_dependencies(context: Mapping[str, Any]) -> bool:
    depends_on = context.get("dependsOn", ())
    if isinstance(depends_on, list | tuple | set | frozenset):
        return len(depends_on) > 0
    return bool(depends_on)


async def call_specialist_with_guard(
    context: RequestContext,
    request: SpecialistRequest,
    specialist: SpecialistLike,
    *,
    enforce_response_agent_id: bool = True,
) -> SpecialistResponse:
    """Apply request guardrails before invoking a specialist."""

    context.require_specialist_call_allowed(request)
    response = await invoke_specialist(
        specialist,
        request,
        enforce_response_agent_id=enforce_response_agent_id,
    )
    context.record_specialist_surface_owner(
        surface_id=response.surface_id,
        agent_id=request.agent_id,
    )
    return response


__all__ = [
    "PlanApprovalStateError",
    "RequestContext",
    "SpecialistCallable",
    "SpecialistPreApprovalError",
    "call_specialist_with_guard",
]
