"""Structured approval state for editable execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.approval_canvas import approval_canvas_data_parts
from orchestrator_demo.a2ui_support.event_parser import parse_plan_user_action
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep, UserAction


PlanState = Literal["draft", "approved", "rejected"]
ApprovalActionStatus = Literal["ignored", "draft_updated", "approved", "rejected"]


class ApprovalStateError(ValueError):
    """Base error for invalid approval-state transitions."""


class PlanNotFoundError(ApprovalStateError):
    """Raised when a userAction references an unknown draft plan."""


class PlanAlreadyFinalError(ApprovalStateError):
    """Raised when a mutation targets an approved or rejected plan."""


class PlanVersionConflictError(ApprovalStateError):
    """Raised when a userAction references a stale draft plan version."""


class PlanMutationError(ApprovalStateError):
    """Raised when a draft mutation payload is invalid."""


@dataclass
class ApprovalRecord:
    """Stored approval state for one plan."""

    draft_plan: ExecutionPlan
    status: PlanState = "draft"
    approved_plan: ExecutionPlan | None = None
    approved_version: int | None = None
    rejection_reason: str | None = None


@dataclass(frozen=True)
class ApprovalActionResult:
    """Result of applying a structured approval event."""

    status: ApprovalActionStatus
    plan_id: str | None = None
    plan_version: int | None = None
    refreshed_a2ui_parts: list[DataPart] = field(default_factory=list)
    approved_plan: ExecutionPlan | None = None
    rejection_reason: str | None = None
    graph_created: bool = False
    specialists_called: bool = False

    @property
    def refreshed_a2ui_part(self) -> DataPart | None:
        """Return the latest component update for callers that need one part."""

        if not self.refreshed_a2ui_parts:
            return None
        return self.refreshed_a2ui_parts[-1]


class ApprovalStateStore:
    """Manage draft, approved, and rejected plan state."""

    def __init__(self, *, agent_descriptors: Sequence[AgentDescriptor]) -> None:
        self._agent_descriptors = list(agent_descriptors)
        self._agent_ids = {
            descriptor.agent_id for descriptor in self._agent_descriptors
        }
        self._records: dict[str, ApprovalRecord] = {}

    def add_draft(self, plan: ExecutionPlan) -> ApprovalRecord:
        """Store a deep copy of a draft plan without owning the caller's object."""

        draft = plan.model_copy(deep=True)
        record = ApprovalRecord(draft_plan=draft)
        self._records[draft.plan_id] = record
        return record

    def get(self, plan_id: str) -> ApprovalRecord:
        try:
            return self._records[plan_id]
        except KeyError:
            raise PlanNotFoundError(f"unknown plan: {plan_id}") from None

    def handle_natural_language(self, _message: str) -> ApprovalActionResult:
        """Ignore conversational approvals; only A2UI userAction events count."""

        return ApprovalActionResult(status="ignored")

    def apply_user_action(self, candidate: Any) -> ApprovalActionResult:
        """Apply a supported structured plan userAction to stored state."""

        action = parse_plan_user_action(candidate)
        assert action.plan_id is not None
        record = self.get(action.plan_id)

        if action.type == "approve_plan":
            return self._approve(record, action)
        if action.type == "reject_plan":
            return self._reject(record, action)

        return self._mutate_draft(record, action)

    def _approve(
        self,
        record: ApprovalRecord,
        action: UserAction,
    ) -> ApprovalActionResult:
        self._require_draft(record)
        self._require_current_version(record, action)

        frozen_plan = record.draft_plan.model_copy(deep=True)
        record.status = "approved"
        record.approved_plan = frozen_plan
        record.approved_version = frozen_plan.plan_version

        return ApprovalActionResult(
            status="approved",
            plan_id=frozen_plan.plan_id,
            plan_version=frozen_plan.plan_version,
            approved_plan=frozen_plan,
            graph_created=False,
            specialists_called=False,
        )

    def _reject(
        self,
        record: ApprovalRecord,
        action: UserAction,
    ) -> ApprovalActionResult:
        self._require_draft(record)
        if action.plan_version is not None:
            self._require_current_version(record, action)

        reason = _optional_string(action.payload, "reason")
        record.status = "rejected"
        record.rejection_reason = reason

        return ApprovalActionResult(
            status="rejected",
            plan_id=record.draft_plan.plan_id,
            plan_version=record.draft_plan.plan_version,
            rejection_reason=reason,
            graph_created=False,
            specialists_called=False,
        )

    def _mutate_draft(
        self,
        record: ApprovalRecord,
        action: UserAction,
    ) -> ApprovalActionResult:
        self._require_draft(record)
        self._require_current_version(record, action)

        plan = record.draft_plan
        if action.type == "edit_plan":
            next_plan = _edit_plan(plan, action.payload)
        elif action.type == "remove_step":
            next_plan = _remove_step(plan, action.payload)
        elif action.type == "reorder_steps":
            next_plan = _reorder_steps(plan, action.payload)
        elif action.type == "replace_agent":
            next_plan = self._replace_agent(plan, action.payload)
        elif action.type == "add_instruction":
            next_plan = _add_instruction(plan, action.payload)
        else:
            raise PlanMutationError(f"unsupported draft mutation: {action.type}")

        record.draft_plan = _with_next_version(next_plan)
        refreshed_parts = approval_canvas_data_parts(
            record.draft_plan,
            agent_descriptors=self._agent_descriptors,
        )
        return ApprovalActionResult(
            status="draft_updated",
            plan_id=record.draft_plan.plan_id,
            plan_version=record.draft_plan.plan_version,
            refreshed_a2ui_parts=refreshed_parts,
            graph_created=False,
            specialists_called=False,
        )

    def _replace_agent(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, Any],
    ) -> ExecutionPlan:
        step_id = _required_string(payload, "stepId")
        replacement_agent_id = _required_string(payload, "replacementAgentId")
        if replacement_agent_id not in self._agent_ids:
            raise PlanMutationError(
                f"replacement agent is unavailable: {replacement_agent_id}"
            )

        replaced = False
        steps: list[PlanStep] = []
        for step in plan.steps:
            if step.step_id != step_id:
                steps.append(step)
                continue
            replaced = True
            steps.append(_step_copy(step, agent_id=replacement_agent_id))

        if not replaced:
            raise PlanMutationError(f"unknown stepId: {step_id}")

        return _plan_copy(plan, steps=steps, selected_agents=_selected_agents(steps))

    def _require_draft(self, record: ApprovalRecord) -> None:
        if record.status == "draft":
            return
        raise PlanAlreadyFinalError(
            f"plan {record.draft_plan.plan_id} is already {record.status}"
        )

    def _require_current_version(
        self,
        record: ApprovalRecord,
        action: UserAction,
    ) -> None:
        if action.plan_version == record.draft_plan.plan_version:
            return
        raise PlanVersionConflictError(
            f"plan {record.draft_plan.plan_id} is version "
            f"{record.draft_plan.plan_version}, got {action.plan_version}"
        )


def _edit_plan(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    objective = _optional_string(payload, "objective")
    if objective is None:
        raise PlanMutationError("edit_plan requires at least an objective edit")
    return _plan_copy(plan, objective=objective)


def _remove_step(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    step_id = _required_string(payload, "stepId")
    if step_id not in {step.step_id for step in plan.steps}:
        raise PlanMutationError(f"unknown stepId: {step_id}")

    steps = [
        _step_copy(
            step,
            depends_on=[
                dependency for dependency in step.depends_on if dependency != step_id
            ],
        )
        for step in plan.steps
        if step.step_id != step_id
    ]
    if not steps:
        raise PlanMutationError("remove_step cannot remove every plan step")

    return _plan_copy(
        plan,
        steps=steps,
        selected_agents=_selected_agents(steps),
        data_source_categories=_data_source_categories(steps),
    )


def _reorder_steps(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    ordered_step_ids = payload.get("orderedStepIds")
    if not isinstance(ordered_step_ids, list) or not all(
        isinstance(step_id, str) for step_id in ordered_step_ids
    ):
        raise PlanMutationError("reorder_steps requires orderedStepIds")

    current_step_ids = [step.step_id for step in plan.steps]
    if sorted(ordered_step_ids) != sorted(current_step_ids) or len(
        set(ordered_step_ids)
    ) != len(ordered_step_ids):
        raise PlanMutationError("orderedStepIds must match current plan steps")

    steps_by_id = {step.step_id: step for step in plan.steps}
    return _plan_copy(plan, steps=[steps_by_id[step_id] for step_id in ordered_step_ids])


def _add_instruction(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    step_id = _required_string(payload, "stepId")
    instruction = _required_string(payload, "instruction")

    updated = False
    steps: list[PlanStep] = []
    for step in plan.steps:
        if step.step_id != step_id:
            steps.append(step)
            continue
        updated = True
        steps.append(
            _step_copy(
                step,
                instruction=f"{step.instruction}\nAdditional instruction: {instruction}",
            )
        )

    if not updated:
        raise PlanMutationError(f"unknown stepId: {step_id}")

    return _plan_copy(plan, steps=steps)


def _with_next_version(plan: ExecutionPlan) -> ExecutionPlan:
    return _plan_copy(plan, plan_version=plan.plan_version + 1)


def _plan_copy(plan: ExecutionPlan, **updates: Any) -> ExecutionPlan:
    payload = plan.model_dump()
    payload.update(updates)
    return ExecutionPlan.model_validate(payload)


def _step_copy(step: PlanStep, **updates: Any) -> PlanStep:
    payload = step.model_dump()
    payload.update(updates)
    return PlanStep.model_validate(payload)


def _selected_agents(steps: Sequence[PlanStep]) -> list[str]:
    selected: list[str] = []
    for step in steps:
        if step.agent_id not in selected:
            selected.append(step.agent_id)
    return selected


def _data_source_categories(steps: Sequence[PlanStep]) -> list[str]:
    categories: list[str] = []
    for step in steps:
        if step.agent_id == "synthesis":
            continue
        for category in step.data_source_categories:
            if category not in categories:
                categories.append(category)
    return categories


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise PlanMutationError(f"{field_name} must be a non-empty string")
    return value


def _optional_string(payload: Mapping[str, Any], field_name: str) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PlanMutationError(f"{field_name} must be a non-empty string")
    return value


__all__ = [
    "ApprovalActionResult",
    "ApprovalRecord",
    "ApprovalStateError",
    "ApprovalStateStore",
    "PlanAlreadyFinalError",
    "PlanMutationError",
    "PlanNotFoundError",
    "PlanState",
    "PlanVersionConflictError",
]
