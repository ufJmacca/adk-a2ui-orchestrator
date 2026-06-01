"""Structured approval state for editable execution plans."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.a2ui_support.approval_canvas import approval_canvas_data_parts
from orchestrator_demo.a2ui_support.event_parser import parse_plan_user_action
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep, UserAction
from orchestrator_demo.orchestrator.planner import _step_metadata


PlanState = Literal["draft", "approved", "rejected"]
ApprovalActionStatus = Literal["ignored", "draft_updated", "approved", "rejected"]
CONDITIONAL_DATA_QUALITY_AGENT_ID = "data_quality"
CONDITIONAL_DATA_QUALITY_ROUTE = "missing_internal_data"


class ApprovalStateError(ValueError):
    """Base error for invalid approval-state transitions."""


class PlanNotFoundError(ApprovalStateError):
    """Raised when a userAction references an unknown draft plan."""


class PlanAlreadyFinalError(ApprovalStateError):
    """Raised when a mutation targets an approved or rejected plan."""


class PlanVersionConflictError(ApprovalStateError):
    """Raised when a userAction references a stale draft plan version."""


class PlanSurfaceMismatchError(ApprovalStateError):
    """Raised when a userAction targets the wrong approval surface."""


class PlanMutationError(ApprovalStateError):
    """Raised when a draft mutation payload is invalid."""


@dataclass
class ApprovalRecord:
    """Stored approval state for one plan."""

    draft_plan: ExecutionPlan
    status: PlanState = "draft"
    approved_version: int | None = None
    rejection_reason: str | None = None
    _approved_plan: ExecutionPlan | None = field(default=None, repr=False)

    @property
    def approved_plan(self) -> ExecutionPlan | None:
        """Return a copy so callers cannot mutate frozen approved state."""

        if self._approved_plan is None:
            return None
        return self._approved_plan.model_copy(deep=True)

    @approved_plan.setter
    def approved_plan(self, plan: ExecutionPlan | None) -> None:
        self._approved_plan = None if plan is None else plan.model_copy(deep=True)


def _record_snapshot(record: ApprovalRecord) -> ApprovalRecord:
    snapshot = ApprovalRecord(
        draft_plan=record.draft_plan.model_copy(deep=True),
        status=record.status,
        approved_version=record.approved_version,
        rejection_reason=record.rejection_reason,
    )
    snapshot.approved_plan = record.approved_plan
    return snapshot


@dataclass(frozen=True)
class ApprovalActionResult:
    """Result of applying a structured approval event."""

    status: ApprovalActionStatus
    plan_id: str | None = None
    plan_version: int | None = None
    refreshed_a2ui_parts: list[DataPart] = field(default_factory=list)
    draft_plan: ExecutionPlan | None = None
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
        existing = self._records.get(draft.plan_id)
        if existing is not None and existing.status != "draft":
            raise PlanAlreadyFinalError(
                f"plan {draft.plan_id} is already {existing.status}"
            )

        record = ApprovalRecord(draft_plan=draft)
        self._records[draft.plan_id] = record
        return _record_snapshot(record)

    def get(self, plan_id: str) -> ApprovalRecord:
        """Return a defensive snapshot of a plan record for inspection."""

        return _record_snapshot(self._get_live_record(plan_id))

    def _get_live_record(self, plan_id: str) -> ApprovalRecord:
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
        record = self._get_live_record(action.plan_id)
        self._require_matching_surface(record, action)

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
        _require_approved_step_ids(record.draft_plan, action.payload)

        frozen_plan = record.draft_plan.model_copy(deep=True)
        record.status = "approved"
        record.approved_plan = frozen_plan
        record.approved_version = frozen_plan.plan_version

        return ApprovalActionResult(
            status="approved",
            plan_id=frozen_plan.plan_id,
            plan_version=frozen_plan.plan_version,
            approved_plan=frozen_plan.model_copy(deep=True),
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

        reason = _optional_string(action.payload, "reason", empty_as_none=True)
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
        elif action.type in {"choose_agent", "replace_agent"}:
            next_plan = self._replace_agent(plan, action.payload)
        elif action.type in {"add_instruction", "add_instructions"}:
            next_plan = _add_instruction(plan, action.payload)
        else:
            raise PlanMutationError(f"unsupported draft mutation: {action.type}")

        candidate_plan = _with_next_version(next_plan)
        _require_final_synthesis_preserved(plan, candidate_plan)
        refreshed_parts = approval_canvas_data_parts(
            candidate_plan,
            agent_descriptors=self._agent_descriptors,
        )
        record.draft_plan = candidate_plan
        return ApprovalActionResult(
            status="draft_updated",
            plan_id=candidate_plan.plan_id,
            plan_version=candidate_plan.plan_version,
            refreshed_a2ui_parts=refreshed_parts,
            draft_plan=candidate_plan.model_copy(deep=True),
            graph_created=False,
            specialists_called=False,
        )

    def _replace_agent(
        self,
        plan: ExecutionPlan,
        payload: Mapping[str, Any],
    ) -> ExecutionPlan:
        step_id = _required_string(payload, "stepId", "step_id")
        replacement_agent_id = _required_string(
            payload,
            "replacementAgentId",
            "replacement_agent_id",
            "selectedAgentId",
            "selected_agent_id",
            "agentId",
            "agent_id",
        )
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
            _require_replaceable_step(step)
            replacement_metadata = _step_metadata(
                replacement_agent_id,
                plan.objective,
            )
            steps.append(
                _step_copy(
                    step,
                    agent_id=replacement_agent_id,
                    instruction=replacement_metadata.instruction,
                    expected_output=replacement_metadata.expected_output,
                    data_source_categories=(
                        replacement_metadata.data_source_categories
                    ),
                )
            )

        if not replaced:
            raise PlanMutationError(f"unknown stepId: {step_id}")
        if not _has_non_synthesis_step(steps):
            raise PlanMutationError(
                "replace_agent must leave at least one non-synthesis specialist step"
            )

        return _plan_copy(
            plan,
            steps=steps,
            selected_agents=_selected_agents(steps),
            data_source_categories=_data_source_categories(steps),
        )

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

    def _require_matching_surface(
        self,
        record: ApprovalRecord,
        action: UserAction,
    ) -> None:
        expected_surface_id = _approval_surface_id(record.draft_plan)
        if action.surface_id == expected_surface_id:
            return
        raise PlanSurfaceMismatchError(
            f"plan {record.draft_plan.plan_id} belongs to approval surface "
            f"{expected_surface_id!r}, got {action.surface_id!r}"
        )


def _approval_surface_id(plan: ExecutionPlan) -> str:
    return plan.approval_surface_id or f"surface_{plan.plan_id}"


def _require_approved_step_ids(
    plan: ExecutionPlan,
    payload: Mapping[str, Any],
) -> None:
    approved_step_ids = payload.get("approvedStepIds")
    if not isinstance(approved_step_ids, list) or not all(
        isinstance(step_id, str) for step_id in approved_step_ids
    ):
        raise PlanMutationError("approve_plan requires approvedStepIds")

    current_step_ids = [step.step_id for step in plan.steps]
    if sorted(approved_step_ids) != sorted(current_step_ids) or len(
        set(approved_step_ids)
    ) != len(approved_step_ids):
        raise PlanMutationError(
            "approvedStepIds must match current draft plan steps"
        )


def _edit_plan(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    objective = _optional_string(payload, "objective")
    if objective is None:
        if _is_generated_edit_control_payload(payload):
            return plan.model_copy(deep=True)
        raise PlanMutationError("edit_plan requires at least an objective edit")
    if objective != plan.objective:
        raise PlanMutationError("edit_plan cannot change the routed objective")
    return plan.model_copy(deep=True)


def _is_generated_edit_control_payload(payload: Mapping[str, Any]) -> bool:
    editable_fields = payload.get("editableFields")
    if not isinstance(editable_fields, list):
        return False
    return all(isinstance(field, str) for field in editable_fields)


def _remove_step(plan: ExecutionPlan, payload: Mapping[str, Any]) -> ExecutionPlan:
    step_id = _required_string(payload, "stepId")
    steps_by_id = {step.step_id: step for step in plan.steps}
    removed_step = steps_by_id.get(step_id)
    if removed_step is None:
        raise PlanMutationError(f"unknown stepId: {step_id}")
    _require_removal_preserves_conditional_sources(plan.steps, removed_step)

    steps = [
        _step_copy(
            step,
            depends_on=_dependencies_after_step_removal(step, removed_step),
        )
        for step in plan.steps
        if step.step_id != step_id
    ]
    if not steps:
        raise PlanMutationError("remove_step cannot remove every plan step")
    if not _has_non_synthesis_step(steps):
        raise PlanMutationError(
            "remove_step must leave at least one non-synthesis specialist step"
        )

    return _plan_copy(
        plan,
        steps=steps,
        selected_agents=_selected_agents(steps),
        data_source_categories=_data_source_categories(steps),
    )


def _require_removal_preserves_conditional_sources(
    steps: Sequence[PlanStep],
    removed_step: PlanStep,
) -> None:
    if (
        removed_step.agent_id == CONDITIONAL_DATA_QUALITY_AGENT_ID
        and removed_step.condition == CONDITIONAL_DATA_QUALITY_ROUTE
    ):
        raise PlanMutationError(
            "remove_step cannot remove a conditional data_quality step"
        )

    for step in steps:
        if step.step_id == removed_step.step_id or step.condition is None:
            continue
        if removed_step.step_id in step.depends_on:
            raise PlanMutationError(
                "remove_step cannot remove the source dependency for "
                f"conditional step {step.step_id}"
            )


def _dependencies_after_step_removal(
    step: PlanStep,
    removed_step: PlanStep,
) -> list[str]:
    dependencies: list[str] = []
    for dependency in step.depends_on:
        if dependency == removed_step.step_id:
            for upstream_dependency in removed_step.depends_on:
                if upstream_dependency not in dependencies:
                    dependencies.append(upstream_dependency)
            continue
        if dependency not in dependencies:
            dependencies.append(dependency)
    return dependencies


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
    step_id = _required_string(payload, "stepId", "step_id")
    instruction = _required_string(payload, "instruction", "instructions")

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


def _has_non_synthesis_step(steps: Sequence[PlanStep]) -> bool:
    return any(step.agent_id != "synthesis" for step in steps)


def _require_replaceable_step(step: PlanStep) -> None:
    if (
        step.agent_id == CONDITIONAL_DATA_QUALITY_AGENT_ID
        and step.condition == CONDITIONAL_DATA_QUALITY_ROUTE
    ):
        raise PlanMutationError(
            "replace_agent cannot replace a conditional data_quality step"
        )


def _require_final_synthesis_preserved(
    previous_plan: ExecutionPlan,
    candidate_plan: ExecutionPlan,
) -> None:
    if not _requires_final_synthesis(previous_plan):
        return

    synthesis_step_count = sum(
        1 for step in candidate_plan.steps if step.agent_id == "synthesis"
    )
    if synthesis_step_count != 1 or candidate_plan.steps[-1].agent_id != "synthesis":
        raise PlanMutationError(
            "complex multi-agent plans must keep exactly one final synthesis step"
        )


def _requires_final_synthesis(plan: ExecutionPlan) -> bool:
    return any(step.agent_id == "synthesis" for step in plan.steps)


def _required_string(payload: Mapping[str, Any], *field_names: str) -> str:
    field_name = " or ".join(field_names)
    value = next(
        (payload[name] for name in field_names if payload.get(name) is not None),
        None,
    )
    value = _resolve_path_bound_value(payload, value, field_name)
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        value = "\n".join(item.strip() for item in value if item.strip())
    if not isinstance(value, str) or not value.strip():
        raise PlanMutationError(f"{field_name} must be a non-empty string")
    return value


def _resolve_path_bound_value(
    payload: Mapping[str, Any],
    value: Any,
    field_name: str,
) -> Any:
    if not _is_path_binding(value):
        return value

    resolved = _value_at_path(payload, value["path"])
    if resolved is _MISSING:
        raise PlanMutationError(f"{field_name} path did not resolve")
    return resolved


def _is_path_binding(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(value) == {"path"}
        and isinstance(value.get("path"), str)
        and value["path"].startswith("/")
    )


_MISSING = object()


def _value_at_path(payload: Mapping[str, Any], path: str) -> Any:
    current: Any = payload
    for raw_segment in path.split("/")[1:]:
        segment = raw_segment.replace("~1", "/").replace("~0", "~")
        if isinstance(current, Mapping):
            current = current.get(segment, _MISSING)
        elif isinstance(current, Sequence) and not isinstance(current, str | bytes):
            try:
                current = current[int(segment)]
            except (IndexError, ValueError):
                return _MISSING
        else:
            return _MISSING

        if current is _MISSING:
            return _MISSING

    return current


def _optional_string(
    payload: Mapping[str, Any],
    field_name: str,
    *,
    empty_as_none: bool = False,
) -> str | None:
    value = payload.get(field_name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise PlanMutationError(f"{field_name} must be a non-empty string")
    stripped = value.strip()
    if not stripped:
        if empty_as_none:
            return None
        raise PlanMutationError(f"{field_name} must be a non-empty string")
    return stripped


__all__ = [
    "ApprovalActionResult",
    "ApprovalRecord",
    "ApprovalStateError",
    "ApprovalStateStore",
    "PlanAlreadyFinalError",
    "PlanMutationError",
    "PlanNotFoundError",
    "PlanState",
    "PlanSurfaceMismatchError",
    "PlanVersionConflictError",
]
