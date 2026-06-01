"""Draft execution planning for complex orchestrator routes."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep
from orchestrator_demo.orchestrator.request_context import RequestContext


SYNTHESIS_AGENT_ID = "synthesis"


class PlanRequiredError(ValueError):
    """Raised when draft planning is requested for a non-plan route."""


class PlanCreationError(PlanRequiredError):
    """Raised when a safe draft plan cannot be formed."""


class DescriptorRegistry(Protocol):
    """Registry surface needed by the planner."""

    def descriptors(self) -> list[AgentDescriptor]:
        """Return currently available specialist descriptors."""
        ...


class DraftExecutionPlanner:
    """Create immutable draft plans from plan-required routing context."""

    def __init__(self, *, registry: DescriptorRegistry) -> None:
        self._registry = registry

    def create_plan(self, context: RequestContext) -> ExecutionPlan:
        """Create a draft plan using only agents available in the registry."""

        if context.decision.path != "plan_required":
            raise PlanRequiredError(
                "draft execution plans can only be created for plan_required routes"
            )

        available_agent_ids = {
            descriptor.agent_id for descriptor in self._registry.descriptors()
        }
        requested_agent_ids = _dedupe(context.llm_assessment.required_agents)
        selected_agent_ids = [
            agent_id
            for agent_id in requested_agent_ids
            if agent_id in available_agent_ids
        ]
        omitted_agent_ids = [
            agent_id
            for agent_id in requested_agent_ids
            if agent_id not in available_agent_ids
        ]

        requires_synthesis = _requires_synthesis(
            context,
            selected_agent_ids=selected_agent_ids,
            requested_agent_ids=requested_agent_ids,
        )
        if requires_synthesis and SYNTHESIS_AGENT_ID not in available_agent_ids:
            raise PlanCreationError(
                "draft execution plan requires unavailable synthesis agent; "
                "requires synthesis but the synthesis agent is unavailable"
            )
        if requires_synthesis and SYNTHESIS_AGENT_ID not in selected_agent_ids:
            selected_agent_ids.append(SYNTHESIS_AGENT_ID)
        selected_agent_ids = _move_synthesis_to_end(selected_agent_ids)
        if not _has_non_synthesis_workstream(selected_agent_ids):
            unavailable_detail = (
                f"; requested agents are unavailable: {', '.join(omitted_agent_ids)}"
                if omitted_agent_ids
                else ""
            )
            raise PlanCreationError(
                "draft execution plan cannot be formed because no available "
                "non-synthesis specialist workstream remains after registry "
                f"filtering{unavailable_detail}"
            )

        plan_id = _plan_id_for(context)
        steps = _build_steps(
            objective=context.user_input,
            selected_agent_ids=selected_agent_ids,
            primary_parallel_group=_parallel_group_for(context),
        )
        risk_notes = ["Synthetic demo data only"]
        if omitted_agent_ids:
            risk_notes.append(
                f"Unavailable agents omitted: {', '.join(omitted_agent_ids)}."
            )

        plan = ExecutionPlan(
            plan_id=plan_id,
            objective=context.user_input,
            detected_intents=context.llm_assessment.intents,
            selected_agents=selected_agent_ids,
            steps=steps,
            data_source_categories=_plan_data_source_categories(steps),
            risk_notes=risk_notes,
            approval_surface_id=f"surface_{plan_id}",
        )
        context.record_draft_plan(plan)
        return plan


def _build_steps(
    *,
    objective: str,
    selected_agent_ids: Sequence[str],
    primary_parallel_group: str,
) -> list[PlanStep]:
    step_ids_by_agent_id = _step_ids_by_agent_id(selected_agent_ids)
    non_synthesis_step_ids = [
        step_id
        for agent_id, step_id in step_ids_by_agent_id.items()
        if agent_id != SYNTHESIS_AGENT_ID
    ]
    steps: list[PlanStep] = []

    for agent_id in selected_agent_ids:
        metadata = _step_metadata(agent_id, objective)
        is_synthesis = agent_id == SYNTHESIS_AGENT_ID
        depends_on = non_synthesis_step_ids if is_synthesis else []
        parallel_group = None if is_synthesis else primary_parallel_group

        steps.append(
            PlanStep(
                step_id=step_ids_by_agent_id[agent_id],
                agent_id=agent_id,
                instruction=metadata.instruction,
                depends_on=depends_on,
                expected_output=metadata.expected_output,
                data_source_categories=metadata.data_source_categories,
                parallel_group=parallel_group,
            )
        )

    return steps


def _step_ids_by_agent_id(selected_agent_ids: Sequence[str]) -> dict[str, str]:
    step_ids_by_agent_id: dict[str, str] = {}
    used_step_ids: set[str] = set()

    for agent_id in selected_agent_ids:
        base_step_id = _step_id_for(agent_id)
        step_id = base_step_id
        suffix = 2
        while step_id in used_step_ids:
            step_id = f"{base_step_id}_{suffix}"
            suffix += 1

        step_ids_by_agent_id[agent_id] = step_id
        used_step_ids.add(step_id)

    return step_ids_by_agent_id


class _StepMetadata:
    def __init__(
        self,
        *,
        instruction: str,
        expected_output: str,
        data_source_categories: list[str],
    ) -> None:
        self.instruction = instruction
        self.expected_output = expected_output
        self.data_source_categories = data_source_categories


def _step_metadata(agent_id: str, objective: str) -> _StepMetadata:
    templates = {
        "relationship_summary": _StepMetadata(
            instruction=(
                "Summarize relationship history, key contacts, prior meetings, "
                f"and open follow-ups for: {objective}"
            ),
            expected_output="Relationship history, contacts, prior meetings, and follow-ups.",
            data_source_categories=["relationship_history"],
        ),
        "internal_knowledge": _StepMetadata(
            instruction=(
                "Review internal CRM notes, policy snippets, relationship records, "
                f"and open items for: {objective}"
            ),
            expected_output="Internal notes, policy snippets, relationship records, and open items.",
            data_source_categories=["internal_crm"],
        ),
        "industry_research": _StepMetadata(
            instruction=(
                "Provide sector context, market drivers, risks, and opportunities "
                f"relevant to: {objective}"
            ),
            expected_output="Industry overview with market drivers, risks, and opportunities.",
            data_source_categories=["industry_research"],
        ),
        "web_search": _StepMetadata(
            instruction=(
                "Gather and summarize public information, recent events, and "
                f"source-backed signals for: {objective}"
            ),
            expected_output="Public research summary with source and recency notes.",
            data_source_categories=["public_web"],
        ),
        "product_opportunity": _StepMetadata(
            instruction=(
                "Identify deposit, lending, treasury, card, merchant services, "
                f"and other product opportunities for: {objective}"
            ),
            expected_output="Prioritized product opportunities and supporting rationale.",
            data_source_categories=["product_fit"],
        ),
        "credit_risk": _StepMetadata(
            instruction=(
                "Flag credit risk themes, missing credit context, covenant concerns, "
                f"and repayment indicators for: {objective}"
            ),
            expected_output="Credit risk themes, missing data, and caveats.",
            data_source_categories=["credit_risk"],
        ),
        "compliance_policy": _StepMetadata(
            instruction=(
                "Check for regulated-output, policy, and unsupported-claim risks "
                f"before RM use: {objective}"
            ),
            expected_output="Policy caveats and regulated-output guardrails.",
            data_source_categories=["compliance_policy"],
        ),
        "data_quality": _StepMetadata(
            instruction=(
                "Highlight missing information, stale context, weak evidence, "
                f"and confidence gaps for: {objective}"
            ),
            expected_output="Data quality gaps and clarification needs.",
            data_source_categories=["data_quality"],
        ),
        "meeting_prep": _StepMetadata(
            instruction=(
                "Draft meeting objectives, talking points, and follow-up questions "
                f"for: {objective}"
            ),
            expected_output="Meeting objectives, talking points, and follow-up questions.",
            data_source_categories=["meeting_preparation"],
        ),
        SYNTHESIS_AGENT_ID: _StepMetadata(
            instruction=(
                "Combine completed specialist outputs into an RM-ready answer for: "
                f"{objective}"
            ),
            expected_output="Final synthesized briefing with caveats and next actions.",
            data_source_categories=["specialist_outputs"],
        ),
    }

    return templates.get(
        agent_id,
        _StepMetadata(
            instruction=(
                f"Complete the {agent_id.replace('_', ' ')} workstream for: "
                f"{objective}"
            ),
            expected_output=f"{agent_id.replace('_', ' ').title()} findings.",
            data_source_categories=[agent_id],
        ),
    )


def _plan_data_source_categories(steps: Sequence[PlanStep]) -> list[str]:
    categories: list[str] = []
    for step in steps:
        if step.agent_id == SYNTHESIS_AGENT_ID:
            continue
        for category in step.data_source_categories:
            if category not in categories:
                categories.append(category)

    return categories


def _requires_synthesis(
    context: RequestContext,
    *,
    selected_agent_ids: Sequence[str],
    requested_agent_ids: Sequence[str],
) -> bool:
    selected_workstreams = [
        agent_id for agent_id in selected_agent_ids if agent_id != SYNTHESIS_AGENT_ID
    ]
    return (
        context.llm_assessment.complexity == "complex"
        or len(selected_workstreams) > 1
        or SYNTHESIS_AGENT_ID in requested_agent_ids
    )


def _parallel_group_for(context: RequestContext) -> str:
    intents = set(context.llm_assessment.intents)
    if "prospect_research" in intents:
        return "parallel_prospect_research"
    if "meeting_prep" in intents:
        return "parallel_meeting_prep_context"
    if {"credit_risk", "compliance_policy"} <= intents:
        return "parallel_risk_policy_review"

    return "parallel_research"


def _plan_id_for(context: RequestContext) -> str:
    plan_id_candidates = [
        *context.llm_assessment.intents,
        *context.llm_assessment.required_agents,
    ]
    for candidate in plan_id_candidates:
        if candidate != "unknown":
            return f"plan_{_slug(candidate)}_{_slug(context.plan_scope_id)}"

    return f"plan_data_quality_{_slug(context.plan_scope_id)}"


def _step_id_for(agent_id: str) -> str:
    return f"step_{_slug(agent_id)}"


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip().lower())
    slug = slug.strip("_-")
    return slug or "unknown"


def _dedupe(values: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)

    return deduped


def _move_synthesis_to_end(agent_ids: Sequence[str]) -> list[str]:
    ordered_agent_ids = [
        agent_id for agent_id in agent_ids if agent_id != SYNTHESIS_AGENT_ID
    ]
    if SYNTHESIS_AGENT_ID in agent_ids:
        ordered_agent_ids.append(SYNTHESIS_AGENT_ID)

    return ordered_agent_ids


def _has_non_synthesis_workstream(agent_ids: Sequence[str]) -> bool:
    return any(agent_id != SYNTHESIS_AGENT_ID for agent_id in agent_ids)


__all__ = [
    "DraftExecutionPlanner",
    "PlanCreationError",
    "PlanRequiredError",
    "SYNTHESIS_AGENT_ID",
]
