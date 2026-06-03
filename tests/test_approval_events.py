import threading
from types import SimpleNamespace

import pytest

from orchestrator_demo.a2a_support.transport import DataPart
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep


def _descriptor(agent_id: str, display_name: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=display_name,
        capabilities=[agent_id.replace("_", " ")],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[],
        execution_mode="local_llm",
    )


def _agent_descriptors() -> list[AgentDescriptor]:
    return [
        _descriptor("relationship_summary", "Relationship Summary Agent"),
        _descriptor("internal_knowledge", "Internal Knowledge Agent"),
        _descriptor("industry_research", "Industry Research Agent"),
        _descriptor("credit_risk", "Credit Risk Agent"),
        _descriptor("synthesis", "Synthesis Agent"),
    ]


def _meeting_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=[
            "meeting_prep",
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
        ],
        selected_agents=[
            "relationship_summary",
            "internal_knowledge",
            "industry_research",
            "synthesis",
        ],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize the relationship history.",
                expected_output="Relationship history and open follow-ups.",
                data_source_categories=["relationship_history"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_industry_research",
                agent_id="industry_research",
                instruction="Gather industry context.",
                expected_output="Industry risks and opportunities.",
                data_source_categories=["industry_research"],
                parallel_group="parallel_meeting_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Create the RM-ready meeting brief.",
                depends_on=[
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                ],
                expected_output="Final meeting preparation brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=[
            "relationship_history",
            "internal_crm",
            "industry_research",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_meeting_prep",
    )


def _dependent_chain_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=[
            "meeting_prep",
            "relationship_summary",
            "internal_knowledge",
            "credit_risk",
        ],
        selected_agents=[
            "relationship_summary",
            "internal_knowledge",
            "credit_risk",
        ],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize the relationship history.",
                expected_output="Relationship history and open follow-ups.",
                data_source_categories=["relationship_history"],
            ),
            PlanStep(
                step_id="step_internal_knowledge",
                agent_id="internal_knowledge",
                instruction="Review internal CRM notes.",
                depends_on=["step_relationship_summary"],
                expected_output="Internal customer context.",
                data_source_categories=["internal_crm"],
            ),
            PlanStep(
                step_id="step_credit_risk",
                agent_id="credit_risk",
                instruction="Assess credit risk signals.",
                depends_on=["step_internal_knowledge"],
                expected_output="Credit risk considerations.",
                data_source_categories=["credit_risk"],
            ),
        ],
        data_source_categories=[
            "relationship_history",
            "internal_crm",
            "credit_risk",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_meeting_prep",
    )


def _action(action_type: str, payload: dict[str, object]) -> dict[str, object]:
    return {
        "userAction": {
            "type": action_type,
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 1,
                **payload,
            },
        }
    }


def _renderer_control_event(
    action_type: str,
    payload: dict[str, object],
    *,
    plan_version: int = 1,
) -> dict[str, object]:
    return {
        "event": {
            "name": action_type,
            "context": {
                "type": action_type,
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "planVersion": plan_version,
                    "editedPlanVersion": plan_version,
                    **payload,
                },
            },
        }
    }


def _store_with_meeting_plan():
    from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore

    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    original_plan = _meeting_plan()
    store.add_draft(original_plan)
    return store, original_plan


def _plan_metadata_text(part: DataPart) -> str:
    components = part.data["updateComponents"]["components"]
    metadata = next(
        component
        for component in components
        if component["id"] == "component_plan_meeting_prep_metadata"
    )
    return metadata["text"]


def _objective_text(part: DataPart) -> str:
    components = part.data["updateComponents"]["components"]
    objective = next(
        component
        for component in components
        if component["id"] == "component_plan_meeting_prep_objective"
    )
    return objective["text"]


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("approve_plan", {"approvedStepIds": ["step_relationship_summary"]}),
        ("reject_plan", {"reason": "Too broad; focus on credit risk only."}),
        ("edit_plan", {"objective": "Narrow the meeting prep to risk themes."}),
        ("remove_step", {"stepId": "step_industry_research"}),
        (
            "reorder_steps",
            {
                "orderedStepIds": [
                    "step_internal_knowledge",
                    "step_relationship_summary",
                    "step_industry_research",
                    "step_synthesis",
                ]
            },
        ),
        (
            "choose_agent",
            {
                "stepId": "step_industry_research",
                "agentId": "credit_risk",
            },
        ),
        (
            "replace_agent",
            {
                "stepId": "step_industry_research",
                "replacementAgentId": "credit_risk",
            },
        ),
        (
            "add_instruction",
            {
                "stepId": "step_internal_knowledge",
                "instruction": "Prioritize open covenant questions.",
            },
        ),
        (
            "add_instructions",
            {
                "stepId": "step_internal_knowledge",
                "instructions": "Prioritize open covenant questions.",
            },
        ),
    ],
)
def test_parser_supports_plan_user_action_types(
    action_type: str,
    payload: dict[str, object],
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import parse_plan_user_action

    event = _action(action_type, payload)

    # Act
    parsed = parse_plan_user_action(event)

    # Assert
    assert parsed.type == action_type
    assert parsed.surface_id == "surface_plan_meeting_prep"
    assert parsed.plan_id == "plan_meeting_prep"
    if action_type == "reject_plan":
        assert parsed.plan_version == 1
    else:
        assert parsed.plan_version == 1


def test_parser_plan_action_support_matches_shared_contract() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import (
        SUPPORTED_PLAN_USER_ACTION_TYPES,
    )
    from orchestrator_demo.contracts import PLAN_USER_ACTION_TYPES

    # Act / Assert
    assert SUPPORTED_PLAN_USER_ACTION_TYPES == PLAN_USER_ACTION_TYPES


def test_parser_rejects_natural_language_approval() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import (
        StructuredUserActionRequiredError,
        parse_plan_user_action,
    )

    natural_language_message = "Looks good, approve this plan."

    # Act / Assert
    with pytest.raises(StructuredUserActionRequiredError, match="structured A2UI"):
        parse_plan_user_action(natural_language_message)


def test_parser_accepts_renderer_control_event_context_from_approval_canvas() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import parse_plan_user_action

    event = _renderer_control_event(
        "add_instruction",
        {
            "stepId": "step_internal_knowledge",
            "instruction": "Prioritize covenant follow-ups.",
        },
    )

    # Act
    parsed = parse_plan_user_action(event)

    # Assert
    assert parsed.type == "add_instruction"
    assert parsed.surface_id == "surface_plan_meeting_prep"
    assert parsed.plan_id == "plan_meeting_prep"
    assert parsed.plan_version == 1
    assert parsed.payload["stepId"] == "step_internal_knowledge"


def test_parser_preserves_context_plan_metadata_for_event_name_payload() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import parse_plan_user_action

    event = {
        "event": {
            "name": "approve_plan",
            "context": {
                "surfaceId": "surface_plan_meeting_prep",
                "planId": "plan_meeting_prep",
                "planVersion": 1,
                "payload": {
                    "approvedStepIds": ["step_relationship_summary"],
                },
            },
        },
    }

    # Act
    parsed = parse_plan_user_action(event)

    # Assert
    assert parsed.type == "approve_plan"
    assert parsed.surface_id == "surface_plan_meeting_prep"
    assert parsed.plan_id == "plan_meeting_prep"
    assert parsed.plan_version == 1
    assert parsed.payload["surfaceId"] == "surface_plan_meeting_prep"
    assert parsed.payload["planId"] == "plan_meeting_prep"
    assert parsed.payload["planVersion"] == 1
    assert parsed.payload["approvedStepIds"] == ["step_relationship_summary"]


@pytest.mark.parametrize(
    ("payload_metadata", "expected_field"),
    [
        ({"surfaceId": "surface_plan_other"}, "surfaceId"),
        ({"planId": "plan_other"}, "planId"),
        ({"planVersion": 2}, "planVersion"),
    ],
)
def test_parser_rejects_conflicting_event_name_plan_metadata(
    payload_metadata: dict[str, object],
    expected_field: str,
) -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import (
        PlanUserActionParseError,
        parse_plan_user_action,
    )

    event = {
        "event": {
            "name": "approve_plan",
            "context": {
                "surfaceId": "surface_plan_meeting_prep",
                "planId": "plan_meeting_prep",
                "planVersion": 1,
                "payload": {
                    "approvedStepIds": ["step_relationship_summary"],
                    **payload_metadata,
                },
            },
        },
    }

    # Act / Assert
    with pytest.raises(
        PlanUserActionParseError,
        match=f"conflicting event metadata for {expected_field}",
    ):
        parse_plan_user_action(event)


def test_natural_language_approval_does_not_approve_or_execute_plan() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()

    # Act
    result = store.handle_natural_language("Approve this plan and run it.")
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "ignored"
    assert result.graph_created is False
    assert result.specialists_called is False
    assert record.status == "draft"
    assert record.approved_plan is None


def test_renderer_control_event_context_mutates_draft_and_returns_canvas() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    event = _renderer_control_event(
        "add_instruction",
        {
            "stepId": "step_internal_knowledge",
            "instruction": "Prioritize covenant follow-ups.",
        },
    )

    # Act
    result = store.apply_user_action(event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert result.graph_created is False
    assert result.specialists_called is False
    assert record.draft_plan.plan_version == 2
    assert (
        "Additional instruction: Prioritize covenant follow-ups."
        in record.draft_plan.steps[1].instruction
    )
    assert result.refreshed_a2ui_part is not None
    assert "planVersion: 2" in _plan_metadata_text(result.refreshed_a2ui_part)


def test_edit_plan_rejects_objective_change_without_mutating_draft() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, original_plan = _store_with_meeting_plan()
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)
    event = _action(
        "edit_plan",
        {"objective": "Prepare only risk-focused meeting notes."},
    )

    # Act / Assert
    with pytest.raises(PlanMutationError, match="cannot change the routed objective"):
        store.apply_user_action(event)

    assert original_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert original_plan.plan_version == 1
    assert store.get("plan_meeting_prep").draft_plan == draft_snapshot


def test_generated_edit_plan_control_payload_refreshes_canvas() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    event = _renderer_control_event(
        "edit_plan",
        {"editableFields": ["steps", "selectedAgents"]},
    )

    # Act
    result = store.apply_user_action(event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert record.draft_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert record.draft_plan.plan_version == 2
    assert result.refreshed_a2ui_part is not None
    assert "planVersion: 2" in _plan_metadata_text(result.refreshed_a2ui_part)


def test_remove_reorder_replace_and_add_instruction_mutations_refresh_draft() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()

    # Act
    removed = store.apply_user_action(
        _action("remove_step", {"stepId": "step_industry_research"})
    )
    reordered = store.apply_user_action(
        {
            "userAction": {
                "type": "reorder_steps",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 2,
                    "orderedStepIds": [
                        "step_internal_knowledge",
                        "step_relationship_summary",
                        "step_synthesis",
                    ],
                },
            }
        }
    )
    replaced = store.apply_user_action(
        {
            "userAction": {
                "type": "replace_agent",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 3,
                    "stepId": "step_relationship_summary",
                    "replacementAgentId": "credit_risk",
                },
            }
        }
    )
    instructed = store.apply_user_action(
        {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 4,
                    "stepId": "step_internal_knowledge",
                    "instruction": "Emphasize open follow-up items.",
                },
            }
        }
    )
    record = store.get("plan_meeting_prep")

    # Assert
    assert removed.refreshed_a2ui_part is not None
    assert reordered.refreshed_a2ui_part is not None
    assert replaced.refreshed_a2ui_part is not None
    assert instructed.refreshed_a2ui_part is not None
    assert "planVersion: 2" in _plan_metadata_text(removed.refreshed_a2ui_part)
    assert "planVersion: 3" in _plan_metadata_text(reordered.refreshed_a2ui_part)
    assert "planVersion: 4" in _plan_metadata_text(replaced.refreshed_a2ui_part)
    assert "planVersion: 5" in _plan_metadata_text(instructed.refreshed_a2ui_part)
    assert record.draft_plan.plan_version == 5
    assert [step.step_id for step in record.draft_plan.steps] == [
        "step_internal_knowledge",
        "step_relationship_summary",
        "step_synthesis",
    ]
    assert set(record.draft_plan.steps[-1].depends_on) == {
        "step_internal_knowledge",
        "step_relationship_summary",
    }
    assert record.draft_plan.steps[1].agent_id == "credit_risk"
    assert record.draft_plan.steps[1].expected_output == (
        "Credit risk themes, missing data, and caveats."
    )
    assert record.draft_plan.steps[1].data_source_categories == ["credit_risk"]
    assert "Flag credit risk themes" in record.draft_plan.steps[1].instruction
    assert "credit_risk" in record.draft_plan.selected_agents
    assert "industry_research" not in record.draft_plan.selected_agents
    assert "credit_risk" in record.draft_plan.data_source_categories
    assert "relationship_history" not in record.draft_plan.data_source_categories
    assert (
        "Additional instruction: Emphasize open follow-up items."
        in record.draft_plan.steps[0].instruction
    )


def test_replace_agent_refreshes_metadata_before_approving_edited_plan() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()

    # Act
    replaced = store.apply_user_action(
        _action(
            "replace_agent",
            {
                "stepId": "step_industry_research",
                "replacementAgentId": "credit_risk",
            },
        )
    )
    draft_record = store.get("plan_meeting_prep")
    approval_result = store.apply_user_action(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 2,
                    "approvedStepIds": [
                        "step_relationship_summary",
                        "step_internal_knowledge",
                        "step_industry_research",
                        "step_synthesis",
                    ],
                },
            }
        }
    )

    # Assert
    assert replaced.status == "draft_updated"
    replacement_step = next(
        step
        for step in draft_record.draft_plan.steps
        if step.step_id == "step_industry_research"
    )
    assert replacement_step.agent_id == "credit_risk"
    assert replacement_step.instruction == (
        "Flag credit risk themes, missing credit context, covenant concerns, "
        "and repayment indicators for: Prepare me for tomorrow's meeting with "
        "ABC Manufacturing."
    )
    assert replacement_step.expected_output == (
        "Credit risk themes, missing data, and caveats."
    )
    assert replacement_step.data_source_categories == ["credit_risk"]
    assert draft_record.draft_plan.selected_agents == [
        "relationship_summary",
        "internal_knowledge",
        "credit_risk",
        "synthesis",
    ]
    assert draft_record.draft_plan.data_source_categories == [
        "relationship_history",
        "internal_crm",
        "credit_risk",
    ]

    assert approval_result.status == "approved"
    assert approval_result.approved_plan is not None
    approved_step = next(
        step
        for step in approval_result.approved_plan.steps
        if step.step_id == "step_industry_research"
    )
    assert approved_step == replacement_step
    assert approval_result.graph_execution is not None
    replacement_request = next(
        request
        for request in approval_result.graph_execution.specialist_requests
        if request.step_id == "step_industry_research"
    )
    assert replacement_request.agent_id == "credit_risk"
    assert replacement_request.user_input == replacement_step.instruction
    assert (
        replacement_request.context["expectedOutput"]
        == replacement_step.expected_output
    )
    assert replacement_request.context["dataSourceCategories"] == ["credit_risk"]


def test_remove_step_rewires_dependents_to_removed_step_dependencies() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore

    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    store.add_draft(_dependent_chain_plan())

    # Act
    result = store.apply_user_action(
        _action("remove_step", {"stepId": "step_internal_knowledge"})
    )
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert record.draft_plan.plan_version == 2
    assert [step.step_id for step in record.draft_plan.steps] == [
        "step_relationship_summary",
        "step_credit_risk",
    ]
    credit_risk_step = record.draft_plan.steps[-1]
    assert credit_risk_step.depends_on == ["step_relationship_summary"]
    assert record.draft_plan.selected_agents == [
        "relationship_summary",
        "credit_risk",
    ]
    assert "internal_crm" not in record.draft_plan.data_source_categories


def test_remove_step_reconnects_dependents_to_removed_step_dependencies() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore

    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    store.add_draft(
        ExecutionPlan(
            plan_id="plan_meeting_prep",
            objective="Prepare sequential credit context.",
            detected_intents=["internal_knowledge", "credit_risk"],
            selected_agents=["internal_knowledge", "credit_risk", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_internal_knowledge",
                    agent_id="internal_knowledge",
                    instruction="Review internal notes.",
                    expected_output="Internal context.",
                    data_source_categories=["internal_crm"],
                ),
                PlanStep(
                    step_id="step_credit_risk",
                    agent_id="credit_risk",
                    instruction="Assess credit risk from internal notes.",
                    depends_on=["step_internal_knowledge"],
                    expected_output="Credit risk context.",
                    data_source_categories=["credit"],
                ),
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the brief.",
                    depends_on=["step_credit_risk"],
                    expected_output="Final brief.",
                    data_source_categories=["specialist_outputs"],
                ),
            ],
            data_source_categories=["internal_crm", "credit"],
            approval_surface_id="surface_plan_meeting_prep",
        )
    )

    # Act
    result = store.apply_user_action(
        _action("remove_step", {"stepId": "step_credit_risk"})
    )
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert [step.step_id for step in record.draft_plan.steps] == [
        "step_internal_knowledge",
        "step_synthesis",
    ]
    assert record.draft_plan.steps[-1].depends_on == ["step_internal_knowledge"]


def test_remove_step_rejects_orphaned_conditional_dependency() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanMutationError,
    )

    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    store.add_draft(
        ExecutionPlan(
            plan_id="plan_meeting_prep",
            objective="Prepare conditional review context.",
            detected_intents=["internal_knowledge", "credit_risk"],
            selected_agents=["internal_knowledge", "credit_risk", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_internal_knowledge",
                    agent_id="internal_knowledge",
                    instruction="Review internal notes.",
                    expected_output="Internal context.",
                    data_source_categories=["internal_crm"],
                ),
                PlanStep(
                    step_id="step_credit_risk",
                    agent_id="credit_risk",
                    instruction="Review credit risk only if needed.",
                    depends_on=["step_internal_knowledge"],
                    condition="needs_review",
                    expected_output="Credit review.",
                    data_source_categories=["credit"],
                ),
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the brief.",
                    depends_on=["step_credit_risk"],
                    expected_output="Final brief.",
                    data_source_categories=["specialist_outputs"],
                ),
            ],
            data_source_categories=["internal_crm", "credit"],
            approval_surface_id="surface_plan_meeting_prep",
        )
    )
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match="conditional step"):
        store.apply_user_action(
            _action("remove_step", {"stepId": "step_internal_knowledge"})
        )

    record_after_failed_remove = store.get("plan_meeting_prep")
    assert record_after_failed_remove.status == "draft"
    assert record_after_failed_remove.draft_plan == draft_snapshot


def test_remove_step_rejects_conditional_data_quality_step() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanMutationError,
    )

    store = ApprovalStateStore(
        agent_descriptors=[
            _descriptor("internal_knowledge", "Internal Knowledge Agent"),
            _descriptor("data_quality", "Data Quality Agent"),
            _descriptor("synthesis", "Synthesis Agent"),
        ]
    )
    store.add_draft(
        ExecutionPlan(
            plan_id="plan_meeting_prep",
            objective="Prepare a briefing and check missing internal data.",
            detected_intents=["meeting_prep"],
            selected_agents=["internal_knowledge", "data_quality", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_internal_knowledge",
                    agent_id="internal_knowledge",
                    instruction="Review internal CRM notes.",
                    expected_output="Internal context.",
                    data_source_categories=["internal_crm"],
                ),
                PlanStep(
                    step_id="step_data_quality",
                    agent_id="data_quality",
                    instruction="Check missing or stale internal data if needed.",
                    depends_on=["step_internal_knowledge"],
                    condition="missing_internal_data",
                    expected_output="Data quality gaps.",
                    data_source_categories=["data_quality"],
                ),
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the available context.",
                    depends_on=["step_data_quality"],
                    expected_output="Final briefing or clarification need.",
                    data_source_categories=["specialist_outputs"],
                ),
            ],
            data_source_categories=["internal_crm", "data_quality"],
            approval_surface_id="surface_plan_meeting_prep",
        )
    )
    draft_snapshot = store.get("plan_meeting_prep").draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match="conditional data_quality step"):
        store.apply_user_action(
            _action("remove_step", {"stepId": "step_data_quality"})
        )

    record = store.get("plan_meeting_prep")
    assert record.status == "draft"
    assert record.draft_plan == draft_snapshot


def test_generated_path_bound_replace_and_instruction_values_mutate_draft() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    replace_event = _renderer_control_event(
        "replace_agent",
        {
            "stepId": "step_relationship_summary",
            "replacementAgentId": {
                "path": "/approvalEdits/step_relationship_summary/replacementAgentId"
            },
        },
    )
    replace_event["approvalEdits"] = {
        "step_relationship_summary": {
            "replacementAgentId": "credit_risk",
        }
    }
    instruction_event = _renderer_control_event(
        "add_instruction",
        {
            "stepId": "step_internal_knowledge",
            "instruction": {
                "path": "/approvalEdits/step_internal_knowledge/instruction"
            },
        },
        plan_version=2,
    )
    instruction_event["data"] = {
        "approvalEdits": {
            "step_internal_knowledge": {
                "instruction": "Emphasize unresolved covenant follow-ups.",
            }
        }
    }

    # Act
    replaced = store.apply_user_action(replace_event)
    instructed = store.apply_user_action(instruction_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert replaced.status == "draft_updated"
    assert instructed.status == "draft_updated"
    assert record.draft_plan.steps[0].agent_id == "credit_risk"
    assert (
        "Additional instruction: Emphasize unresolved covenant follow-ups."
        in record.draft_plan.steps[1].instruction
    )
    assert record.draft_plan.plan_version == 3


def test_user_action_wrapper_merges_sibling_edit_state_for_path_bound_values() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    replace_event = {
        "userAction": {
            "type": "replace_agent",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 1,
                "stepId": "step_relationship_summary",
                "replacementAgentId": {
                    "path": "/approvalEdits/step_relationship_summary/replacementAgentId"
                },
            },
        },
        "values": {
            "approvalEdits": {
                "step_relationship_summary": {
                    "replacementAgentId": "credit_risk",
                }
            }
        },
    }
    instruction_event = {
        "userAction": {
            "type": "add_instruction",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 2,
                "stepId": "step_internal_knowledge",
                "instruction": {
                    "path": "/approvalEdits/step_internal_knowledge/instruction"
                },
            },
        },
        "formData": {
            "approvalEdits": {
                "step_internal_knowledge": {
                    "instruction": "Add deposit concentration watch-outs.",
                }
            }
        },
    }

    # Act
    replaced = store.apply_user_action(replace_event)
    instructed = store.apply_user_action(instruction_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert replaced.status == "draft_updated"
    assert instructed.status == "draft_updated"
    assert record.draft_plan.steps[0].agent_id == "credit_risk"
    assert (
        "Additional instruction: Add deposit concentration watch-outs."
        in record.draft_plan.steps[1].instruction
    )
    assert record.draft_plan.plan_version == 3


def test_direct_action_context_merges_and_strips_sibling_edit_state() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    replace_event = {
        "type": "replace_agent",
        "surfaceId": "surface_plan_meeting_prep",
        "payload": {
            "planId": "plan_meeting_prep",
            "editedPlanVersion": 1,
            "stepId": "step_relationship_summary",
            "replacementAgentId": {
                "path": "/approvalEdits/step_relationship_summary/replacementAgentId"
            },
        },
        "formData": {
            "approvalEdits": {
                "step_relationship_summary": {
                    "replacementAgentId": "credit_risk",
                }
            }
        },
    }
    instruction_event = {
        "type": "add_instruction",
        "surfaceId": "surface_plan_meeting_prep",
        "payload": {
            "planId": "plan_meeting_prep",
            "editedPlanVersion": 2,
            "stepId": "step_internal_knowledge",
            "instruction": {
                "path": "/approvalEdits/step_internal_knowledge/instruction"
            },
        },
        "formData": {
            "approvalEdits": {
                "step_internal_knowledge": {
                    "instruction": "Add deposit concentration watch-outs.",
                }
            }
        },
    }

    # Act
    replaced = store.apply_user_action(replace_event)
    instructed = store.apply_user_action(instruction_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert replaced.status == "draft_updated"
    assert instructed.status == "draft_updated"
    assert record.draft_plan.steps[0].agent_id == "credit_risk"
    assert (
        "Additional instruction: Add deposit concentration watch-outs."
        in record.draft_plan.steps[1].instruction
    )
    assert record.draft_plan.plan_version == 3


def test_direct_action_context_merges_data_edit_state_before_unwrapping() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    event = {
        "type": "add_instruction",
        "surfaceId": "surface_plan_meeting_prep",
        "payload": {
            "planId": "plan_meeting_prep",
            "editedPlanVersion": 1,
            "stepId": "step_internal_knowledge",
            "instruction": {
                "path": "/approvalEdits/step_internal_knowledge/instruction"
            },
        },
        "data": {
            "approvalEdits": {
                "step_internal_knowledge": {
                    "instruction": "Add borrower liquidity watch-outs.",
                }
            }
        },
    }

    # Act
    result = store.apply_user_action(event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert record.draft_plan.plan_version == 2
    assert (
        "Additional instruction: Add borrower liquidity watch-outs."
        in record.draft_plan.steps[1].instruction
    )


@pytest.mark.parametrize("edit_state_key", ["values", "formData", "state"])
def test_data_wrapper_merges_sibling_edit_state_for_path_bound_values(
    edit_state_key: str,
) -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    event = {
        "data": {
            "userAction": {
                "type": "add_instruction",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 1,
                    "stepId": "step_internal_knowledge",
                    "instruction": {
                        "path": "/approvalEdits/step_internal_knowledge/instruction"
                    },
                },
            }
        },
        edit_state_key: {
            "approvalEdits": {
                "step_internal_knowledge": {
                    "instruction": (
                        f"Include treasury exposure notes from {edit_state_key}."
                    ),
                }
            }
        },
    }

    # Act
    result = store.apply_user_action(event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert record.draft_plan.plan_version == 2
    assert (
        f"Additional instruction: Include treasury exposure notes from {edit_state_key}."
        in record.draft_plan.steps[1].instruction
    )


def test_remove_step_rejects_plan_with_only_synthesis_remaining() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    store.apply_user_action(
        _action("remove_step", {"stepId": "step_relationship_summary"})
    )
    store.apply_user_action(
        {
            "userAction": {
                "type": "remove_step",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 2,
                    "stepId": "step_internal_knowledge",
                },
            }
        }
    )
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match="non-synthesis specialist step"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "remove_step",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 3,
                        "stepId": "step_industry_research",
                    },
                }
            }
        )

    record_after_failed_remove = store.get("plan_meeting_prep")
    assert record_after_failed_remove.status == "draft"
    assert record_after_failed_remove.draft_plan == draft_snapshot


@pytest.mark.parametrize(
    ("action_type", "payload", "message"),
    [
        (
            "remove_step",
            {"stepId": "step_synthesis"},
            "final synthesis step",
        ),
        (
            "replace_agent",
            {
                "stepId": "step_synthesis",
                "replacementAgentId": "credit_risk",
            },
            "final synthesis step",
        ),
        (
            "reorder_steps",
            {
                "orderedStepIds": [
                    "step_synthesis",
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                ]
            },
            "final synthesis step",
        ),
    ],
)
def test_draft_mutations_preserve_required_final_synthesis_step(
    action_type: str,
    payload: dict[str, object],
    message: str,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match=message):
        store.apply_user_action(_action(action_type, payload))

    record_after_failed_mutation = store.get("plan_meeting_prep")
    assert record_after_failed_mutation.status == "draft"
    assert record_after_failed_mutation.draft_plan == draft_snapshot


def test_choose_agent_and_add_instructions_aliases_mutate_draft() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()

    # Act
    chosen = store.apply_user_action(
        _action(
            "choose_agent",
            {
                "stepId": "step_industry_research",
                "agentId": "credit_risk",
            },
        )
    )
    instructed = store.apply_user_action(
        {
            "userAction": {
                "type": "add_instructions",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 2,
                    "stepId": "step_internal_knowledge",
                    "instructions": ["Prioritize covenant follow-ups."],
                },
            }
        }
    )
    record = store.get("plan_meeting_prep")

    # Assert
    assert chosen.status == "draft_updated"
    assert instructed.status == "draft_updated"
    assert record.draft_plan.plan_version == 3
    assert record.draft_plan.steps[2].agent_id == "credit_risk"
    assert "credit_risk" in record.draft_plan.selected_agents
    assert "industry_research" not in record.draft_plan.selected_agents
    assert (
        "Additional instruction: Prioritize covenant follow-ups."
        in record.draft_plan.steps[1].instruction
    )


def test_replace_agent_refreshes_step_and_plan_metadata_before_approval() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()

    # Act
    replaced = store.apply_user_action(
        _action(
            "replace_agent",
            {
                "stepId": "step_industry_research",
                "replacementAgentId": "credit_risk",
            },
        )
    )
    record = store.get("plan_meeting_prep")
    replaced_step = next(
        step
        for step in record.draft_plan.steps
        if step.step_id == "step_industry_research"
    )
    approval = store.apply_user_action(
        {
            "userAction": {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": replaced.plan_version,
                    "approvedStepIds": [
                        step.step_id for step in record.draft_plan.steps
                    ],
                },
            }
        }
    )

    # Assert
    assert replaced.status == "draft_updated"
    assert replaced_step.agent_id == "credit_risk"
    assert replaced_step.instruction.startswith("Flag credit risk themes")
    assert (
        replaced_step.expected_output
        == "Credit risk themes, missing data, and caveats."
    )
    assert replaced_step.data_source_categories == ["credit_risk"]
    assert record.draft_plan.data_source_categories == [
        "relationship_history",
        "internal_crm",
        "credit_risk",
    ]
    assert approval.approved_plan is not None
    approved_step = next(
        step
        for step in approval.approved_plan.steps
        if step.step_id == "step_industry_research"
    )
    assert approved_step.instruction == replaced_step.instruction
    assert approved_step.data_source_categories == ["credit_risk"]


def test_replace_agent_rejects_conditional_data_quality_step_before_approval() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanMutationError,
    )

    store = ApprovalStateStore(
        agent_descriptors=[
            _descriptor("internal_knowledge", "Internal Knowledge Agent"),
            _descriptor("data_quality", "Data Quality Agent"),
            _descriptor("credit_risk", "Credit Risk Agent"),
            _descriptor("synthesis", "Synthesis Agent"),
        ]
    )
    store.add_draft(
        ExecutionPlan(
            plan_id="plan_meeting_prep",
            objective="Prepare a briefing and check missing internal data.",
            detected_intents=["meeting_prep"],
            selected_agents=["internal_knowledge", "data_quality", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_internal_knowledge",
                    agent_id="internal_knowledge",
                    instruction="Review internal CRM notes.",
                    expected_output="Internal context.",
                    data_source_categories=["internal_crm"],
                ),
                PlanStep(
                    step_id="step_data_quality",
                    agent_id="data_quality",
                    instruction="Check missing or stale internal data if needed.",
                    depends_on=["step_internal_knowledge"],
                    condition="missing_internal_data",
                    expected_output="Data quality gaps.",
                    data_source_categories=["data_quality"],
                ),
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the available context.",
                    depends_on=["step_data_quality"],
                    expected_output="Final briefing or clarification need.",
                    data_source_categories=["specialist_outputs"],
                ),
            ],
            data_source_categories=["internal_crm", "data_quality"],
            approval_surface_id="surface_plan_meeting_prep",
        )
    )
    draft_snapshot = store.get("plan_meeting_prep").draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match="conditional data_quality step"):
        store.apply_user_action(
            _action(
                "replace_agent",
                {
                    "stepId": "step_data_quality",
                    "replacementAgentId": "credit_risk",
                },
            )
        )

    record = store.get("plan_meeting_prep")
    assert record.status == "draft"
    assert record.draft_plan == draft_snapshot


@pytest.mark.parametrize(
    ("action_type", "replacement_payload"),
    [
        (
            "replace_agent",
            {
                "replacementAgentId": (
                    "OPENROUTER_API_KEY="
                    "sk-or-v1-replacement-agent-secret-should-not-appear"
                )
            },
        ),
        (
            "choose_agent",
            {
                "agentId": (
                    "OPENROUTER_API_KEY="
                    "sk-or-v1-replacement-agent-secret-should-not-appear"
                )
            },
        ),
    ],
)
def test_unregistered_replacement_agent_error_omits_secret_like_input(
    action_type: str,
    replacement_payload: dict[str, object],
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    leaked_replacement_agent_id = str(next(iter(replacement_payload.values())))

    # Act / Assert
    with pytest.raises(PlanMutationError) as exc_info:
        store.apply_user_action(
            _action(
                action_type,
                {
                    "stepId": "step_industry_research",
                    **replacement_payload,
                },
            )
        )

    message = str(exc_info.value)
    assert "replacement agent is unavailable" in message
    assert leaked_replacement_agent_id not in message
    assert "OPENROUTER_API_KEY" not in message
    assert "sk-or-v1-replacement-agent-secret-should-not-appear" not in message


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        ("remove_step", {}),
        ("add_instruction", {"instruction": "Focus on recent deposit trends."}),
        ("replace_agent", {"replacementAgentId": "credit_risk"}),
    ],
)
def test_unknown_step_mutation_error_omits_secret_like_step_id(
    action_type: str,
    payload: dict[str, object],
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    leaked_step_id = (
        "OPENROUTER_API_KEY=sk-or-v1-step-id-secret-should-not-appear"
    )

    # Act / Assert
    with pytest.raises(PlanMutationError) as exc_info:
        store.apply_user_action(
            _action(
                action_type,
                {
                    "stepId": leaked_step_id,
                    **payload,
                },
            )
        )

    message = str(exc_info.value)
    assert message == "unknown stepId"
    assert leaked_step_id not in message
    assert "OPENROUTER_API_KEY" not in message
    assert "sk-or-v1-step-id-secret-should-not-appear" not in message
    assert leaked_step_id not in caplog.text
    assert "OPENROUTER_API_KEY" not in caplog.text
    assert "sk-or-v1-step-id-secret-should-not-appear" not in caplog.text


def test_missing_plan_error_omits_secret_like_plan_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanNotFoundError

    store, _original_plan = _store_with_meeting_plan()
    leaked_plan_id = "plan_sk-or-v1-missing-plan-secret-should-not-appear"

    # Act / Assert
    with pytest.raises(PlanNotFoundError) as exc_info:
        store.apply_user_action(
            {
                "userAction": {
                    "type": "approve_plan",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": leaked_plan_id,
                        "editedPlanVersion": 1,
                        "approvedStepIds": [
                            "step_relationship_summary",
                            "step_internal_knowledge",
                            "step_industry_research",
                            "step_synthesis",
                        ],
                    },
                }
            }
        )

    message = str(exc_info.value)
    assert message == "unknown plan"
    assert leaked_plan_id not in message
    assert "sk-or-v1-missing-plan-secret-should-not-appear" not in message
    assert leaked_plan_id not in caplog.text
    assert "sk-or-v1-missing-plan-secret-should-not-appear" not in caplog.text


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({}, "requires approvedStepIds"),
        (
            {"approvedStepIds": ["step_relationship_summary"]},
            "approvedStepIds must match",
        ),
        (
            {
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                    "step_unknown",
                ]
            },
            "approvedStepIds must match",
        ),
        (
            {
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                    "step_synthesis",
                ]
            },
            "approvedStepIds must match",
        ),
    ],
)
def test_approve_plan_rejects_malformed_approved_step_ids_before_freezing(
    payload: dict[str, object],
    message: str,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match=message):
        store.apply_user_action(_action("approve_plan", payload))

    record_after_failed_approval = store.get("plan_meeting_prep")
    assert record_after_failed_approval.status == "draft"
    assert record_after_failed_approval.approved_plan is None
    assert record_after_failed_approval.draft_plan == draft_snapshot


def test_approve_plan_rejects_mutable_plan_before_freezing_or_graph_execution() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanMutationError,
    )

    class RecordingGraphRuntime:
        def __init__(self) -> None:
            self.executed_plan_ids: list[str] = []

        def execute(self, plan: ExecutionPlan) -> object:
            self.executed_plan_ids.append(plan.plan_id)
            raise AssertionError("mutable plans must not reach graph execution")

    graph_runtime = RecordingGraphRuntime()
    store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=graph_runtime,
    )
    mutable_plan = _meeting_plan().model_copy(
        update={"immutable_after_approval": False}
    )
    store.add_draft(mutable_plan)
    draft_snapshot = store.get("plan_meeting_prep").draft_plan.model_copy(deep=True)
    approve_event = _action(
        "approve_plan",
        {
            "approvedStepIds": [
                "step_relationship_summary",
                "step_internal_knowledge",
                "step_industry_research",
                "step_synthesis",
            ]
        },
    )

    # Act / Assert
    with pytest.raises(PlanMutationError, match="immutable_after_approval=True"):
        store.apply_user_action(approve_event)

    record_after_failed_approval = store.get("plan_meeting_prep")
    assert graph_runtime.executed_plan_ids == []
    assert record_after_failed_approval.status == "draft"
    assert record_after_failed_approval.approved_plan is None
    assert record_after_failed_approval.draft_plan == draft_snapshot


def test_failed_graph_execution_freezes_approved_plan_state() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanAlreadyFinalError,
    )

    class FailingGraphRuntime:
        def __init__(self) -> None:
            self.specialist_calls: list[tuple[str, str]] = []

        def execute(self, plan: ExecutionPlan) -> object:
            self.specialist_calls.append((plan.plan_id, plan.steps[0].step_id))
            raise RuntimeError(f"graph failed for {plan.plan_id}")

    graph_runtime = FailingGraphRuntime()
    store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=graph_runtime,
    )
    store.add_draft(_meeting_plan())
    approve_event = _action(
        "approve_plan",
        {
            "approvedStepIds": [
                "step_relationship_summary",
                "step_internal_knowledge",
                "step_industry_research",
                "step_synthesis",
            ]
        },
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="graph failed for plan_meeting_prep"):
        store.apply_user_action(approve_event)

    record_after_failed_execution = store.get("plan_meeting_prep")
    assert graph_runtime.specialist_calls == [
        ("plan_meeting_prep", "step_relationship_summary")
    ]
    assert record_after_failed_execution.status == "approved_execution_failed"
    assert record_after_failed_execution.approved_plan is not None
    assert record_after_failed_execution.approved_version == 1
    assert record_after_failed_execution.execution_failure_reason == (
        "RuntimeError: graph failed for plan_meeting_prep"
    )

    with pytest.raises(
        PlanAlreadyFinalError,
        match="already approved_execution_failed",
    ):
        store.apply_user_action(
            _action(
                "add_instruction",
                {
                    "stepId": "step_internal_knowledge",
                    "instruction": "Retry after graph recovery.",
                },
            )
        )

    with pytest.raises(
        PlanAlreadyFinalError,
        match="already approved_execution_failed",
    ):
        store.apply_user_action(approve_event)


def test_draft_update_commits_only_after_refreshed_canvas_validation() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import A2UIEmissionError

    store, _original_plan = _store_with_meeting_plan()
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)
    event = _action(
        "add_instruction",
        {
            "stepId": "step_internal_knowledge",
            "instruction": (
                "Use OPENROUTER_API_KEY=sk-1234567890abcdef before review."
            ),
        },
    )

    # Act / Assert
    with pytest.raises(A2UIEmissionError, match="approval canvas failed"):
        store.apply_user_action(event)

    record_after_failed_edit = store.get("plan_meeting_prep")
    assert record_after_failed_edit.status == "draft"
    assert record_after_failed_edit.approved_plan is None
    assert record_after_failed_edit.draft_plan == draft_snapshot


def test_concurrent_duplicate_approvals_execute_graph_once() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanAlreadyFinalError,
    )

    class SlowGraphRuntime:
        def __init__(self) -> None:
            self.calls = 0
            self.first_execution_started = threading.Event()
            self.second_execution_started = threading.Event()
            self.release_execution = threading.Event()
            self._lock = threading.Lock()

        def execute(self, _plan: ExecutionPlan) -> object:
            with self._lock:
                self.calls += 1
                call_number = self.calls
            if call_number == 1:
                self.first_execution_started.set()
            else:
                self.second_execution_started.set()
            assert self.release_execution.wait(timeout=5)
            return SimpleNamespace(specialist_requests=("request",))

    graph_runtime = SlowGraphRuntime()
    store = ApprovalStateStore(
        agent_descriptors=_agent_descriptors(),
        graph_runtime=graph_runtime,
    )
    store.add_draft(_meeting_plan())
    approve_event = _action(
        "approve_plan",
        {
            "approvedStepIds": [
                "step_relationship_summary",
                "step_internal_knowledge",
                "step_industry_research",
                "step_synthesis",
            ]
        },
    )
    results: list[object] = []
    errors: list[BaseException] = []

    def approve() -> None:
        try:
            results.append(store.apply_user_action(approve_event))
        except BaseException as exc:  # pragma: no cover - asserted below
            errors.append(exc)

    first_thread = threading.Thread(target=approve)
    second_thread = threading.Thread(target=approve)

    # Act
    first_thread.start()
    assert graph_runtime.first_execution_started.wait(timeout=5)
    second_thread.start()
    overlapped = graph_runtime.second_execution_started.wait(timeout=0.2)
    graph_runtime.release_execution.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)

    # Assert
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not overlapped
    assert graph_runtime.calls == 1
    assert [getattr(result, "status", None) for result in results] == ["approved"]
    assert len(errors) == 1
    assert isinstance(errors[0], PlanAlreadyFinalError)


@pytest.mark.parametrize(
    ("action_type", "replacement_payload"),
    [
        ("replace_agent", {"replacementAgentId": "synthesis"}),
        ("choose_agent", {"agentId": "synthesis"}),
    ],
)
def test_replace_agent_aliases_reject_synthesis_only_plan(
    action_type: str,
    replacement_payload: dict[str, object],
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanMutationError

    store, _original_plan = _store_with_meeting_plan()
    store.apply_user_action(
        _action("remove_step", {"stepId": "step_relationship_summary"})
    )
    store.apply_user_action(
        {
            "userAction": {
                "type": "remove_step",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "editedPlanVersion": 2,
                    "stepId": "step_internal_knowledge",
                },
            }
        }
    )
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)

    # Act / Assert
    with pytest.raises(PlanMutationError, match="non-synthesis specialist step"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": action_type,
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 3,
                        "stepId": "step_industry_research",
                        **replacement_payload,
                    },
                }
            }
        )

    record_after_failed_replace = store.get("plan_meeting_prep")
    assert record_after_failed_replace.status == "draft"
    assert record_after_failed_replace.draft_plan == draft_snapshot


def test_approval_freezes_referenced_plan_version_and_rejects_future_mutation() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanAlreadyFinalError

    store, _original_plan = _store_with_meeting_plan()
    edit_result = store.apply_user_action(
        _action(
            "add_instruction",
            {
                "stepId": "step_internal_knowledge",
                "instruction": "Prepare risk-focused notes.",
            },
        )
    )
    approve_event = {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 2,
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                ],
            },
        }
    }

    # Act
    approval_result = store.apply_user_action(approve_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert edit_result.status == "draft_updated"
    assert approval_result.status == "approved"
    assert approval_result.graph_created is True
    assert approval_result.specialists_called is True
    assert approval_result.graph_execution is not None
    assert approval_result.graph_execution.graph.plan_id == "plan_meeting_prep"
    assert approval_result.graph_execution.graph.pattern == "fan_out_fan_in"
    assert approval_result.graph_execution.workflow.__class__.__module__.startswith(
        "google.adk.workflow"
    )
    assert [
        request.step_id for request in approval_result.graph_execution.specialist_requests
    ] == [
        "step_relationship_summary",
        "step_internal_knowledge",
        "step_industry_research",
        "step_synthesis",
    ]
    responses_by_agent = {
        response.agent_id: response
        for response in approval_result.graph_execution.specialist_responses
    }
    assert set(responses_by_agent) == {
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    }
    assert responses_by_agent["relationship_summary"].structured_output[
        "provenance"
    ]["generated_by"] == "relationship_summary"
    assert responses_by_agent["internal_knowledge"].structured_output[
        "provenance"
    ]["generated_by"] == "internal_knowledge"
    assert all(
        "synthetic" not in response.structured_output
        for response in responses_by_agent.values()
    )
    assert all(
        "completed approved step" not in response.content
        for response in responses_by_agent.values()
    )
    status_events = approval_result.graph_execution.status_events
    assert [event.status for event in status_events[:2]] == [
        "plan_approved",
        "graph_created",
    ]
    assert status_events[0].details == {"planVersion": 2}
    assert {event.status for event in status_events} >= {
        "plan_approved",
        "graph_created",
        "step_started",
        "step_completed",
        "synthesis_started",
        "final_response_ready",
    }
    assert record.status == "approved"
    assert record.approved_version == 2
    assert record.approved_plan is not None
    assert record.approved_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert (
        "Additional instruction: Prepare risk-focused notes."
        in record.approved_plan.steps[1].instruction
    )
    assert record.approved_plan.plan_version == 2
    assert approval_result.approved_plan is not None
    approval_result.approved_plan.objective = "Tampered after approval."
    assert record.approved_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    with pytest.raises(PlanAlreadyFinalError, match="already approved"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "add_instruction",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 2,
                        "stepId": "step_internal_knowledge",
                        "instruction": "This must not mutate an approved plan.",
                    },
                }
            }
        )


def test_approve_plan_rejects_unavailable_step_agent_before_execution() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanMutationError,
    )

    plan = ExecutionPlan(
        plan_id="plan_typo_agent",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=["meeting_prep"],
        selected_agents=["relationship_summarry"],
        steps=[
            PlanStep(
                step_id="step_typo_agent",
                agent_id="relationship_summarry",
                instruction="Summarize the relationship history.",
                expected_output="Relationship history and open follow-ups.",
            ),
        ],
        approval_surface_id="surface_plan_typo_agent",
    )
    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    store.add_draft(plan)

    # Act / Assert
    with pytest.raises(
        PlanMutationError,
        match="plan references unavailable agents: relationship_summarry",
    ):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "approve_plan",
                    "surfaceId": "surface_plan_typo_agent",
                    "payload": {
                        "planId": "plan_typo_agent",
                        "editedPlanVersion": 1,
                        "approvedStepIds": ["step_typo_agent"],
                    },
                }
            }
        )

    record_after_failed_execution = store.get("plan_typo_agent")
    assert record_after_failed_execution.status == "approved_execution_failed"
    assert record_after_failed_execution.approved_plan is not None
    assert record_after_failed_execution.approved_version == 1
    assert record_after_failed_execution.execution_failure_reason == (
        "GraphRuntimeError: no specialist handler registered for approved plan "
        "step step_typo_agent agent relationship_summarry"
    )


def test_default_graph_runtime_refreshes_handlers_for_live_registry_additions() -> None:
    from orchestrator_demo.orchestrator.approval_state import ApprovalStateStore

    current_descriptors = [
        descriptor
        for descriptor in _agent_descriptors()
        if descriptor.agent_id != "industry_research"
    ]
    store = ApprovalStateStore(agent_descriptors=lambda: current_descriptors)
    plan = _meeting_plan()
    store.add_draft(plan)

    current_descriptors = _agent_descriptors()

    result = store.apply_user_action(
        _action(
            "approve_plan",
            {"approvedStepIds": [step.step_id for step in plan.steps]},
        )
    )

    assert result.status == "approved"
    assert result.graph_execution is not None
    assert {
        request.agent_id for request in result.graph_execution.specialist_requests
    } == set(plan.selected_agents)
    record = store.get(plan.plan_id)
    assert record.status == "approved"
    assert record.approved_plan is not None


def test_graph_runtime_awaits_async_specialist_handlers() -> None:
    # Arrange
    from orchestrator_demo.contracts import SpecialistResponse
    from orchestrator_demo.orchestrator.graph_runtime import AdkGraphRuntime

    handler_requests = []

    async def async_relationship_handler(request):
        handler_requests.append(request)
        return SpecialistResponse(
            response_id=f"response_{request.request_id.removeprefix('request_')}",
            agent_id=request.agent_id,
            content="async relationship summary completed",
            structured_output={"handler": "async", "step_id": request.step_id},
        )

    runtime = AdkGraphRuntime(
        specialist_handlers={"relationship_summary": async_relationship_handler}
    )
    relationship_step = _meeting_plan().steps[0].model_copy(
        update={"parallel_group": None}
    )
    plan = ExecutionPlan(
        plan_id="plan_relationship_summary",
        objective="Summarize relationship history.",
        detected_intents=["relationship_summary"],
        selected_agents=["relationship_summary"],
        steps=[relationship_step],
        data_source_categories=["relationship_history"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_relationship_summary",
    )

    # Act
    result = runtime.execute(plan)

    # Assert
    assert len(handler_requests) == 1
    assert handler_requests[0].agent_id == "relationship_summary"
    relationship_response = next(
        response
        for response in result.specialist_responses
        if response.agent_id == "relationship_summary"
    )
    assert relationship_response.content == "async relationship summary completed"
    assert relationship_response.structured_output == {
        "handler": "async",
        "step_id": "step_relationship_summary",
    }


def test_approval_record_accessors_return_defensive_snapshots() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import (
        ApprovalStateStore,
        PlanAlreadyFinalError,
    )

    store = ApprovalStateStore(agent_descriptors=_agent_descriptors())
    add_draft_snapshot = store.add_draft(_meeting_plan())

    # Act
    add_draft_snapshot.status = "approved"
    add_draft_snapshot.draft_plan.steps[0].instruction = "Tampered by caller."

    # Assert
    live_draft_snapshot = store.get("plan_meeting_prep")
    assert live_draft_snapshot.status == "draft"
    assert live_draft_snapshot.draft_plan.steps[0].instruction == (
        "Summarize the relationship history."
    )

    store.apply_user_action(
        _action(
            "approve_plan",
            {
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                ]
            },
        )
    )
    approved_snapshot = store.get("plan_meeting_prep")
    approved_plan_snapshot = approved_snapshot.approved_plan
    assert approved_plan_snapshot is not None

    approved_snapshot.status = "draft"
    approved_snapshot.draft_plan.steps[1].instruction = "Tampered draft."
    approved_plan_snapshot.objective = "Tampered approved objective."

    with pytest.raises(PlanAlreadyFinalError, match="already approved"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "add_instruction",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 1,
                        "stepId": "step_internal_knowledge",
                        "instruction": "This must not mutate an approved plan.",
                    },
                }
            }
        )

    live_approved_snapshot = store.get("plan_meeting_prep")
    assert live_approved_snapshot.status == "approved"
    assert live_approved_snapshot.draft_plan.steps[1].instruction == (
        "Review internal CRM notes."
    )
    assert live_approved_snapshot.approved_plan is not None
    assert live_approved_snapshot.approved_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )


def test_add_draft_rejects_duplicate_approved_plan_id_without_reopening_plan() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanAlreadyFinalError

    store, original_plan = _store_with_meeting_plan()
    store.apply_user_action(
        _action(
            "approve_plan",
            {
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                ]
            },
        )
    )

    # Act / Assert
    with pytest.raises(PlanAlreadyFinalError, match="already approved"):
        store.add_draft(original_plan)

    record = store.get("plan_meeting_prep")
    assert record.status == "approved"
    assert record.approved_plan is not None
    with pytest.raises(PlanAlreadyFinalError, match="already approved"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "add_instruction",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 1,
                        "stepId": "step_internal_knowledge",
                        "instruction": "This must not reopen an approved plan.",
                    },
                }
            }
        )


def test_add_draft_rejects_duplicate_rejected_plan_id_without_reopening_plan() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanAlreadyFinalError

    store, original_plan = _store_with_meeting_plan()
    store.apply_user_action(
        _action("reject_plan", {"reason": "Do not run this workflow."})
    )

    # Act / Assert
    with pytest.raises(PlanAlreadyFinalError, match="already rejected"):
        store.add_draft(original_plan)

    record = store.get("plan_meeting_prep")
    assert record.status == "rejected"
    assert record.rejection_reason == "Do not run this workflow."
    with pytest.raises(PlanAlreadyFinalError, match="already rejected"):
        store.apply_user_action(
            {
                "userAction": {
                    "type": "add_instruction",
                    "surfaceId": "surface_plan_meeting_prep",
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 1,
                        "stepId": "step_internal_knowledge",
                        "instruction": "This must not reopen a rejected plan.",
                    },
                }
            }
        )


def test_stale_plan_version_rejects_approval_and_draft_mutation_without_state_change() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanVersionConflictError

    store, _original_plan = _store_with_meeting_plan()
    update_result = store.apply_user_action(
        _action(
            "add_instruction",
            {
                "stepId": "step_internal_knowledge",
                "instruction": "Prepare risk-focused notes.",
            },
        )
    )
    record = store.get("plan_meeting_prep")
    draft_snapshot = record.draft_plan.model_copy(deep=True)
    stale_approval_event = _action(
        "approve_plan",
        {
            "approvedStepIds": [
                "step_relationship_summary",
                "step_internal_knowledge",
                "step_industry_research",
                "step_synthesis",
            ]
        },
    )
    stale_edit_event = _action(
        "add_instruction",
        {
            "stepId": "step_internal_knowledge",
            "instruction": "This stale edit must not mutate the draft.",
        },
    )

    # Act / Assert
    assert update_result.status == "draft_updated"
    assert draft_snapshot.plan_version == 2
    with pytest.raises(PlanVersionConflictError, match="version 2, got 1"):
        store.apply_user_action(stale_approval_event)

    record_after_stale_approval = store.get("plan_meeting_prep")
    assert record_after_stale_approval.status == "draft"
    assert record_after_stale_approval.approved_plan is None
    assert record_after_stale_approval.draft_plan == draft_snapshot

    with pytest.raises(PlanVersionConflictError, match="version 2, got 1"):
        store.apply_user_action(stale_edit_event)

    record_after_stale_edit = store.get("plan_meeting_prep")
    assert record_after_stale_edit.status == "draft"
    assert record_after_stale_edit.approved_plan is None
    assert record_after_stale_edit.draft_plan == draft_snapshot


def test_rejection_marks_draft_rejected_and_does_not_create_graph_or_call_specialists() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    reject_event = _action(
        "reject_plan",
        {"reason": "Too broad; focus on credit risk only."},
    )

    # Act
    result = store.apply_user_action(reject_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "rejected"
    assert result.graph_created is False
    assert result.specialists_called is False
    assert result.refreshed_a2ui_part is None
    assert record.status == "rejected"
    assert record.rejection_reason == "Too broad; focus on credit risk only."
    assert record.approved_plan is None


def test_empty_reject_reason_from_canvas_marks_draft_rejected_without_reason() -> None:
    # Arrange
    store, _original_plan = _store_with_meeting_plan()
    reject_event = _renderer_control_event("reject_plan", {"reason": ""})

    # Act
    result = store.apply_user_action(reject_event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "rejected"
    assert result.rejection_reason is None
    assert record.status == "rejected"
    assert record.rejection_reason is None
    assert record.approved_plan is None


def test_plan_action_requires_matching_approval_surface_before_state_change() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanSurfaceMismatchError

    store, _original_plan = _store_with_meeting_plan()
    wrong_surface_event = {
        "userAction": {
            "type": "approve_plan",
            "surfaceId": "surface_plan_other",
            "payload": {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 1,
                "approvedStepIds": [
                    "step_relationship_summary",
                    "step_internal_knowledge",
                    "step_industry_research",
                    "step_synthesis",
                ],
            },
        }
    }

    # Act / Assert
    with pytest.raises(PlanSurfaceMismatchError, match="approval surface"):
        store.apply_user_action(wrong_surface_event)

    record = store.get("plan_meeting_prep")
    assert record.status == "draft"
    assert record.approved_plan is None
    assert record.draft_plan.plan_version == 1


def test_surface_mismatch_error_omits_secret_like_surface_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanSurfaceMismatchError

    store, _original_plan = _store_with_meeting_plan()
    leaked_surface_id = "surface_plan_sk-or-v1-surface-secret-should-not-appear"

    # Act / Assert
    with pytest.raises(PlanSurfaceMismatchError) as exc_info:
        store.apply_user_action(
            {
                "userAction": {
                    "type": "approve_plan",
                    "surfaceId": leaked_surface_id,
                    "payload": {
                        "planId": "plan_meeting_prep",
                        "editedPlanVersion": 1,
                        "approvedStepIds": [
                            "step_relationship_summary",
                            "step_internal_knowledge",
                            "step_industry_research",
                            "step_synthesis",
                        ],
                    },
                }
            }
        )

    message = str(exc_info.value)
    assert "approval surface" in message
    assert leaked_surface_id not in message
    assert "sk-or-v1-surface-secret-should-not-appear" not in message
    assert leaked_surface_id not in caplog.text
    assert "sk-or-v1-surface-secret-should-not-appear" not in caplog.text
