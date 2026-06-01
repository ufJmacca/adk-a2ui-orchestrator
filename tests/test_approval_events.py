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


def test_edit_plan_mutates_only_draft_state_increments_version_and_returns_canvas() -> None:
    # Arrange
    store, original_plan = _store_with_meeting_plan()
    event = _action(
        "edit_plan",
        {"objective": "Prepare only risk-focused meeting notes."},
    )

    # Act
    result = store.apply_user_action(event)
    record = store.get("plan_meeting_prep")

    # Assert
    assert result.status == "draft_updated"
    assert result.graph_created is False
    assert result.specialists_called is False
    assert original_plan.objective == (
        "Prepare me for tomorrow's meeting with ABC Manufacturing."
    )
    assert original_plan.plan_version == 1
    assert record.draft_plan.objective == "Prepare only risk-focused meeting notes."
    assert record.draft_plan.plan_version == 2
    assert len(result.refreshed_a2ui_parts) == 2
    assert isinstance(result.refreshed_a2ui_part, DataPart)
    assert "planVersion: 2" in _plan_metadata_text(result.refreshed_a2ui_part)
    assert (
        "Prepare only risk-focused meeting notes."
        in _objective_text(result.refreshed_a2ui_part)
    )


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
    assert "credit_risk" in record.draft_plan.selected_agents
    assert "industry_research" not in record.draft_plan.selected_agents
    assert (
        "Additional instruction: Emphasize open follow-up items."
        in record.draft_plan.steps[0].instruction
    )


def test_approval_freezes_referenced_plan_version_and_rejects_future_mutation() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.approval_state import PlanAlreadyFinalError

    store, _original_plan = _store_with_meeting_plan()
    edit_result = store.apply_user_action(
        _action("edit_plan", {"objective": "Prepare risk-focused notes."})
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
    assert approval_result.graph_created is False
    assert approval_result.specialists_called is False
    assert record.status == "approved"
    assert record.approved_version == 2
    assert record.approved_plan is not None
    assert record.approved_plan.objective == "Prepare risk-focused notes."
    assert record.approved_plan.plan_version == 2
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
