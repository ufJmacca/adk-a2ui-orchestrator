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
        plan_version=2,
    )


def test_approval_canvas_data_part_validates_with_a2ui_mime_type() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart
    from orchestrator_demo.a2ui_support.approval_canvas import (
        approval_canvas_data_part,
    )
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui

    plan = _meeting_plan()
    descriptors = [
        _descriptor("relationship_summary", "Relationship Summary Agent"),
        _descriptor("internal_knowledge", "Internal Knowledge Agent"),
        _descriptor("industry_research", "Industry Research Agent"),
        _descriptor("synthesis", "Synthesis Agent"),
    ]

    # Act
    part = approval_canvas_data_part(plan, agent_descriptors=descriptors)
    validation_result = validate_outbound_a2ui(part)

    # Assert
    assert isinstance(part, DataPart)
    assert part.metadata["mimeType"] == A2UI_MIME_TYPE
    assert validation_result.valid is True
    assert validation_result.renderer_part == part


def test_approval_canvas_contains_required_metadata_plan_shape_and_controls() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas

    plan = _meeting_plan()
    descriptors = [
        _descriptor("relationship_summary", "Relationship Summary Agent"),
        _descriptor("internal_knowledge", "Internal Knowledge Agent"),
        _descriptor("industry_research", "Industry Research Agent"),
        _descriptor("synthesis", "Synthesis Agent"),
    ]

    # Act
    canvas = build_approval_canvas(plan, agent_descriptors=descriptors)
    workflow_component = canvas["components"][0]

    # Assert
    assert canvas["surfaceId"] == "surface_plan_meeting_prep"
    assert canvas["planId"] == "plan_meeting_prep"
    assert canvas["planVersion"] == 2
    assert workflow_component["objective"] == plan.objective
    assert [agent["agentId"] for agent in workflow_component["selectedAgents"]] == [
        "relationship_summary",
        "internal_knowledge",
        "industry_research",
        "synthesis",
    ]
    assert [step["stepId"] for step in workflow_component["steps"]] == [
        "step_relationship_summary",
        "step_internal_knowledge",
        "step_industry_research",
        "step_synthesis",
    ]
    assert workflow_component["steps"][-1]["dependsOn"] == [
        "step_relationship_summary",
        "step_internal_knowledge",
        "step_industry_research",
    ]
    assert workflow_component["parallelGroups"] == [
        {
            "groupId": "parallel_meeting_context",
            "stepIds": [
                "step_relationship_summary",
                "step_internal_knowledge",
                "step_industry_research",
            ],
        }
    ]

    action_types = {
        control["action"]["type"] for control in workflow_component["controls"]
    }
    assert {
        "approve_plan",
        "reject_plan",
        "edit_plan",
        "remove_step",
        "reorder_steps",
        "replace_agent",
        "add_instruction",
    } <= action_types
    assert all(
        control["action"]["surfaceId"] == canvas["surfaceId"]
        and control["action"]["planId"] == canvas["planId"]
        and control["action"]["planVersion"] == canvas["planVersion"]
        for control in workflow_component["controls"]
    )
