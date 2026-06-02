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
        approval_canvas_data_parts,
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
    parts = approval_canvas_data_parts(plan, agent_descriptors=descriptors)
    validation_results = [
        validate_outbound_a2ui(part)
        for part in parts
    ]

    # Assert
    assert len(parts) == 2
    assert all(isinstance(part, DataPart) for part in parts)
    assert all(part.metadata["mimeType"] == A2UI_MIME_TYPE for part in parts)
    assert all(result.valid is True for result in validation_results)
    assert [set(part.data) for part in parts] == [
        {"version", "createSurface"},
        {"version", "updateComponents"},
    ]
    assert all(
        result.renderer_part == part
        for result, part in zip(validation_results, parts, strict=True)
    )


def test_approval_canvas_contains_required_metadata_plan_shape_and_controls() -> None:
    # Arrange
    from orchestrator_demo.a2a_support.transport import A2uiUserAction
    from orchestrator_demo.a2ui_support.schema_manager import (
        A2UI_VERSION,
        BASIC_CATALOG_ID,
    )
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas

    plan = _meeting_plan()
    descriptors = [
        _descriptor("relationship_summary", "Relationship Summary Agent"),
        _descriptor("internal_knowledge", "Internal Knowledge Agent"),
        _descriptor("industry_research", "Industry Research Agent"),
        _descriptor("synthesis", "Synthesis Agent"),
    ]

    # Act
    canvas_messages = build_approval_canvas(plan, agent_descriptors=descriptors)
    create_surface = canvas_messages[0]["createSurface"]
    update = canvas_messages[1]["updateComponents"]
    components = update["components"]
    components_by_id = {component["id"]: component for component in components}
    text_by_id = {
        component["id"]: component["text"]
        for component in components
        if component["component"] == "Text"
    }

    # Assert
    assert [message["version"] for message in canvas_messages] == [
        A2UI_VERSION,
        A2UI_VERSION,
    ]
    assert create_surface == {
        "surfaceId": "surface_plan_meeting_prep",
        "catalogId": BASIC_CATALOG_ID,
    }
    assert update["surfaceId"] == "surface_plan_meeting_prep"
    assert components[0] == {
        "component": "Column",
        "id": "root",
        "children": [
            "component_plan_meeting_prep_title",
            "component_plan_meeting_prep_metadata",
            "component_plan_meeting_prep_objective",
            "component_plan_meeting_prep_agents",
            "component_plan_meeting_prep_steps",
            "component_plan_meeting_prep_dependencies",
            "component_plan_meeting_prep_parallel_groups",
            "component_plan_meeting_prep_available_agents",
            "component_plan_meeting_prep_risk_notes",
            "component_plan_meeting_prep_controls",
        ],
    }
    assert "surfaceId: surface_plan_meeting_prep" in text_by_id[
        "component_plan_meeting_prep_metadata"
    ]
    assert "planId: plan_meeting_prep" in text_by_id[
        "component_plan_meeting_prep_metadata"
    ]
    assert "planVersion: 2" in text_by_id[
        "component_plan_meeting_prep_metadata"
    ]
    assert plan.objective in text_by_id["component_plan_meeting_prep_objective"]
    assert "relationship_summary" in text_by_id["component_plan_meeting_prep_agents"]
    assert "internal_knowledge" in text_by_id["component_plan_meeting_prep_agents"]
    assert "industry_research" in text_by_id["component_plan_meeting_prep_agents"]
    assert "synthesis" in text_by_id["component_plan_meeting_prep_agents"]
    assert "step_relationship_summary" in text_by_id[
        "component_plan_meeting_prep_steps"
    ]
    assert "step_internal_knowledge" in text_by_id[
        "component_plan_meeting_prep_steps"
    ]
    assert "step_industry_research" in text_by_id[
        "component_plan_meeting_prep_steps"
    ]
    assert "step_synthesis" in text_by_id["component_plan_meeting_prep_steps"]
    assert "step_synthesis dependsOn" in text_by_id[
        "component_plan_meeting_prep_dependencies"
    ]
    assert "parallel_meeting_context" in text_by_id[
        "component_plan_meeting_prep_parallel_groups"
    ]

    controls = [
        component
        for component in components
        if component.get("component") == "Button"
    ]
    action_types = {
        control["action"]["event"]["name"] for control in controls
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
    for control in controls:
        context = control["action"]["event"]["context"]
        user_action = A2uiUserAction.model_validate(context)
        assert user_action.surface_id == update["surfaceId"]
        assert user_action.plan_id == plan.plan_id
        assert user_action.plan_version == plan.plan_version
        assert "planId" not in context
        assert "planVersion" not in context
        assert context["payload"]["planId"] == plan.plan_id
        assert context["payload"]["planVersion"] == plan.plan_version

    edit_control = next(
        control
        for control in controls
        if control["action"]["event"]["name"] == "edit_plan"
    )
    assert edit_control["action"]["event"]["context"]["payload"]["editableFields"] == [
        "steps",
        "selectedAgents",
    ]
    control_children = components_by_id["component_plan_meeting_prep_controls"][
        "children"
    ]
    assert control_children[:4] == [
        "control_approve_plan",
        "control_reject_plan",
        "control_edit_plan",
        "control_reorder_steps",
    ]
    assert "control_remove_step" not in components_by_id
    assert "control_replace_agent" not in components_by_id
    assert "control_add_instruction" not in components_by_id

    step_ids = {step.step_id for step in plan.steps}
    for step_id in step_ids:
        replacement_path = f"/approvalEdits/{step_id}/replacementAgentId"
        instruction_path = f"/approvalEdits/{step_id}/instruction"
        remove_id = f"control_remove_step_{step_id}"
        replace_input_id = f"control_replace_agent_{step_id}_input"
        replace_id = f"control_replace_agent_{step_id}"
        instruction_input_id = f"control_add_instruction_{step_id}_input"
        instruction_id = f"control_add_instruction_{step_id}"

        assert remove_id in control_children
        assert replace_input_id in control_children
        assert replace_id in control_children
        assert instruction_input_id in control_children
        assert instruction_id in control_children

        remove_context = components_by_id[remove_id]["action"]["event"]["context"][
            "payload"
        ]
        assert remove_context["stepId"] == step_id

        replacement_input = components_by_id[replace_input_id]
        assert replacement_input["component"] == "TextField"
        assert replacement_input["value"] == {"path": replacement_path}
        replace_context = components_by_id[replace_id]["action"]["event"][
            "context"
        ]["payload"]
        assert replace_context["stepId"] == step_id
        assert replace_context["replacementAgentId"] == {"path": replacement_path}

        instruction_input = components_by_id[instruction_input_id]
        assert instruction_input["component"] == "TextField"
        assert instruction_input["value"] == {"path": instruction_path}
        instruction_context = components_by_id[instruction_id]["action"]["event"][
            "context"
        ]["payload"]
        assert instruction_context["stepId"] == step_id
        assert instruction_context["instruction"] == {"path": instruction_path}
