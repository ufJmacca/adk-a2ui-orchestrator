import importlib
from pathlib import Path
from typing import get_args

import pytest
from pydantic import ValidationError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_MODULES = [
    "orchestrator_demo",
    "orchestrator_demo.app",
    "orchestrator_demo.intent",
    "orchestrator_demo.orchestrator",
    "orchestrator_demo.registry",
    "orchestrator_demo.agents",
    "orchestrator_demo.a2ui_support",
    "orchestrator_demo.a2a_support",
    "orchestrator_demo.contracts",
]
REQUIRED_PACKAGE_DIRECTORIES = [
    "app",
    "intent",
    "orchestrator",
    "registry",
    "agents",
    "a2ui_support",
    "a2a_support",
]


def test_package_modules_import_without_required_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    # Arrange
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LLM_MODEL", raising=False)

    # Act
    imported_modules = [importlib.import_module(module) for module in PACKAGE_MODULES]

    # Assert
    assert [module.__name__ for module in imported_modules] == PACKAGE_MODULES


def test_required_package_directories_exist_with_init_files() -> None:
    # Arrange
    package_root = REPOSITORY_ROOT / "orchestrator_demo"

    # Act
    missing_packages = []
    if not package_root.is_dir() or not (package_root / "__init__.py").is_file():
        missing_packages.append("orchestrator_demo")

    for package_name in REQUIRED_PACKAGE_DIRECTORIES:
        package_directory = package_root / package_name
        if (
            not package_directory.is_dir()
            or not (package_directory / "__init__.py").is_file()
        ):
            missing_packages.append(f"orchestrator_demo.{package_name}")

    # Assert
    assert missing_packages == []


def test_core_contracts_accept_valid_business_banking_workflow() -> None:
    # Arrange
    from orchestrator_demo.contracts import (
        AgentDescriptor,
        GraphEdge,
        GraphSpec,
        GraphStep,
        IntentSuggestion,
        LlmIntentAssessment,
        PlanStep,
        RoutingDecision,
        SpecialistRequest,
        SpecialistResponse,
        StatusEvent,
        UserAction,
        ExecutionPlan,
    )

    # Act
    slm_suggestion = IntentSuggestion(intent="internal_knowledge", confidence=0.92)
    llm_assessment = LlmIntentAssessment(
        intents=["relationship_summary", "internal_knowledge"],
        confidence=0.91,
        complexity="complex",
        required_agents=["relationship_summary", "internal_knowledge", "synthesis"],
        rationale="Meeting preparation requires internal context and synthesis.",
    )
    routing_decision = RoutingDecision(
        path="plan_required",
        selected_agent=None,
        confidence=0.914,
        reason="Multiple agents are required.",
    )
    plan_step = PlanStep(
        step_id="step_internal_notes",
        agent_id="internal_knowledge",
        instruction="Summarize CRM notes for ABC Manufacturing.",
        depends_on=[],
        expected_output="A concise internal relationship context summary.",
    )
    synthesis_step = PlanStep(
        step_id="step_synthesis",
        agent_id="synthesis",
        instruction="Combine internal context into meeting preparation guidance.",
        depends_on=[plan_step.step_id],
        expected_output="A synthesized meeting preparation brief.",
    )
    execution_plan = ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare for a customer meeting.",
        detected_intents=["relationship_summary", "internal_knowledge"],
        selected_agents=["internal_knowledge", "synthesis"],
        steps=[plan_step, synthesis_step],
        data_source_categories=["internal_crm"],
        risk_notes=["Synthetic demo data only."],
    )
    descriptor = AgentDescriptor(
        agent_id="internal_knowledge",
        display_name="Internal Knowledge Agent",
        capabilities=["crm notes", "relationship records"],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=["Summarize the internal notes for ABC Manufacturing."],
        execution_mode="local_llm",
    )
    specialist_request = SpecialistRequest(
        request_id="request_internal_notes",
        user_input="Summarize the internal notes for ABC Manufacturing.",
        agent_id="internal_knowledge",
        context={"customer": "ABC Manufacturing"},
    )
    specialist_response = SpecialistResponse(
        response_id="response_internal_notes",
        agent_id="internal_knowledge",
        content="ABC Manufacturing has two open follow-ups.",
        a2ui_payload=[
            {
                "version": "v0.9",
                "createSurface": {
                    "surfaceId": "surface_internal_notes",
                    "catalogId": (
                        "https://a2ui.org/specification/v0_9/basic_catalog.json"
                    ),
                },
            }
        ],
        surface_id="surface_internal_notes",
    )
    graph = GraphSpec(
        graph_id="graph_meeting_prep",
        plan_id=execution_plan.plan_id,
        pattern="sequential",
        steps=[
            GraphStep(
                graph_step_id="graph_step_internal_notes",
                plan_step_id=plan_step.step_id,
                agent_id=plan_step.agent_id,
                depends_on=[],
            ),
            GraphStep(
                graph_step_id="graph_step_synthesis",
                plan_step_id=synthesis_step.step_id,
                agent_id=synthesis_step.agent_id,
                depends_on=["graph_step_internal_notes"],
            ),
        ],
        edges=[
            GraphEdge(
                from_step_id="graph_step_internal_notes",
                to_step_id="graph_step_synthesis",
            )
        ],
    )
    user_action = UserAction.model_validate(
        {
            "type": "approve_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "planId": execution_plan.plan_id,
            "planVersion": execution_plan.plan_version,
            "payload": {"approvedStepIds": [plan_step.step_id]},
        }
    )
    status_event = StatusEvent(
        event_id="event_step_started",
        graph_id=graph.graph_id,
        status="step_started",
        message="Internal Knowledge Agent started.",
        step_id="graph_step_internal_notes",
    )

    # Assert
    assert slm_suggestion.intent == "internal_knowledge"
    assert llm_assessment.required_agents[-1] == "synthesis"
    assert routing_decision.path == "plan_required"
    assert execution_plan.immutable_after_approval is True
    assert descriptor.execution_mode == "local_llm"
    assert specialist_request.context["customer"] == "ABC Manufacturing"
    assert specialist_response.surface_id == "surface_internal_notes"
    assert graph.pattern == "sequential"
    assert user_action.type == "approve_plan"
    assert status_event.status == "step_started"


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"intent": "web_search", "confidence": 1.01}, "confidence"),
        ({"intent": "web_search", "confidence": -0.01}, "confidence"),
        ({"intent": "unsupported", "confidence": 0.5}, "intent"),
    ],
)
def test_intent_suggestion_validates_confidence_and_allowed_intents(
    payload: dict[str, object],
    expected_error: str,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import IntentSuggestion

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        IntentSuggestion.model_validate(payload)

    assert expected_error in str(exc_info.value)


@pytest.mark.parametrize(
    ("model_name", "payload"),
    [
        (
            "LlmIntentAssessment",
            {
                "intents": ["meeting_prep"],
                "confidence": 1.01,
                "complexity": "simple",
                "required_agents": ["meeting_prep"],
                "rationale": "Out of range high confidence.",
            },
        ),
        (
            "LlmIntentAssessment",
            {
                "intents": ["meeting_prep"],
                "confidence": -0.01,
                "complexity": "simple",
                "required_agents": ["meeting_prep"],
                "rationale": "Out of range low confidence.",
            },
        ),
        (
            "RoutingDecision",
            {
                "path": "direct",
                "selected_agent": "internal_knowledge",
                "confidence": 1.01,
                "reason": "Out of range high confidence.",
            },
        ),
        (
            "RoutingDecision",
            {
                "path": "direct",
                "selected_agent": "internal_knowledge",
                "confidence": -0.01,
                "reason": "Out of range low confidence.",
            },
        ),
    ],
)
def test_llm_assessment_and_routing_decision_validate_confidence_ranges(
    model_name: str,
    payload: dict[str, object],
) -> None:
    # Arrange
    import orchestrator_demo.contracts as contracts

    model = getattr(contracts, model_name)

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        model(**payload)

    assert "confidence" in str(exc_info.value)


def test_llm_assessment_requires_at_least_one_intent() -> None:
    # Arrange
    from orchestrator_demo.contracts import LlmIntentAssessment

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        LlmIntentAssessment(
            intents=[],
            confidence=0.42,
            complexity="complex",
            required_agents=["data_quality"],
            rationale="Ambiguous request should use the unknown intent.",
        )

    assert "intents" in str(exc_info.value)


@pytest.mark.parametrize(
    ("model_name", "payload", "expected_error"),
    [
        (
            "LlmIntentAssessment",
            {
                "intents": ["meeting_prep"],
                "confidence": 0.9,
                "complexity": "multi_step",
                "required_agents": ["meeting_prep"],
                "rationale": "Invalid complexity literal.",
            },
            "complexity",
        ),
        (
            "RoutingDecision",
            {
                "path": "guess",
                "selected_agent": None,
                "confidence": 0.9,
                "reason": "Invalid path literal.",
            },
            "path",
        ),
        (
            "AgentDescriptor",
            {
                "agent_id": "internal_knowledge",
                "display_name": "Internal Knowledge Agent",
                "capabilities": ["crm notes"],
                "input_schema": {},
                "output_schema": {},
                "a2ui_catalogs": ["basic"],
                "routing_examples": ["Summarize notes."],
                "execution_mode": "serverless",
            },
            "execution_mode",
        ),
        (
            "GraphSpec",
            {
                "graph_id": "graph_invalid",
                "plan_id": "plan_invalid",
                "pattern": "mesh",
                "steps": [],
            },
            "pattern",
        ),
        (
            "StatusEvent",
            {
                "event_id": "event_invalid",
                "graph_id": "graph_meeting_prep",
                "status": "thinking",
                "message": "Invalid status literal.",
            },
            "status",
        ),
    ],
)
def test_contract_literals_reject_unknown_values(
    model_name: str,
    payload: dict[str, object],
    expected_error: str,
) -> None:
    # Arrange
    import orchestrator_demo.contracts as contracts

    model = getattr(contracts, model_name)

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        model(**payload)

    assert expected_error in str(exc_info.value)


def test_execution_plan_requires_at_least_one_step() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_empty",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["meeting_prep"],
            steps=[],
        )

    assert "steps" in str(exc_info.value)


@pytest.mark.parametrize("selected_agent", [None, "", "   "])
def test_routing_decision_direct_path_requires_selected_agent(
    selected_agent: str | None,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import RoutingDecision

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        RoutingDecision(
            path="direct",
            selected_agent=selected_agent,
            confidence=0.92,
            reason="Simple direct route.",
        )

    assert "selected_agent" in str(exc_info.value)


def test_execution_plan_rejects_dependencies_on_missing_steps() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_dangling_dependency",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["synthesis"],
            steps=[
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize unavailable research.",
                    depends_on=["step_missing"],
                    expected_output="Meeting preparation brief.",
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "declared plan steps" in error_message
    assert "step_missing" in error_message


def test_execution_plan_rejects_duplicate_step_ids() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_duplicate_step",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["internal_knowledge", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_research",
                    agent_id="internal_knowledge",
                    instruction="Summarize internal notes.",
                    expected_output="Internal context.",
                ),
                PlanStep(
                    step_id="step_research",
                    agent_id="synthesis",
                    instruction="Synthesize the meeting brief.",
                    expected_output="Meeting preparation brief.",
                ),
            ],
        )

    error_message = str(exc_info.value)
    assert "unique" in error_message
    assert "step_research" in error_message


def test_execution_plan_rejects_self_dependencies() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_self_dependency",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["synthesis"],
            steps=[
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the meeting brief.",
                    depends_on=["step_synthesis"],
                    expected_output="Meeting preparation brief.",
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "themselves" in error_message
    assert "step_synthesis" in error_message


def test_execution_plan_rejects_cyclic_dependencies() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_cyclic_dependency",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["internal_knowledge", "synthesis"],
            steps=[
                PlanStep(
                    step_id="step_internal_notes",
                    agent_id="internal_knowledge",
                    instruction="Summarize internal notes.",
                    depends_on=["step_synthesis"],
                    expected_output="Internal context.",
                ),
                PlanStep(
                    step_id="step_synthesis",
                    agent_id="synthesis",
                    instruction="Synthesize the meeting brief.",
                    depends_on=["step_internal_notes"],
                    expected_output="Meeting preparation brief.",
                ),
            ],
        )

    error_message = str(exc_info.value)
    assert "acyclic" in error_message
    assert "step_internal_notes" in error_message
    assert "step_synthesis" in error_message


def test_graph_spec_requires_at_least_one_step() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphSpec

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_empty",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[],
        )

    assert "steps" in str(exc_info.value)


def test_graph_spec_rejects_edges_to_undeclared_steps() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphEdge, GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_dangling_edge",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_internal_notes",
                    plan_step_id="step_internal_notes",
                    agent_id="internal_knowledge",
                )
            ],
            edges=[
                GraphEdge(
                    from_step_id="graph_step_internal_notes",
                    to_step_id="graph_step_missing",
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "declared graph steps" in error_message
    assert "graph_step_missing" in error_message


def test_graph_spec_rejects_depends_on_undeclared_steps() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_dangling_dependency",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_synthesis",
                    plan_step_id="step_synthesis",
                    agent_id="synthesis",
                    depends_on=["graph_step_missing"],
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "declared graph steps" in error_message
    assert "graph_step_missing" in error_message


def test_graph_spec_rejects_duplicate_graph_step_ids() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_duplicate_step",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_research",
                    plan_step_id="step_internal_notes",
                    agent_id="internal_knowledge",
                ),
                GraphStep(
                    graph_step_id="graph_step_research",
                    plan_step_id="step_synthesis",
                    agent_id="synthesis",
                ),
            ],
        )

    error_message = str(exc_info.value)
    assert "unique" in error_message
    assert "graph_step_research" in error_message


def test_graph_spec_rejects_self_loop_edges() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphEdge, GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_self_loop",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_synthesis",
                    plan_step_id="step_synthesis",
                    agent_id="synthesis",
                )
            ],
            edges=[
                GraphEdge(
                    from_step_id="graph_step_synthesis",
                    to_step_id="graph_step_synthesis",
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "themselves" in error_message
    assert "graph_step_synthesis" in error_message


def test_graph_spec_rejects_self_dependencies() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_self_dependency",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_synthesis",
                    plan_step_id="step_synthesis",
                    agent_id="synthesis",
                    depends_on=["graph_step_synthesis"],
                )
            ],
        )

    error_message = str(exc_info.value)
    assert "themselves" in error_message
    assert "graph_step_synthesis" in error_message


def test_graph_spec_rejects_cyclic_edges() -> None:
    # Arrange
    from orchestrator_demo.contracts import GraphEdge, GraphSpec, GraphStep

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        GraphSpec(
            graph_id="graph_cyclic_edges",
            plan_id="plan_meeting_prep",
            pattern="sequential",
            steps=[
                GraphStep(
                    graph_step_id="graph_step_internal_notes",
                    plan_step_id="step_internal_notes",
                    agent_id="internal_knowledge",
                ),
                GraphStep(
                    graph_step_id="graph_step_synthesis",
                    plan_step_id="step_synthesis",
                    agent_id="synthesis",
                ),
            ],
            edges=[
                GraphEdge(
                    from_step_id="graph_step_internal_notes",
                    to_step_id="graph_step_synthesis",
                ),
                GraphEdge(
                    from_step_id="graph_step_synthesis",
                    to_step_id="graph_step_internal_notes",
                ),
            ],
        )

    error_message = str(exc_info.value)
    assert "acyclic" in error_message
    assert "graph_step_internal_notes" in error_message
    assert "graph_step_synthesis" in error_message


def test_user_action_accepts_wire_format_and_specialist_action_types() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    approval_action = UserAction.model_validate(
        {
            "type": "approve_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "planId": "plan_meeting_prep",
            "planVersion": 1,
            "payload": {"approvedStepIds": ["step_internal_notes"]},
        }
    )
    specialist_action = UserAction.model_validate(
        {
            "type": "expand_relationship_card",
            "surfaceId": "surface_relationship_summary",
            "payload": {"customerId": "customer_abc"},
        }
    )

    # Assert
    assert approval_action.action_id is None
    assert approval_action.surface_id == "surface_plan_meeting_prep"
    assert approval_action.plan_id == "plan_meeting_prep"
    assert approval_action.plan_version == 1
    assert specialist_action.type == "expand_relationship_card"
    assert specialist_action.surface_id == "surface_relationship_summary"


def test_specialist_user_action_with_colliding_plan_type_remains_pass_through() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    user_action = UserAction.model_validate(
        {
            "type": "approve_plan",
            "surfaceId": "surface_relationship_summary",
            "payload": {"cardId": "relationship_overview"},
        }
    )

    # Assert
    assert user_action.type == "approve_plan"
    assert user_action.surface_id == "surface_relationship_summary"
    assert user_action.plan_id is None
    assert user_action.plan_version is None
    assert user_action.payload["cardId"] == "relationship_overview"


def test_specialist_user_action_payload_plan_id_remains_pass_through() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    user_action = UserAction.model_validate(
        {
            "type": "expand_relationship_card",
            "surfaceId": "surface_relationship_summary",
            "payload": {
                "planId": "abc123",
                "customerId": "customer_abc",
            },
        }
    )

    # Assert
    assert user_action.type == "expand_relationship_card"
    assert user_action.plan_id is None
    assert user_action.payload["planId"] == "abc123"


def test_reject_plan_user_action_accepts_payload_without_plan_version() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    user_action = UserAction.model_validate(
        {
            "type": "reject_plan",
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                "reason": "Relationship manager wants a narrower plan.",
            },
        }
    )

    # Assert
    assert user_action.type == "reject_plan"
    assert user_action.plan_id == "plan_meeting_prep"
    assert user_action.plan_version is None


def test_execution_plan_approval_surface_requires_plan_prefix() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    plan_step = PlanStep(
        step_id="step_internal_notes",
        agent_id="internal_knowledge",
        instruction="Summarize CRM notes for ABC Manufacturing.",
        expected_output="A concise internal relationship context summary.",
    )

    # Act
    execution_plan = ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare for a customer meeting.",
        detected_intents=["meeting_prep"],
        selected_agents=["internal_knowledge"],
        steps=[plan_step],
        approval_surface_id="surface_plan_meeting_prep",
    )

    # Assert
    assert execution_plan.approval_surface_id == "surface_plan_meeting_prep"


def test_execution_plan_rejects_non_plan_approval_surface() -> None:
    # Arrange
    from orchestrator_demo.contracts import ExecutionPlan, PlanStep

    plan_step = PlanStep(
        step_id="step_internal_notes",
        agent_id="internal_knowledge",
        instruction="Summarize CRM notes for ABC Manufacturing.",
        expected_output="A concise internal relationship context summary.",
    )

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        ExecutionPlan(
            plan_id="plan_meeting_prep",
            objective="Prepare for a customer meeting.",
            detected_intents=["meeting_prep"],
            selected_agents=["internal_knowledge"],
            steps=[plan_step],
            approval_surface_id="surface_approval_123",
        )

    assert "approval_surface_id" in str(exc_info.value)


def test_user_action_type_literals_cover_plan_action_types() -> None:
    # Arrange
    import orchestrator_demo.contracts as contracts

    exported_user_action_types = set(get_args(contracts.UserActionType))

    # Act / Assert
    assert contracts.PLAN_USER_ACTION_TYPES <= exported_user_action_types


@pytest.mark.parametrize(
    ("action_type", "payload"),
    [
        (
            "approve_plan",
            {
                "planId": "plan_meeting_prep",
                "planVersion": 1,
                "approvedStepIds": ["step_internal_notes"],
            },
        ),
        (
            "reject_plan",
            {
                "planId": "plan_meeting_prep",
                "reason": "Relationship manager wants a narrower plan.",
            },
        ),
        (
            "edit_plan",
            {
                "planId": "plan_meeting_prep",
                "editedPlanVersion": 2,
                "instruction": "Focus on treasury products.",
            },
        ),
    ],
)
def test_user_action_accepts_documented_user_action_envelope(
    action_type: str,
    payload: dict[str, object],
) -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    user_action = UserAction.model_validate(
        {
            "userAction": {
                "type": action_type,
                "surfaceId": "surface_plan_meeting_prep",
                "payload": payload,
            }
        }
    )

    # Assert
    assert user_action.type == action_type
    assert user_action.surface_id == "surface_plan_meeting_prep"
    assert user_action.plan_id == "plan_meeting_prep"


@pytest.mark.parametrize(
    ("action_type", "version_key"),
    [
        ("approve_plan", "editedPlanVersion"),
        ("reject_plan", "planVersion"),
        ("edit_plan", "planVersion"),
    ],
)
def test_plan_user_action_accepts_plan_identifiers_inside_payload(
    action_type: str,
    version_key: str,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act
    user_action = UserAction.model_validate(
        {
            "type": action_type,
            "surfaceId": "surface_plan_meeting_prep",
            "payload": {
                "planId": "plan_meeting_prep",
                version_key: 2,
                "approvedStepIds": ["step_internal_notes"],
            },
        }
    )

    # Assert
    assert user_action.plan_id == "plan_meeting_prep"
    assert user_action.plan_version == 2


def test_plan_user_action_requires_plan_identifiers() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {"approvedStepIds": ["step_internal_notes"]},
            }
        )

    error_message = str(exc_info.value)
    assert "planId" in error_message
    assert "planVersion" in error_message


def test_plan_user_action_rejects_conflicting_payload_plan_id() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "planId": "plan_meeting_prep",
                "planVersion": 1,
                "payload": {
                    "planId": "plan_prospect_research",
                    "approvedStepIds": ["step_internal_notes"],
                },
            }
        )

    assert "planId" in str(exc_info.value)


def test_plan_user_action_rejects_conflicting_payload_plan_id_aliases() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "approve_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "planVersion": 1,
                "payload": {
                    "planId": "plan_meeting_prep",
                    "plan_id": "plan_prospect_research",
                    "approvedStepIds": ["step_internal_notes"],
                },
            }
        )

    assert "planId" in str(exc_info.value)


def test_plan_approval_surface_rejects_unknown_user_action_type() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "approv_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "planVersion": 1,
                },
            }
        )

    assert "plan user action types" in str(exc_info.value)


@pytest.mark.parametrize("action_type", [["approve_plan"], {"name": "approve_plan"}])
def test_user_action_rejects_non_string_type_with_validation_error(
    action_type: object,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": action_type,
                "surfaceId": "surface_plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "planVersion": 1,
                },
            }
        )

    assert "type" in str(exc_info.value)


@pytest.mark.parametrize(
    "payload_version_key",
    ["planVersion", "editedPlanVersion"],
)
def test_plan_user_action_rejects_conflicting_payload_plan_version(
    payload_version_key: str,
) -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "edit_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "planId": "plan_meeting_prep",
                "planVersion": 1,
                "payload": {
                    "planId": "plan_meeting_prep",
                    payload_version_key: 2,
                    "instruction": "Focus on treasury products.",
                },
            }
        )

    assert "planVersion" in str(exc_info.value)


def test_plan_user_action_rejects_conflicting_payload_only_plan_versions() -> None:
    # Arrange
    from orchestrator_demo.contracts import UserAction

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        UserAction.model_validate(
            {
                "type": "edit_plan",
                "surfaceId": "surface_plan_meeting_prep",
                "planId": "plan_meeting_prep",
                "payload": {
                    "planId": "plan_meeting_prep",
                    "planVersion": 1,
                    "editedPlanVersion": 2,
                    "instruction": "Focus on treasury products.",
                },
            }
        )

    assert "planVersion" in str(exc_info.value)


@pytest.mark.parametrize(
    ("model_name", "payload", "expected_error"),
    [
        (
            "ExecutionPlan",
            {
                "plan_id": "meeting_prep",
                "objective": "Prepare for a customer meeting.",
                "detected_intents": ["meeting_prep"],
                "selected_agents": ["meeting_prep"],
                "steps": [],
            },
            "plan_id",
        ),
        (
            "PlanStep",
            {
                "step_id": "internal_notes",
                "agent_id": "internal_knowledge",
                "instruction": "Summarize notes.",
                "expected_output": "Summary.",
            },
            "step_id",
        ),
        (
            "UserAction",
            {
                "action_id": "approve_plan",
                "type": "approve_plan",
                "surface_id": "surface_plan",
                "payload": {},
            },
            "action_id",
        ),
        (
            "SpecialistRequest",
            {
                "request_id": "internal_notes",
                "user_input": "Summarize notes.",
                "agent_id": "internal_knowledge",
            },
            "request_id",
        ),
    ],
)
def test_contract_identifiers_require_domain_prefixes(
    model_name: str,
    payload: dict[str, object],
    expected_error: str,
) -> None:
    # Arrange
    import orchestrator_demo.contracts as contracts

    model = getattr(contracts, model_name)

    # Act / Assert
    with pytest.raises(ValidationError) as exc_info:
        model(**payload)

    assert expected_error in str(exc_info.value)
