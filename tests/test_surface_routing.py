from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from orchestrator_demo.a2a_support.transport import A2UI_MIME_TYPE, DataPart
from orchestrator_demo.a2ui_support.schema_manager import (
    A2UI_VERSION,
    BASIC_CATALOG_ID,
)
from orchestrator_demo.contracts import (
    AgentDescriptor,
    ExecutionPlan,
    PlanStep,
    SpecialistResponse,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _specialist_a2ui(surface_id: str) -> list[dict[str, Any]]:
    return [
        {
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": BASIC_CATALOG_ID,
            },
        },
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Text",
                        "id": "root",
                        "text": "Treasury services fit the stated need.",
                    }
                ],
            },
        },
    ]


def _delete_surface_a2ui(surface_id: str) -> list[dict[str, Any]]:
    return [
        {
            "version": A2UI_VERSION,
            "deleteSurface": {
                "surfaceId": surface_id,
            },
        }
    ]


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


def _approval_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_meeting_prep",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=[
            "meeting_prep",
            "relationship_summary",
            "internal_knowledge",
        ],
        selected_agents=[
            "relationship_summary",
            "internal_knowledge",
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
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Create the RM-ready meeting brief.",
                depends_on=[
                    "step_relationship_summary",
                    "step_internal_knowledge",
                ],
                expected_output="Final meeting preparation brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=[
            "relationship_history",
            "internal_crm",
        ],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_meeting_prep",
        plan_version=2,
    )


def _approval_descriptors() -> list[AgentDescriptor]:
    return [
        _descriptor("relationship_summary", "Relationship Summary Agent"),
        _descriptor("internal_knowledge", "Internal Knowledge Agent"),
        _descriptor("synthesis", "Synthesis Agent"),
    ]


class RecordingSpecialistAdapter:
    def __init__(self) -> None:
        self.received_user_actions: list[Any] = []

    async def handle_user_action(self, user_action: Any) -> dict[str, str]:
        self.received_user_actions.append(user_action)
        return {"status": "handled", "agent_id": "product_opportunity"}


class SurfaceReturningSpecialistAdapter:
    def __init__(self, surface_id: str) -> None:
        self.surface_id = surface_id
        self.received_user_actions: list[Any] = []
        self.response = SpecialistResponse(
            response_id="response_product_opportunity_followup",
            agent_id="product_opportunity",
            content="Product Opportunity Agent: follow-up details.",
            structured_output={"summary": "follow-up details"},
            a2ui_payload=_specialist_a2ui(surface_id),
            surface_id=surface_id,
        )

    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return self.response


class SurfaceDeletingSpecialistAdapter:
    def __init__(self, surface_id: str) -> None:
        self.surface_id = surface_id
        self.received_user_actions: list[Any] = []
        self.response = SpecialistResponse(
            response_id="response_product_opportunity_delete_surface",
            agent_id="product_opportunity",
            content="Product Opportunity Agent: close surface.",
            structured_output={"status": "closed"},
            a2ui_payload=_delete_surface_a2ui(surface_id),
            surface_id=surface_id,
        )

    async def handle_user_action(self, user_action: Any) -> SpecialistResponse:
        self.received_user_actions.append(user_action)
        return self.response


@pytest.mark.parametrize(
    "import_script",
    [
        (
            "import orchestrator_demo.a2ui_support.event_parser\n"
            "import orchestrator_demo.orchestrator.surface_routes\n"
            "import orchestrator_demo.a2a_support.transport\n"
        ),
        (
            "import orchestrator_demo.orchestrator.surface_routes\n"
            "import orchestrator_demo.a2ui_support.event_parser\n"
            "import orchestrator_demo.a2a_support.transport\n"
        ),
        (
            "import orchestrator_demo.a2a_support.transport\n"
            "import orchestrator_demo.a2ui_support.event_parser\n"
            "import orchestrator_demo.orchestrator.surface_routes\n"
        ),
    ],
)
def test_user_action_routing_modules_import_in_fresh_interpreter(
    import_script: str,
) -> None:
    completed = subprocess.run(
        [sys.executable, "-c", import_script],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_registry_maps_surface_ids_to_orchestrator_and_specialist_owner() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()

    # Act
    registry.register_orchestrator_surface(
        "surface_plan_meeting_prep",
        plan_id="plan_meeting_prep",
    )
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )

    # Assert
    approval_owner = registry.owner_for("surface_plan_meeting_prep")
    specialist_owner = registry.owner_for("surface_product_recommendation")
    assert approval_owner is not None
    assert approval_owner.owner_type == "orchestrator"
    assert approval_owner.owner_id == "orchestrator"
    assert approval_owner.plan_id == "plan_meeting_prep"
    assert specialist_owner is not None
    assert specialist_owner.owner_type == "specialist"
    assert specialist_owner.owner_id == "product_opportunity"


def test_approval_surfaces_are_registered_as_orchestrator_owned() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()

    # Act
    owner = registry.register_approval_surface(
        surface_id="surface_plan_meeting_prep",
        plan_id="plan_meeting_prep",
    )

    # Assert
    assert owner.owner_type == "orchestrator"
    assert owner.owner_id == "orchestrator"
    assert registry.owner_for("surface_plan_meeting_prep") == owner


def test_prepare_approval_a2ui_preserves_payload_and_registers_orchestrator_owner() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_approval_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    plan = _approval_plan()
    payload = build_approval_canvas(
        plan,
        agent_descriptors=_approval_descriptors(),
    )
    original_payload = deepcopy(payload)

    # Act
    parts = prepare_approval_a2ui_for_renderer(
        payload,
        plan_id=plan.plan_id,
        surface_registry=registry,
    )

    # Assert
    assert payload == original_payload
    assert [part.data for part in parts] == original_payload
    assert all(isinstance(part, DataPart) for part in parts)
    assert all(part.metadata["mimeType"] == A2UI_MIME_TYPE for part in parts)
    surface_ids = {
        message["createSurface"]["surfaceId"]
        if "createSurface" in message
        else message["updateComponents"]["surfaceId"]
        for message in payload
    }
    assert surface_ids == {"surface_plan_meeting_prep"}
    owner = registry.owner_for("surface_plan_meeting_prep")
    assert owner is not None
    assert owner.owner_type == "orchestrator"
    assert owner.owner_id == "orchestrator"
    assert owner.plan_id == plan.plan_id


def test_validated_specialist_a2ui_is_preserved_unchanged_and_registered() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_specialist_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    payload = _specialist_a2ui("surface_product_recommendation")
    original_payload = deepcopy(payload)

    # Act
    parts = prepare_specialist_a2ui_for_renderer(
        payload,
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )

    # Assert
    assert payload == original_payload
    assert [part.data for part in parts] == original_payload
    assert all(isinstance(part, DataPart) for part in parts)
    assert all(part.metadata["mimeType"] == A2UI_MIME_TYPE for part in parts)
    owner = registry.owner_for("surface_product_recommendation")
    assert owner is not None
    assert owner.owner_type == "specialist"
    assert owner.owner_id == "product_opportunity"


def test_mixed_incremental_update_allows_registered_component_reference() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_specialist_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    surface_id = "surface_product_incremental"
    initial_payload = [
        {
            "version": A2UI_VERSION,
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": BASIC_CATALOG_ID,
            },
        },
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Column",
                        "id": "root",
                        "children": ["component_existing_summary"],
                    },
                    {
                        "component": "Text",
                        "id": "component_existing_summary",
                        "text": "Existing relationship summary.",
                    },
                ],
            },
        },
    ]
    incremental_payload = [
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Table",
                        "id": "component_metrics",
                        "columns": [{"key": "metric", "label": "Metric"}],
                        "rows": [{"metric": "Deposit growth"}],
                    },
                    {
                        "component": "Row",
                        "id": "component_actions",
                        "children": ["component_existing_summary"],
                    },
                ],
            },
        }
    ]

    # Act
    prepare_specialist_a2ui_for_renderer(
        initial_payload,
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )
    parts = prepare_specialist_a2ui_for_renderer(
        incremental_payload,
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )

    # Assert
    assert [part.data for part in parts] == incremental_payload
    owner = registry.owner_for(surface_id)
    assert owner is not None
    assert owner.owner_id == "product_opportunity"


def test_mixed_incremental_update_still_rejects_unknown_component_reference() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        RendererContractError,
        prepare_specialist_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    surface_id = "surface_product_incremental_rejected"
    prepare_specialist_a2ui_for_renderer(
        _specialist_a2ui(surface_id),
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )
    incremental_payload = [
        {
            "version": A2UI_VERSION,
            "updateComponents": {
                "surfaceId": surface_id,
                "components": [
                    {
                        "component": "Table",
                        "id": "component_metrics",
                        "columns": [{"key": "metric", "label": "Metric"}],
                        "rows": [{"metric": "Deposit growth"}],
                    },
                    {
                        "component": "Row",
                        "id": "component_actions",
                        "children": ["component_missing"],
                    },
                ],
            },
        }
    ]

    # Act / Assert
    with pytest.raises(RendererContractError, match="component_missing"):
        prepare_specialist_a2ui_for_renderer(
            incremental_payload,
            owner_agent_id="product_opportunity",
            surface_registry=registry,
        )


def test_failed_specialist_renderer_preparation_leaves_registry_unchanged() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_specialist_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import (
        SurfaceOwnershipError,
        SurfaceRouteRegistry,
    )

    registry = SurfaceRouteRegistry()
    original_owner = registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    payload = (
        _delete_surface_a2ui("surface_product_recommendation")
        + _specialist_a2ui("surface_product_recommendation_detail")
        + _specialist_a2ui("surface_plan_specialist_claim")
    )

    # Act / Assert
    with pytest.raises(SurfaceOwnershipError, match="reserved approval surface prefix"):
        prepare_specialist_a2ui_for_renderer(
            payload,
            owner_agent_id="product_opportunity",
            surface_registry=registry,
        )

    assert registry.owner_for("surface_product_recommendation") == original_owner
    assert registry.owner_for("surface_product_recommendation_detail") is None
    assert registry.owner_for("surface_plan_specialist_claim") is None


def test_failed_approval_renderer_preparation_leaves_registry_unchanged() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_approval_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import (
        SurfaceOwnershipError,
        SurfaceRouteRegistry,
    )

    registry = SurfaceRouteRegistry()
    approval_owner = registry.register_approval_surface(
        "surface_plan_meeting_prep",
        plan_id="plan_meeting_prep",
    )
    specialist_owner = registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    payload = (
        _delete_surface_a2ui("surface_plan_meeting_prep")
        + _specialist_a2ui("surface_product_recommendation")
    )

    # Act / Assert
    with pytest.raises(SurfaceOwnershipError, match="already owned"):
        prepare_approval_a2ui_for_renderer(
            payload,
            plan_id="plan_meeting_prep",
            surface_registry=registry,
        )

    assert registry.owner_for("surface_plan_meeting_prep") == approval_owner
    assert registry.owner_for("surface_product_recommendation") == specialist_owner


def test_specialist_surfaces_reject_reserved_plan_prefix() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import (
        SurfaceOwnershipError,
        SurfaceRouteRegistry,
    )

    registry = SurfaceRouteRegistry()
    approval_owner = registry.register_approval_surface(
        "surface_plan_meeting_prep",
        plan_id="plan_meeting_prep",
    )

    # Act / Assert
    with pytest.raises(SurfaceOwnershipError, match="reserved approval surface prefix"):
        registry.register_specialist_surface(
            "surface_plan_meeting_prep",
            agent_id="product_opportunity",
        )

    assert registry.owner_for("surface_plan_meeting_prep") == approval_owner
    with pytest.raises(SurfaceOwnershipError, match="reserved approval surface prefix"):
        registry.register_specialist_surface(
            "surface_plan_specialist_claim",
            agent_id="product_opportunity",
        )
    assert registry.owner_for("surface_plan_specialist_claim") is None


@pytest.mark.asyncio
async def test_specialist_delete_surface_clears_ownership() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.renderer_contract import (
        prepare_specialist_a2ui_for_renderer,
    )
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    payload = _delete_surface_a2ui("surface_product_recommendation")
    original_payload = deepcopy(payload)
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    parts = prepare_specialist_a2ui_for_renderer(
        payload,
        owner_agent_id="product_opportunity",
        surface_registry=registry,
    )
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert payload == original_payload
    assert [part.data for part in parts] == original_payload
    assert registry.owner_for("surface_product_recommendation") is None
    assert result.status == "error"
    assert result.error is not None
    assert result.error["code"] == "unknown_surface"
    assert result.error["ownerInferenceAttempted"] is False
    assert adapter.received_user_actions == []
    reused_owner = registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="internal_knowledge",
    )
    assert reused_owner.owner_id == "internal_knowledge"


@pytest.mark.asyncio
async def test_generated_specialist_a2ui_actions_are_structured_user_actions() -> None:
    # Arrange
    from orchestrator_demo.a2ui_support.event_parser import parse_user_action
    from orchestrator_demo.a2ui_support.validation import validate_outbound_a2ui
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent
    from orchestrator_demo.contracts import SpecialistRequest

    agent = ProductOpportunityAgent()
    request = SpecialistRequest(
        request_id="request_product_opportunity_a2ui_action",
        user_input="Suggest product opportunities for ABC Manufacturing.",
        agent_id="product_opportunity",
    )

    # Act
    response = await agent.handle(request)

    # Assert
    assert response.a2ui_payload is not None
    update_components = response.a2ui_payload[1]
    assert validate_outbound_a2ui(update_components).valid
    button = next(
        component
        for component in update_components["updateComponents"]["components"]
        if component.get("component") == "Button"
    )
    action = parse_user_action(button["action"])
    assert action.type == "specialist_action"
    assert action.surface_id == response.surface_id
    assert action.payload == {
        "agentId": "product_opportunity",
        "action": "show_more_detail",
        "componentId": "component_product_opportunity_details",
    }


@pytest.mark.asyncio
async def test_specialist_owned_user_action_is_forwarded_with_original_payload() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {
                "buttonId": "show_more_detail",
                "filters": ["treasury", "merchant_services"],
            },
        }
    }
    original_user_action = deepcopy(user_action)

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.owner is not None
    assert result.owner.owner_id == "product_opportunity"
    assert result.response == {"status": "handled", "agent_id": "product_opportunity"}
    assert adapter.received_user_actions == [user_action]
    assert adapter.received_user_actions[0] is user_action
    assert user_action == original_user_action


@pytest.mark.asyncio
async def test_wrapped_specialist_user_action_key_value_payload_routes() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": [
                {"key": "buttonId", "value": "show_more_detail"},
                {
                    "key": "filters",
                    "value": ["treasury", "merchant_services"],
                },
            ],
        }
    }
    original_user_action = deepcopy(user_action)

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.owner is not None
    assert result.owner.owner_id == "product_opportunity"
    assert adapter.received_user_actions == [user_action]
    assert adapter.received_user_actions[0] is user_action
    assert user_action == original_user_action


@pytest.mark.asyncio
async def test_data_part_key_value_user_action_routes_to_specialist() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = RecordingSpecialistAdapter()
    user_action = DataPart(
        data={
            "event": {
                "name": "specialist_action",
                "context": {
                    "type": "specialist_action",
                    "surfaceId": "surface_product_recommendation",
                    "payload": [
                        {"key": "buttonId", "value": "show_more_detail"},
                        {
                            "key": "filters",
                            "value": ["treasury", "merchant_services"],
                        },
                    ],
                },
            }
        },
        metadata={"mimeType": A2UI_MIME_TYPE},
    )
    original_user_action_data = deepcopy(user_action.data)

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.owner is not None
    assert result.owner.owner_id == "product_opportunity"
    assert adapter.received_user_actions == [user_action]
    assert adapter.received_user_actions[0] is user_action
    assert user_action.data == original_user_action_data


@pytest.mark.asyncio
async def test_forwarded_specialist_action_registers_returned_a2ui_surfaces() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = SurfaceReturningSpecialistAdapter("surface_product_recommendation_detail")
    user_action = {
        "event": {
            "name": "specialist_action",
            "context": {
                "type": "specialist_action",
                "surfaceId": "surface_product_recommendation",
                "payload": {"buttonId": "show_more_detail"},
            },
        }
    }
    original_user_action = deepcopy(user_action)

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.response is adapter.response
    assert adapter.received_user_actions == [user_action]
    assert adapter.received_user_actions[0] is user_action
    assert user_action == original_user_action
    owner = registry.owner_for("surface_product_recommendation_detail")
    assert owner is not None
    assert owner.owner_type == "specialist"
    assert owner.owner_id == "product_opportunity"


@pytest.mark.asyncio
async def test_forwarded_specialist_response_rejects_reserved_plan_surface() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = SurfaceReturningSpecialistAdapter("surface_plan_specialist_claim")
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "error"
    assert result.error is not None
    assert result.error["code"] == "surface_registration_rejected"
    assert result.error["ownerInferenceAttempted"] is False
    assert registry.owner_for("surface_plan_specialist_claim") is None
    assert adapter.received_user_actions == [user_action]


@pytest.mark.asyncio
async def test_failed_multi_message_response_registration_leaves_registry_unchanged() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    original_owner = registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = SurfaceReturningSpecialistAdapter("surface_product_recommendation_detail")
    adapter.response = SpecialistResponse(
        response_id="response_product_opportunity_mixed_surfaces",
        agent_id="product_opportunity",
        content="Product Opportunity Agent: mixed follow-up surfaces.",
        structured_output={"summary": "mixed follow-up surfaces"},
        a2ui_payload=(
            _delete_surface_a2ui("surface_product_recommendation")
            + _specialist_a2ui("surface_product_recommendation_detail")
            + _specialist_a2ui("surface_plan_specialist_claim")
        ),
        surface_id="surface_product_recommendation_detail",
    )
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "error"
    assert result.error is not None
    assert result.error["code"] == "surface_registration_rejected"
    assert result.error["ownerInferenceAttempted"] is False
    assert registry.owner_for("surface_product_recommendation") == original_owner
    assert registry.owner_for("surface_product_recommendation_detail") is None
    assert registry.owner_for("surface_plan_specialist_claim") is None
    assert adapter.received_user_actions == [user_action]


@pytest.mark.asyncio
async def test_mixed_invalid_response_registration_leaves_registry_unchanged() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    original_owner = registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = SurfaceReturningSpecialistAdapter("surface_product_recommendation_detail")
    adapter.response = SpecialistResponse(
        response_id="response_product_opportunity_mixed_invalid_a2ui",
        agent_id="product_opportunity",
        content="Product Opportunity Agent: mixed follow-up surfaces.",
        structured_output={"summary": "mixed follow-up surfaces"},
        a2ui_payload=(
            _delete_surface_a2ui("surface_product_recommendation")
            + _specialist_a2ui("surface_product_recommendation_detail")
            + [
                {
                    "version": A2UI_VERSION,
                    "updateComponents": {
                        "surfaceId": "surface_invalid_specialist_delta",
                        "components": [],
                    },
                }
            ]
        ),
        surface_id="surface_product_recommendation_detail",
    )
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.response is adapter.response
    assert registry.owner_for("surface_product_recommendation") == original_owner
    assert registry.owner_for("surface_product_recommendation_detail") is None
    assert registry.owner_for("surface_invalid_specialist_delta") is None
    assert adapter.received_user_actions == [user_action]


@pytest.mark.asyncio
async def test_forwarded_specialist_delete_surface_clears_ownership() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    registry.register_specialist_surface(
        "surface_product_recommendation",
        agent_id="product_opportunity",
    )
    adapter = SurfaceDeletingSpecialistAdapter("surface_product_recommendation")
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "close"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )
    late_result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "forwarded"
    assert result.response is adapter.response
    assert registry.owner_for("surface_product_recommendation") is None
    assert late_result.status == "error"
    assert late_result.error is not None
    assert late_result.error["code"] == "unknown_surface"
    assert adapter.received_user_actions == [user_action]


@pytest.mark.asyncio
async def test_invalid_user_action_routing_error_redacts_secret_like_field_names() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    secret_field_name = "sk-or-v1-location-field-should-not-appear"
    registry = SurfaceRouteRegistry()
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_product_recommendation",
            "payload": {"buttonId": "show_more_detail"},
            secret_field_name: "accidentally pasted as a field name",
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "error"
    assert result.error is not None
    assert result.error["code"] == "invalid_user_action"
    assert result.error["ownerInferenceAttempted"] is False
    assert secret_field_name not in result.error["message"]
    assert "<redacted-key>" in result.error["message"]
    assert adapter.received_user_actions == []


@pytest.mark.asyncio
async def test_unknown_surface_returns_structured_error_without_owner_inference() -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    registry = SurfaceRouteRegistry()
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": "surface_unknown",
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "error"
    assert result.surface_id is None
    assert result.error is not None
    assert result.error == {
        "code": "unknown_surface",
        "surfaceId": None,
        "message": "No owner is registered for the requested A2UI surface.",
        "ownerInferenceAttempted": False,
    }
    assert result.owner is None
    assert result.response is None
    assert adapter.received_user_actions == []


@pytest.mark.asyncio
async def test_unknown_surface_routing_error_omits_secret_like_surface_id(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Arrange
    from orchestrator_demo.orchestrator.surface_routes import SurfaceRouteRegistry

    leaked_surface_id = "surface_sk-or-v1-renderer-secret-should-not-appear"
    registry = SurfaceRouteRegistry()
    adapter = RecordingSpecialistAdapter()
    user_action = {
        "userAction": {
            "type": "specialist_action",
            "surfaceId": leaked_surface_id,
            "payload": {"buttonId": "show_more_detail"},
        }
    }

    # Act
    result = await registry.route_user_action(
        user_action,
        specialist_adapters={"product_opportunity": adapter},
    )

    # Assert
    assert result.status == "error"
    assert result.surface_id is None
    assert result.error is not None
    assert result.error["code"] == "unknown_surface"
    assert result.error["surfaceId"] is None
    assert result.error["ownerInferenceAttempted"] is False
    assert leaked_surface_id not in str(result.error)
    assert "sk-or-v1-renderer-secret-should-not-appear" not in str(result.error)
    assert leaked_surface_id not in caplog.text
    assert "sk-or-v1-renderer-secret-should-not-appear" not in caplog.text
    assert adapter.received_user_actions == []
