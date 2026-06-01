import importlib

import pytest

from orchestrator_demo.contracts import SpecialistRequest, SpecialistResponse


REQUIRED_SPECIALIST_MODULES = {
    "industry_research": "IndustryResearchAgent",
    "web_search": "WebSearchAgent",
    "internal_knowledge": "InternalKnowledgeAgent",
    "credit_risk": "CreditRiskAgent",
    "relationship_summary": "RelationshipSummaryAgent",
    "product_opportunity": "ProductOpportunityAgent",
    "compliance_policy": "CompliancePolicyAgent",
    "data_quality": "DataQualityAgent",
    "meeting_prep": "MeetingPrepAgent",
    "synthesis": "SynthesisAgent",
}

SENSITIVE_OR_ADVISORY_AGENT_IDS = {
    "credit_risk",
    "compliance_policy",
    "data_quality",
    "product_opportunity",
}

A2UI_SPECIALIST_AGENT_IDS = {
    "meeting_prep",
    "product_opportunity",
}


def _request_for(agent_id: str) -> SpecialistRequest:
    return SpecialistRequest(
        request_id=f"request_{agent_id}",
        user_input="Prepare synthetic business banking context for ABC Manufacturing.",
        agent_id=agent_id,
        context={"customer": "ABC Manufacturing", "data_scope": "synthetic_demo"},
    )


def _basic_a2ui_validator():
    from a2ui.basic_catalog import BasicCatalog
    from a2ui.schema.manager import A2uiSchemaManager

    catalog = A2uiSchemaManager(
        version="0.9",
        catalogs=[BasicCatalog.get_config("0.9")],
    ).get_selected_catalog()
    return catalog.validator


def test_required_specialist_modules_expose_agent_classes() -> None:
    # Arrange
    expected_modules = REQUIRED_SPECIALIST_MODULES

    # Act
    imported_classes = {}
    for module_name, class_name in expected_modules.items():
        module = importlib.import_module(f"orchestrator_demo.agents.{module_name}")
        imported_classes[module_name] = getattr(module, class_name, None)

    # Assert
    assert sorted(imported_classes) == sorted(expected_modules)
    assert all(agent_class is not None for agent_class in imported_classes.values())


@pytest.mark.asyncio
async def test_all_specialists_return_structured_synthetic_response_envelopes() -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists

    agents = build_default_specialists()

    # Act
    responses = {
        agent_id: await agent.handle(_request_for(agent_id))
        for agent_id, agent in agents.items()
    }

    # Assert
    assert set(responses) == set(REQUIRED_SPECIALIST_MODULES)
    for agent_id, response in responses.items():
        assert isinstance(response, SpecialistResponse)
        assert response.agent_id == agent_id
        assert response.content
        assert response.structured_output["provenance"]["data_classification"] == (
            "synthetic_demo"
        )
        assert response.structured_output["provenance"]["customer_data"] == (
            "synthetic_only"
        )
        assert response.structured_output["citations"] != []
        assert all(
            citation["uri"].startswith("synthetic://")
            for citation in response.structured_output["citations"]
        )
        assert "account number" not in response.content.lower()
        assert "social security" not in response.content.lower()


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", sorted(SENSITIVE_OR_ADVISORY_AGENT_IDS))
async def test_sensitive_advisory_and_missing_data_specialists_include_caveats(
    agent_id: str,
) -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists

    agent = build_default_specialists()[agent_id]

    # Act
    response = await agent.handle(_request_for(agent_id))

    # Assert
    assert response.structured_output["caveats"] != []
    assert response.structured_output["risk_controls"] == {
        "binding_decision": False,
        "requires_human_review": True,
    }
    assert "not a binding" in " ".join(
        response.structured_output["caveats"]
    ).lower()
    assert "approve" not in response.content.lower()
    assert "decline" not in response.content.lower()


def test_a2ui_specialist_parameters_match_default_emitters() -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists

    agents = build_default_specialists()

    # Act
    configured_a2ui_agent_ids = {
        agent_id
        for agent_id, agent in agents.items()
        if getattr(agent.profile, "emits_a2ui", False)
    }

    # Assert
    assert configured_a2ui_agent_ids == A2UI_SPECIALIST_AGENT_IDS


@pytest.mark.asyncio
async def test_specialist_calls_are_observable_for_exact_call_counts() -> None:
    # Arrange
    from orchestrator_demo.agents.product_opportunity import ProductOpportunityAgent

    agent = ProductOpportunityAgent()
    request = _request_for("product_opportunity")

    # Act
    first_response = await agent.handle(request)
    second_response = await agent.handle(request)

    # Assert
    assert agent.call_count == 2
    assert [call.request_id for call in agent.calls] == [
        request.request_id,
        request.request_id,
    ]
    assert first_response.response_id == second_response.response_id


@pytest.mark.asyncio
@pytest.mark.parametrize("agent_id", sorted(A2UI_SPECIALIST_AGENT_IDS))
async def test_specialist_a2ui_surfaces_use_stable_surface_ids_when_produced(
    agent_id: str,
) -> None:
    # Arrange
    from orchestrator_demo.agents import build_default_specialists

    agent = build_default_specialists()[agent_id]
    request = _request_for(agent_id)

    # Act
    first_response = await agent.handle(request)
    second_response = await agent.handle(request)

    # Assert
    assert first_response.surface_id is not None
    assert first_response.surface_id == second_response.surface_id
    assert first_response.a2ui_payload is not None
    assert second_response.a2ui_payload is not None

    first_payload = first_response.a2ui_payload
    second_payload = second_response.a2ui_payload
    assert isinstance(first_payload, list)
    assert isinstance(second_payload, list)
    assert first_payload[0]["createSurface"]["surfaceId"] == first_response.surface_id
    assert second_payload[0]["createSurface"]["surfaceId"] == second_response.surface_id
    assert first_payload[1]["updateComponents"]["surfaceId"] == (
        first_response.surface_id
    )
    assert second_payload[1]["updateComponents"]["surfaceId"] == (
        second_response.surface_id
    )
    assert first_payload[0]["createSurface"]["theme"]["agentDisplayName"] == (
        agent.display_name
    )
    assert first_payload[1]["updateComponents"]["components"][0]["id"] == "root"

    from orchestrator_demo.a2ui_support.event_parser import parse_user_action

    button_component = next(
        component
        for component in first_payload[1]["updateComponents"]["components"]
        if component["component"] == "Button"
    )
    user_action = parse_user_action(button_component["action"])
    assert user_action.type == "specialist_action"
    assert user_action.surface_id == first_response.surface_id
    assert user_action.payload == {"agentId": agent_id}
    event_action = parse_user_action(button_component["action"]["event"])
    assert event_action == user_action

    validator = _basic_a2ui_validator()
    validator.validate(first_payload)
    validator.validate(second_payload)
