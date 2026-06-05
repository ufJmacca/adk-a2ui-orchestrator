from __future__ import annotations

import json

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types

from orchestrator_demo.contracts import LlmIntentAssessment
from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel


@pytest.mark.asyncio
async def test_deterministic_model_routes_latest_user_turn_after_prior_tool_response() -> None:
    # Arrange
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text="Prepare a meeting brief.")],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response={"status": "plan_required"},
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Research this prospect for risks and opportunities."
                    )
                ],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            )
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "submit_orchestrator_request"
    assert part.function_call.args == {
        "user_input": "Research this prospect for risks and opportunities."
    }


@pytest.mark.asyncio
async def test_deterministic_classifier_matches_user_request_not_prompt_example() -> None:
    # Arrange
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Assess the business banking relationship-manager request.\n\n"
                            "Return only JSON matching:\n"
                            "{\n"
                            '  "intents": ["meeting_prep"],\n'
                            '  "confidence": 0.91,\n'
                            '  "complexity": "simple|complex",\n'
                            '  "required_agents": ["relationship_summary"],\n'
                            '  "rationale": "short reason"\n'
                            "}\n\n"
                            "User request: Research this prospect and give me risks, "
                            "opportunities, and talking points.\n"
                            'SLM suggestion: {"intent":"meeting_prep","confidence":0.82}\n'
                            'Available agents: ["relationship_summary","web_search"]'
                        )
                    )
                ],
            )
        ]
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    assessment = json.loads(response.content.parts[0].text or "{}")
    assert assessment["intents"] == [
        "prospect_research",
        "industry_research",
        "product_opportunity",
        "credit_risk",
    ]
    assert assessment["required_agents"] == [
        "web_search",
        "industry_research",
        "product_opportunity",
        "credit_risk",
        "synthesis",
    ]


@pytest.mark.asyncio
async def test_deterministic_classifier_fallback_validates_to_data_quality() -> None:
    # Arrange
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Assess the business banking relationship-manager request.\n\n"
                            "User request: Help me decide what to do next.\n"
                            'SLM suggestion: {"intent":"unknown","confidence":0.35}\n'
                            'Available agents: ["data_quality"]'
                        )
                    )
                ],
            )
        ]
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    assessment = LlmIntentAssessment.model_validate_json(
        response.content.parts[0].text or "{}"
    )
    assert assessment.intents == ["unknown"]
    assert assessment.required_agents == ["data_quality"]
    assert assessment.complexity == "complex"
    assert "ambiguous" in assessment.rationale.casefold()


async def _single_model_response(
    model: DeterministicOrchestratorModel,
    request: LlmRequest,
) -> LlmResponse:
    responses = [
        response async for response in model.generate_content_async(request)
    ]
    assert len(responses) == 1
    return responses[0]


def _submit_orchestrator_request(user_input: str) -> dict[str, str]:
    return {"userInput": user_input}
