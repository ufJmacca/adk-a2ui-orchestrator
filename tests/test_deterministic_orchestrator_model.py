from __future__ import annotations

import json

import pytest
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import FunctionTool
from google.genai import types

from orchestrator_demo.contracts import LlmIntentAssessment
from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel


DRAFT_PLAN_RESPONSE: dict[str, object] = {
    "status": "plan_required",
    "path": "plan_required",
    "planId": "plan_meeting_prep",
    "planVersion": 2,
    "approvalSurfaceId": "surface_plan_meeting_prep",
    "stepIds": ["step_relationship", "step_synthesis"],
}


@pytest.mark.asyncio
async def test_deterministic_model_routes_first_turn_business_request_exactly() -> None:
    # Arrange
    model = DeterministicOrchestratorModel()
    request_text = "Summarize the internal notes for ABC Manufacturing."
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=request_text)],
            )
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
    assert part.function_call.args == {"user_input": request_text}


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


@pytest.mark.parametrize(
    ("tool_response", "expected_text"),
    [
        pytest.param(
            {
                "status": "direct",
                "path": "direct",
                "artifacts": {
                    "final_response": {
                        "agent_id": "internal_knowledge",
                        "summary": (
                            "Synthetic relationship summary for ABC Manufacturing: "
                            "review recent call themes and open follow-up items."
                        ),
                    }
                },
            },
            (
                "Direct orchestrator response from internal_knowledge: Synthetic "
                "relationship summary for ABC Manufacturing: review recent call "
                "themes and open follow-up items."
            ),
            id="direct",
        ),
        pytest.param(
            DRAFT_PLAN_RESPONSE,
            (
                "Draft plan plan_meeting_prep v2 requires structured approval on "
                "surface_plan_meeting_prep. Step ids: step_relationship, "
                "step_synthesis. No specialist graph has executed. Use "
                "approve_orchestrator_plan with planId, approvalSurfaceId, current "
                "planVersion, and approved step ids, or reject_orchestrator_plan "
                "with a reason."
            ),
            id="plan-required",
        ),
        pytest.param(
            {
                "status": "rejected",
                "path": "rejected",
                "planId": "plan_meeting_prep",
                "planVersion": 2,
                "approvalSurfaceId": "surface_plan_meeting_prep",
                "reason": "Too broad for today.",
                "graphCreated": False,
                "specialistsCalled": False,
            },
            (
                "Plan plan_meeting_prep v2 was rejected on "
                "surface_plan_meeting_prep. Reason: Too broad for today. No "
                "specialist graph executed."
            ),
            id="rejected",
        ),
    ],
)
@pytest.mark.asyncio
async def test_deterministic_model_summarizes_post_tool_eval_states_without_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
    tool_response: dict[str, object],
    expected_text: str,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=tool_response,
                    )
                ],
            )
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.function_call is None
    assert part.text == expected_text


@pytest.mark.parametrize("approval_text", ["looks good, run it", "yes", "ok", "sure"])
@pytest.mark.asyncio
async def test_deterministic_model_natural_language_approval_after_draft_asks_for_structured_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    approval_text: str,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Prepare me for a meeting with ABC Manufacturing."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=_draft_plan_response(),
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=approval_text)],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.function_call is None
    assert part.text == (
        "Natural-language approval cannot execute draft plan plan_meeting_prep. "
        "Use structured approval with planId plan_meeting_prep, approvalSurfaceId "
        "surface_plan_meeting_prep, current planVersion 2, and approved step ids: "
        "step_relationship, step_synthesis."
    )


@pytest.mark.parametrize(
    "negated_rejection_text",
    ["don't reject it, go ahead", "no need to cancel"],
)
@pytest.mark.asyncio
async def test_deterministic_model_negated_rejection_after_draft_asks_for_structured_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    negated_rejection_text: str,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Prepare me for a meeting with ABC Manufacturing."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=_draft_plan_response(),
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=negated_rejection_text)],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.function_call is None
    assert part.text == (
        "Natural-language approval cannot execute draft plan plan_meeting_prep. "
        "Use structured approval with planId plan_meeting_prep, approvalSurfaceId "
        "surface_plan_meeting_prep, current planVersion 2, and approved step ids: "
        "step_relationship, step_synthesis."
    )


@pytest.mark.parametrize(
    ("rejection_text", "expected_reason"),
    [
        ("Reject it: Too broad for today.", "Too broad for today."),
        ("don't approve this plan", "User rejected the draft plan."),
        ("do not approve it", "User rejected the draft plan."),
        ("I can't approve this plan", "User rejected the draft plan."),
        ("I cannot approve", "User rejected the draft plan."),
        ("I am unable to approve it", "User rejected the draft plan."),
        ("do not proceed", "User rejected the draft plan."),
    ],
)
@pytest.mark.asyncio
async def test_deterministic_model_explicit_rejection_after_draft_uses_current_plan_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    rejection_text: str,
    expected_reason: str,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Prepare me for a meeting with ABC Manufacturing."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=_draft_plan_response(),
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=rejection_text)],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.text is None
    assert part.function_call is not None
    assert part.function_call.name == "reject_orchestrator_plan"
    assert part.function_call.args == {
        "plan_id": "plan_meeting_prep",
        "approval_surface_id": "surface_plan_meeting_prep",
        "reason": expected_reason,
        "edited_plan_version": 2,
    }


@pytest.mark.asyncio
async def test_deterministic_model_pending_draft_routes_approved_credit_follow_up_as_new_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    follow_up = "show approved credit limits"
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Prepare me for a meeting with ABC Manufacturing."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=_draft_plan_response(),
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=follow_up)],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.text is None
    assert part.function_call is not None
    assert part.function_call.name == "submit_orchestrator_request"
    assert part.function_call.args == {"user_input": follow_up}


@pytest.mark.asyncio
async def test_deterministic_model_routes_new_request_after_rejected_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    model = DeterministicOrchestratorModel()
    next_request = "go ahead and summarize the internal notes for ABC Manufacturing."
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text="Prepare me for a meeting with ABC Manufacturing."
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=_draft_plan_response(),
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="reject_orchestrator_plan",
                        response={
                            "status": "rejected",
                            "path": "rejected",
                            "planId": "plan_meeting_prep",
                            "planVersion": 2,
                            "approvalSurfaceId": "surface_plan_meeting_prep",
                            "reason": "Too broad for today.",
                        },
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=next_request)],
            ),
        ],
        tools_dict={
            "submit_orchestrator_request": FunctionTool(
                _submit_orchestrator_request
            ),
            "approve_orchestrator_plan": FunctionTool(_approve_orchestrator_plan),
            "reject_orchestrator_plan": FunctionTool(_reject_orchestrator_plan),
        },
    )

    # Act
    response = await _single_model_response(model, request)

    # Assert
    assert response.content is not None
    part = response.content.parts[0]
    assert part.function_call is not None
    assert part.function_call.name == "submit_orchestrator_request"
    assert part.function_call.args == {"user_input": next_request}


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


def _approve_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    approved_step_ids: list[str],
    edited_plan_version: int,
) -> dict[str, object]:
    return {
        "planId": plan_id,
        "approvalSurfaceId": approval_surface_id,
        "approvedStepIds": approved_step_ids,
        "editedPlanVersion": edited_plan_version,
    }


def _reject_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    reason: str,
    edited_plan_version: int | None = None,
) -> dict[str, object]:
    return {
        "planId": plan_id,
        "approvalSurfaceId": approval_surface_id,
        "reason": reason,
        "editedPlanVersion": edited_plan_version,
    }


def _draft_plan_response() -> dict[str, object]:
    return {
        **DRAFT_PLAN_RESPONSE,
        "stepIds": list(DRAFT_PLAN_RESPONSE["stepIds"]),
    }
