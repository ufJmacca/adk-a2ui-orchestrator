from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest


def _import_adk_eval_symbol(module_name: str, symbol_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API is unavailable in this locked "
            f"environment: cannot import {module_name}: {exc}"
        )

    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        pytest.skip(
            "google-adk==2.1.0 eval API shape is incompatible in this locked "
            f"environment: {module_name}.{symbol_name} is missing: {exc}"
        )


def test_locked_adk_eval_symbols_import_or_skip_with_version_reason() -> None:
    # Arrange
    required_symbols = {
        "google.adk.evaluation.agent_evaluator": "AgentEvaluator",
        "google.adk.evaluation.eval_set": "EvalSet",
        "google.adk.evaluation.eval_config": "EvalConfig",
    }

    # Act
    imported_symbols = {
        symbol_name: _import_adk_eval_symbol(module_name, symbol_name)
        for module_name, symbol_name in required_symbols.items()
    }

    # Assert
    assert set(imported_symbols) == {"AgentEvaluator", "EvalSet", "EvalConfig"}
    assert all(callable(symbol) for symbol in imported_symbols.values())


def test_eval_readme_records_locked_adk_compatibility_findings(
    repository_root: Path,
) -> None:
    # Arrange
    readme_path = repository_root / "orchestrator_demo" / "evals" / "README.md"
    pyproject_text = (repository_root / "pyproject.toml").read_text()
    lock_text = (repository_root / "uv.lock").read_text()

    # Act
    readme_text = readme_path.read_text()

    # Assert
    assert '"google-adk[eval]==2.1.0"' in pyproject_text
    assert "name = \"google-adk\"" in lock_text
    assert "version = \"2.1.0\"" in lock_text
    assert (
        '{ name = "google-adk", extras = ["eval"], specifier = "==2.1.0" }'
        in lock_text
    )
    for eval_dependency in ("pandas", "rouge-score", "tabulate"):
        assert f'name = "{eval_dependency}"' in lock_text
    assert "google-adk[eval]==2.1.0" in readme_text
    assert "AgentEvaluator" in readme_text
    assert "EvalSet" in readme_text
    assert "EvalConfig" in readme_text
    assert "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1" in readme_text
    assert "ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1" in readme_text
    assert "uv run --locked pytest tests/test_adk_evalsets.py" in readme_text
    assert "adk eval" in readme_text
    assert (
        "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \\\n"
        "ORCHESTRATOR_DEMO_ADK_EVAL_MODE=1 \\\n"
        "  uv run --locked adk eval \\\n"
        "  orchestrator_demo/orchestrator \\"
    ) in readme_text
    assert "CLI compatibility" in readme_text


def test_documented_cli_agent_path_loads_under_locked_adk(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    get_root_agent = _import_adk_eval_symbol(
        "google.adk.cli.cli_eval",
        "get_root_agent",
    )
    documented_agent_path = repository_root / "orchestrator_demo" / "orchestrator"
    cli_eval_module_names = ("agent", "agent.agent")
    saved_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in cli_eval_module_names
    }
    for module_name in cli_eval_module_names:
        sys.modules.pop(module_name, None)

    try:
        # Act
        root_agent = get_root_agent(str(documented_agent_path))
    finally:
        for module_name in cli_eval_module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module

    # Assert
    assert root_agent.name == "orchestrator"


def test_basic_eval_config_uses_conservative_fixed_eval_criteria(
    repository_root: Path,
) -> None:
    # Arrange
    config_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_eval_config.json"
    )

    # Act
    config = json.loads(config_path.read_text(encoding="utf-8"))

    # Assert
    assert config == {
        "criteria": {
            "tool_trajectory_avg_score": {
                "threshold": 1.0,
                "match_type": "IN_ORDER",
            },
            "response_match_score": 0.6,
        }
    }


def test_basic_evalset_uses_locked_adk_2_1_static_conversation_shape(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    EvalSet = _import_adk_eval_symbol("google.adk.evaluation.eval_set", "EvalSet")

    # Act
    raw_evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    parsed_evalset = EvalSet.model_validate(raw_evalset)

    # Assert
    assert set(raw_evalset) == {
        "eval_set_id",
        "name",
        "description",
        "eval_cases",
    }
    assert parsed_evalset.eval_set_id == "orchestrator_basic_regression"
    assert parsed_evalset.name == "ADK A2UI Orchestrator Basic Regression"
    assert len(parsed_evalset.eval_cases) == 5

    for raw_case in raw_evalset["eval_cases"]:
        assert set(raw_case) == {
            "evalId",
            "conversation",
            "sessionInput",
        }
        assert raw_case["sessionInput"] == {
            "appName": "orchestrator",
            "userId": "synthetic_eval_rm",
            "state": {},
        }
        assert raw_case["conversation"]
        for turn in raw_case["conversation"]:
            assert set(turn) == {
                "invocationId",
                "userContent",
                "finalResponse",
                "intermediateData",
            }
            assert turn["userContent"]["role"] == "user"
            assert _single_text_part(turn["userContent"]).strip()
            assert turn["finalResponse"]["role"] == "model"
            assert _single_text_part(turn["finalResponse"]).strip()
            assert set(turn["intermediateData"]) == {"toolUses"}
            assert isinstance(turn["intermediateData"]["toolUses"], list)


def test_basic_evalset_contains_required_cases_with_exact_tool_calls(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    expected_tool_calls_by_case = {
        "direct_internal_notes_summary": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Summarize synthetic CRM internal notes for Riverbend Cafe, "
                        "focusing on relationship themes, open follow-ups, and "
                        "non-binding questions for the RM."
                    )
                },
            }
        ],
        "complex_meeting_prep_plan_required": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding briefing for my meeting with "
                        "synthetic customer ABC Manufacturing. Include relationship "
                        "context, banker notes, industry themes, and questions to ask."
                    )
                },
            }
        ],
        "complex_prospect_research_plan_required": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Research synthetic prospect Northstar Components for risks, "
                        "opportunities, and RM talking points, with safe caveats and "
                        "no binding credit decision."
                    )
                },
            }
        ],
        "natural_language_approval_does_not_execute": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding meeting plan for synthetic customer "
                        "ABC Manufacturing with relationship history, banker notes, "
                        "industry context, and final synthesis."
                    )
                },
            }
        ],
        "explicit_rejection_records_final_state": [
            {
                "name": "submit_orchestrator_request",
                "args": {
                    "user_input": (
                        "Prepare a non-binding meeting plan for synthetic customer "
                        "ABC Manufacturing with relationship history, banker notes, "
                        "industry context, and final synthesis."
                    )
                },
            },
            {
                "name": "reject_orchestrator_plan",
                "args": {
                    "plan_id": "plan_meeting_prep_e213a54b9bdf",
                    "approval_surface_id": (
                        "surface_plan_meeting_prep_e213a54b9bdf"
                    ),
                    "reason": (
                        "Synthetic RM rejected the draft pending updated customer "
                        "context."
                    ),
                    "edited_plan_version": 1,
                },
            },
        ],
    }

    # Act
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}

    # Assert
    assert set(cases_by_id) == set(expected_tool_calls_by_case)
    for case_id, expected_tool_calls in expected_tool_calls_by_case.items():
        actual_tool_calls = [
            tool_call
            for turn in cases_by_id[case_id]["conversation"]
            for tool_call in turn["intermediateData"]["toolUses"]
        ]
        assert actual_tool_calls == expected_tool_calls
        for tool_call in actual_tool_calls:
            assert set(tool_call) == {"name", "args"}
            assert isinstance(tool_call["name"], str)
            assert isinstance(tool_call["args"], dict)

    for case_id in (
        "complex_meeting_prep_plan_required",
        "natural_language_approval_does_not_execute",
        "explicit_rejection_records_final_state",
    ):
        first_prompt = _single_text_part(
            cases_by_id[case_id]["conversation"][0]["userContent"]
        )
        assert "internal notes" not in first_prompt.casefold()

    natural_language_case = cases_by_id["natural_language_approval_does_not_execute"]
    second_turn = natural_language_case["conversation"][1]
    natural_language_tool_names = [
        tool_call["name"]
        for turn in natural_language_case["conversation"]
        for tool_call in turn["intermediateData"]["toolUses"]
    ]
    assert _single_text_part(second_turn["userContent"]) == "looks good, run it"
    assert second_turn["intermediateData"]["toolUses"] == []
    assert "approve_orchestrator_plan" not in natural_language_tool_names
    assert "Natural-language approval cannot execute" in _single_text_part(
        second_turn["finalResponse"]
    )


@pytest.mark.asyncio
async def test_basic_evalset_plan_goldens_match_deterministic_eval_runtime(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    from orchestrator_demo.intent.classifier import LiteLlmIntentClassifier
    from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel
    from orchestrator_demo.orchestrator.service import OrchestratorService

    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    monkeypatch.setenv("ORCHESTRATOR_DEMO_ADK_EVAL_MODE", "1")
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}
    plan_case_ids = (
        "complex_meeting_prep_plan_required",
        "complex_prospect_research_plan_required",
        "natural_language_approval_does_not_execute",
        "explicit_rejection_records_final_state",
    )

    async def plan_for_prompt(prompt: str) -> Any:
        service = OrchestratorService(
            intent_classifier=LiteLlmIntentClassifier(
                model=DeterministicOrchestratorModel()
            )
        )
        result = await service.handle_user_request(prompt)
        assert result.approval_plan is not None
        return result.approval_plan

    # Act
    plans_by_case_id = {}
    for case_id in plan_case_ids:
        first_turn = cases_by_id[case_id]["conversation"][0]
        prompt = _single_text_part(first_turn["userContent"])
        plan = await plan_for_prompt(prompt)
        repeated_plan = await plan_for_prompt(prompt)
        plans_by_case_id[case_id] = plan

        # Assert
        assert repeated_plan.plan_id == plan.plan_id
        assert repeated_plan.approval_surface_id == plan.approval_surface_id
        expected_response = _single_text_part(first_turn["finalResponse"])
        expected_step_ids = ", ".join(step.step_id for step in plan.steps)
        assert expected_response == (
            f"Draft plan {plan.plan_id} v{plan.plan_version} requires "
            f"structured approval on {plan.approval_surface_id}. Step ids: "
            f"{expected_step_ids}. No specialist graph has executed. Use "
            "approve_orchestrator_plan with planId, approvalSurfaceId, current "
            "planVersion, and approved step ids, or reject_orchestrator_plan "
            "with a reason."
        )

    prospect_step_ids = [
        step.step_id
        for step in plans_by_case_id[
            "complex_prospect_research_plan_required"
        ].steps
    ]
    assert prospect_step_ids == [
        "step_web_search",
        "step_industry_research",
        "step_product_opportunity",
        "step_credit_risk",
        "step_synthesis",
    ]

    rejection_plan = plans_by_case_id["explicit_rejection_records_final_state"]
    rejection_turn = cases_by_id["explicit_rejection_records_final_state"][
        "conversation"
    ][1]
    rejection_call = rejection_turn["intermediateData"]["toolUses"][0]
    assert rejection_call["args"]["plan_id"] == rejection_plan.plan_id
    assert (
        rejection_call["args"]["approval_surface_id"]
        == rejection_plan.approval_surface_id
    )


@pytest.mark.asyncio
async def test_natural_language_approval_eval_case_has_runtime_negative_guardrail(
    repository_root: Path,
) -> None:
    # Arrange
    from google.adk.models.llm_request import LlmRequest
    from google.adk.tools import FunctionTool
    from google.genai import types

    from orchestrator_demo.orchestrator.agent import DeterministicOrchestratorModel

    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}
    natural_language_case = cases_by_id["natural_language_approval_does_not_execute"]
    first_turn, second_turn = natural_language_case["conversation"]
    draft_response = {
        "status": "plan_required",
        "path": "plan_required",
        "planId": "plan_meeting_prep_e213a54b9bdf",
        "planVersion": 1,
        "approvalSurfaceId": "surface_plan_meeting_prep_e213a54b9bdf",
        "stepIds": [
            "step_relationship_summary",
            "step_internal_knowledge",
            "step_industry_research",
            "step_synthesis",
        ],
    }
    request = LlmRequest(
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=_single_text_part(first_turn["userContent"])
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="submit_orchestrator_request",
                        response=draft_response,
                    )
                ],
            ),
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=_single_text_part(second_turn["userContent"])
                    )
                ],
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
    responses = [
        response
        async for response in DeterministicOrchestratorModel().generate_content_async(
            request
        )
    ]

    # Assert
    assert len(responses) == 1
    assert responses[0].content is not None
    parts = responses[0].content.parts
    assert parts is not None
    assert len(parts) == 1
    assert parts[0].function_call is None
    assert parts[0].text == _single_text_part(second_turn["finalResponse"])


def test_basic_evalset_final_responses_encode_required_case_outcomes(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )

    # Act
    evalset = json.loads(evalset_path.read_text(encoding="utf-8"))
    cases_by_id = {case["evalId"]: case for case in evalset["eval_cases"]}

    direct_response = _single_text_part(
        cases_by_id["direct_internal_notes_summary"]["conversation"][0][
            "finalResponse"
        ]
    )
    meeting_plan_response = _single_text_part(
        cases_by_id["complex_meeting_prep_plan_required"]["conversation"][0][
            "finalResponse"
        ]
    )
    prospect_plan_response = _single_text_part(
        cases_by_id["complex_prospect_research_plan_required"]["conversation"][0][
            "finalResponse"
        ]
    )
    natural_language_response = _single_text_part(
        cases_by_id["natural_language_approval_does_not_execute"]["conversation"][1][
            "finalResponse"
        ]
    )
    rejection_response = _single_text_part(
        cases_by_id["explicit_rejection_records_final_state"]["conversation"][1][
            "finalResponse"
        ]
    )

    # Assert
    assert "Direct orchestrator response" in direct_response
    assert "internal_knowledge" in direct_response
    assert "Internal Knowledge Agent" in direct_response
    assert "synthetic CRM notes" in direct_response
    assert "Synthetic demo data only" in direct_response

    assert "Draft plan plan_meeting_prep_c0587cbb09fb v1" in (
        meeting_plan_response
    )
    assert (
        "requires structured approval on surface_plan_meeting_prep_c0587cbb09fb"
        in meeting_plan_response
    )
    assert "step_relationship_summary" in meeting_plan_response
    assert "step_internal_knowledge" in meeting_plan_response
    assert "step_industry_research" in meeting_plan_response
    assert "step_synthesis" in meeting_plan_response
    assert "No specialist graph has executed" in meeting_plan_response
    assert "approve_orchestrator_plan" in meeting_plan_response
    assert "reject_orchestrator_plan" in meeting_plan_response

    assert "Draft plan plan_prospect_research_fee8cf1dfc0b v1" in (
        prospect_plan_response
    )
    assert (
        "requires structured approval on surface_plan_prospect_research_fee8cf1dfc0b"
        in prospect_plan_response
    )
    assert "step_web_search" in prospect_plan_response
    assert "step_industry_research" in prospect_plan_response
    assert "step_product_opportunity" in prospect_plan_response
    assert "step_credit_risk" in prospect_plan_response
    assert "step_compliance_policy" not in prospect_plan_response
    assert "step_synthesis" in prospect_plan_response
    assert "No specialist graph has executed" in prospect_plan_response
    assert "approve_orchestrator_plan" in prospect_plan_response
    assert "reject_orchestrator_plan" in prospect_plan_response

    assert (
        "Natural-language approval cannot execute draft plan "
        "plan_meeting_prep_e213a54b9bdf"
    ) in natural_language_response
    assert "Use structured approval" in natural_language_response
    assert "approvalSurfaceId surface_plan_meeting_prep_e213a54b9bdf" in (
        natural_language_response
    )
    assert "current planVersion 1" in natural_language_response
    assert "step_relationship_summary" in natural_language_response
    assert "step_internal_knowledge" in natural_language_response
    assert "step_industry_research" in natural_language_response
    assert "step_synthesis" in natural_language_response

    assert "Plan plan_meeting_prep_e213a54b9bdf v1 was rejected" in (
        rejection_response
    )
    assert "status: rejected" in rejection_response
    assert "surface_plan_meeting_prep_e213a54b9bdf" in rejection_response
    assert (
        "Reason: Synthetic RM rejected the draft pending updated customer context."
        in rejection_response
    )
    assert "No specialist graph executed" in rejection_response
    assert "no specialist execution artifacts were produced" in rejection_response


def test_basic_evalset_content_is_synthetic_and_safe(
    repository_root: Path,
) -> None:
    # Arrange
    evalset_path = (
        repository_root
        / "orchestrator_demo"
        / "evals"
        / "basic_evalset.evalset.json"
    )
    forbidden_markers = (
        "api_key",
        "authorization",
        "bearer",
        "password",
        "token",
        "sk-",
        "account number",
        "ssn",
        "social security",
        "real customer",
        "binding approval",
        "approved credit",
        "guaranteed rate",
    )

    # Act
    evalset_text = evalset_path.read_text(encoding="utf-8")
    normalized_text = evalset_text.casefold()

    # Assert
    assert "Riverbend Cafe" in evalset_text
    assert "ABC Manufacturing" in evalset_text
    assert "Northstar Components" in evalset_text
    assert "synthetic" in normalized_text
    assert "non-binding" in normalized_text
    assert "safe caveats" in normalized_text
    assert all(marker not in normalized_text for marker in forbidden_markers)


def _single_text_part(content: dict[str, Any]) -> str:
    parts = content["parts"]
    assert len(parts) == 1
    assert set(parts[0]) == {"text"}
    return parts[0]["text"]


def _submit_orchestrator_request(user_input: str) -> dict[str, Any]:
    return {"user_input": user_input}


def _approve_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    approved_step_ids: list[str],
    edited_plan_version: int,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "approval_surface_id": approval_surface_id,
        "approved_step_ids": approved_step_ids,
        "edited_plan_version": edited_plan_version,
    }


def _reject_orchestrator_plan(
    plan_id: str,
    approval_surface_id: str,
    reason: str,
    edited_plan_version: int | None = None,
) -> dict[str, Any]:
    return {
        "plan_id": plan_id,
        "approval_surface_id": approval_surface_id,
        "reason": reason,
        "edited_plan_version": edited_plan_version,
    }
