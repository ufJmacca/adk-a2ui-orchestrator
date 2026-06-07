from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest


EVALS_DIR = Path("orchestrator_demo/evals")


def _load_eval_json(repository_root: Path, filename: str) -> Any:
    return json.loads(
        (repository_root / EVALS_DIR / filename).read_text(encoding="utf-8")
    )


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
    assert '"google-adk==2.1.0"' in pyproject_text
    assert "name = \"google-adk\"" in lock_text
    assert "version = \"2.1.0\"" in lock_text
    assert "google-adk==2.1.0" in readme_text
    assert "AgentEvaluator" in readme_text
    assert "EvalSet" in readme_text
    assert "EvalConfig" in readme_text
    assert "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1" in readme_text
    assert "uv run --locked pytest tests/test_adk_evalsets.py" in readme_text
    assert "adk eval" in readme_text
    assert (
        "ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL=1 \\\n"
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


def test_user_simulation_artifacts_parse_with_locked_adk_models(
    repository_root: Path,
) -> None:
    # Arrange
    ConversationScenarios = _import_adk_eval_symbol(
        "google.adk.evaluation.conversation_scenarios",
        "ConversationScenarios",
    )
    ConversationGenerationConfig = _import_adk_eval_symbol(
        "google.adk.evaluation.conversation_scenarios",
        "ConversationGenerationConfig",
    )
    SessionInput = _import_adk_eval_symbol(
        "google.adk.evaluation.eval_case",
        "SessionInput",
    )
    EvalConfig = _import_adk_eval_symbol(
        "google.adk.evaluation.eval_config",
        "EvalConfig",
    )

    # Act
    scenarios = ConversationScenarios.model_validate(
        _load_eval_json(repository_root, "conversation_scenarios.json")
    )
    session_input = SessionInput.model_validate(
        _load_eval_json(repository_root, "session_input.json")
    )
    eval_config = EvalConfig.model_validate(
        _load_eval_json(repository_root, "user_sim_eval_config.json")
    )
    generation_config = ConversationGenerationConfig.model_validate(
        _load_eval_json(
            repository_root,
            "user_simulation_generation_config.example.json",
        )
    )

    # Assert
    assert len(scenarios.scenarios) >= 5
    assert session_input.app_name == "orchestrator_demo"
    assert session_input.user_id == "synthetic_relationship_manager"
    assert eval_config.user_simulator_config is not None
    assert (
        eval_config.user_simulator_config.model_dump()["model"]
        == "gemini-flash-latest"
    )
    assert generation_config.count == 5
    assert generation_config.model_name == "gemini-flash-latest"


def test_user_simulation_scenarios_cover_required_conversation_behaviors(
    repository_root: Path,
) -> None:
    # Arrange
    payload = _load_eval_json(repository_root, "conversation_scenarios.json")
    expected_scenario_ids = {
        "novice_meeting_prep_edit_then_approve",
        "expert_prospect_research_reorders_plan",
        "impatient_rm_rejects_plan",
        "risk_sensitive_rm_checks_caveats",
        "ambiguous_request_requires_clarification_or_safe_plan",
    }

    # Act
    scenarios = payload["scenarios"]
    actual_ids_by_scenario = {}
    for scenario in scenarios:
        assert set(scenario) == {
            "startingPrompt",
            "conversationPlan",
            "userPersona",
        }
        match = re.search(
            r"Scenario ID: ([a-z0-9_]+)",
            scenario["conversationPlan"],
        )
        assert match is not None, scenario["conversationPlan"]
        actual_ids_by_scenario[match.group(1)] = scenario

    # Assert
    assert expected_scenario_ids <= set(actual_ids_by_scenario)
    assert (
        actual_ids_by_scenario[
            "novice_meeting_prep_edit_then_approve"
        ]["userPersona"]
        == "NOVICE"
    )
    assert (
        actual_ids_by_scenario[
            "expert_prospect_research_reorders_plan"
        ]["userPersona"]
        == "EXPERT"
    )
    assert (
        actual_ids_by_scenario[
            "risk_sensitive_rm_checks_caveats"
        ]["userPersona"]
        == "EXPERT"
    )
    assert (
        actual_ids_by_scenario[
            "ambiguous_request_requires_clarification_or_safe_plan"
        ]["userPersona"]
        == "NOVICE"
    )

    impatient_persona = actual_ids_by_scenario[
        "impatient_rm_rejects_plan"
    ]["userPersona"]
    assert impatient_persona["id"] == "IMPATIENT_RM"
    assert len(impatient_persona["behaviors"]) >= 2

    scenario_plan_checks = {
        "novice_meeting_prep_edit_then_approve": [
            "request one edit",
            "structured approval",
        ],
        "expert_prospect_research_reorders_plan": [
            "reorder",
            "approval semantics",
        ],
        "impatient_rm_rejects_plan": [
            "reject",
            "does not execute",
        ],
        "risk_sensitive_rm_checks_caveats": [
            "caveats",
            "non-binding",
        ],
        "ambiguous_request_requires_clarification_or_safe_plan": [
            "underspecified",
            "clarification",
            "safe plan",
        ],
    }
    for scenario_id, required_phrases in scenario_plan_checks.items():
        plan = actual_ids_by_scenario[scenario_id]["conversationPlan"].lower()
        missing_phrases = [
            phrase for phrase in required_phrases if phrase not in plan
        ]
        assert missing_phrases == []


def test_user_sim_eval_config_uses_only_user_simulation_criteria(
    repository_root: Path,
) -> None:
    # Arrange
    config = _load_eval_json(repository_root, "user_sim_eval_config.json")
    expected_thresholds = {
        "hallucinations_v1": 0.5,
        "safety_v1": 0.8,
        "multi_turn_task_success_v1": 0.7,
        "multi_turn_trajectory_quality_v1": 0.7,
        "multi_turn_tool_use_quality_v1": 0.7,
        "per_turn_user_simulator_quality_v1": 0.7,
    }
    fixed_eval_metrics = {
        "tool_trajectory_avg_score",
        "response_match_score",
        "final_response_match_v2",
    }

    # Act
    criteria = config["criteria"]
    simulator_config = config["user_simulator_config"]

    # Assert
    assert set(criteria) == set(expected_thresholds)
    assert not fixed_eval_metrics & set(criteria)
    for metric_name, threshold in expected_thresholds.items():
        assert criteria[metric_name]["threshold"] == threshold
    assert criteria["hallucinations_v1"][
        "evaluate_intermediate_nl_responses"
    ] is True
    assert simulator_config == {
        "model": "gemini-flash-latest",
        "max_allowed_invocations": 12,
    }


def test_user_simulation_generation_config_is_synthetic_example_only(
    repository_root: Path,
) -> None:
    # Arrange
    config = _load_eval_json(
        repository_root,
        "user_simulation_generation_config.example.json",
    )

    # Act
    instruction = config["generation_instruction"]
    environment_context = config["environment_context"]

    # Assert
    assert config["count"] == 5
    assert config["model_name"] == "gemini-flash-latest"
    assert "synthetic business banking relationship-manager scenarios" in instruction
    assert "Do not include real customer data" in instruction
    assert "real secrets" in instruction
    assert "regulated decisions" in instruction
    assert "synthetic specialists" in environment_context
    assert "structured plan approval" in environment_context
