from __future__ import annotations

import importlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


EVALS_DIR = Path("orchestrator_demo/evals")


def _load_eval_json(repository_root: Path, filename: str) -> Any:
    return json.loads(
        (repository_root / EVALS_DIR / filename).read_text(encoding="utf-8")
    )


def _squash_whitespace(value: str) -> str:
    return " ".join(value.split())


def _readme_bash_tokens_after(readme_text: str, marker: str) -> list[str]:
    marker_index = readme_text.index(marker)
    block_start = readme_text.index("```bash", marker_index) + len("```bash")
    block_end = readme_text.index("```", block_start)
    command_text = readme_text[block_start:block_end].strip()
    normalized_command = command_text.replace("\\\n", " ")
    return shlex.split(normalized_command)


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


def _is_git_ignored(repository_root: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--quiet", path],
        cwd=repository_root,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise AssertionError(f"git check-ignore failed for {path}: {result.returncode}")


def _load_cli_root_agent(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
    agent_path: str,
) -> Any:
    monkeypatch.setenv("ORCHESTRATOR_DEMO_DETERMINISTIC_MODEL", "1")
    get_root_agent = _import_adk_eval_symbol(
        "google.adk.cli.cli_eval",
        "get_root_agent",
    )
    cli_eval_module_names = ("agent", "agent.agent")
    saved_modules = {
        module_name: sys.modules.get(module_name)
        for module_name in cli_eval_module_names
    }
    for module_name in cli_eval_module_names:
        sys.modules.pop(module_name, None)

    try:
        return get_root_agent(str(repository_root / agent_path))
    finally:
        for module_name in cli_eval_module_names:
            sys.modules.pop(module_name, None)
        for module_name, module in saved_modules.items():
            if module is not None:
                sys.modules[module_name] = module


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
    documented_agent_path = "orchestrator_demo/orchestrator"

    # Act
    root_agent = _load_cli_root_agent(
        monkeypatch,
        repository_root,
        documented_agent_path,
    )

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


def test_user_simulation_json_shape_is_static_and_credential_free(
    repository_root: Path,
) -> None:
    # Arrange
    scenarios = _load_eval_json(repository_root, "conversation_scenarios.json")
    session_input = _load_eval_json(repository_root, "session_input.json")
    eval_config = _load_eval_json(repository_root, "user_sim_eval_config.json")
    generation_config = _load_eval_json(
        repository_root,
        "user_simulation_generation_config.example.json",
    )
    forbidden_live_credential_fields = {
        "api_key",
        "apiKey",
        "credentials",
        "credential_file",
        "service_account",
        "token",
    }

    # Act
    scenario_items = scenarios["scenarios"]
    config_text = json.dumps(
        {
            "session_input": session_input,
            "eval_config": eval_config,
            "generation_config": generation_config,
        },
        sort_keys=True,
    )

    # Assert
    assert len(scenario_items) >= 5
    assert all(
        set(scenario) == {"startingPrompt", "conversationPlan", "userPersona"}
        for scenario in scenario_items
    )
    assert session_input == {
        "appName": "orchestrator_demo",
        "userId": "synthetic_relationship_manager",
        "state": {
            "eval_mode": "user_simulation",
            "data_policy": "synthetic_only",
        },
    }
    assert "criteria" in eval_config
    assert "user_simulator_config" in eval_config
    assert generation_config["count"] == 5
    assert not any(field in config_text for field in forbidden_live_credential_fields)


def test_eval_readme_documents_loadable_user_simulation_evalset_workflow(
    monkeypatch: pytest.MonkeyPatch,
    repository_root: Path,
) -> None:
    # Arrange
    readme_path = repository_root / "orchestrator_demo" / "evals" / "README.md"

    # Act
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_flat = _squash_whitespace(readme_text)
    create_tokens = _readme_bash_tokens_after(
        readme_text,
        "Create a scenario-backed evalset:",
    )
    add_tokens = _readme_bash_tokens_after(
        readme_text,
        "Add eval cases from the checked-in scenario pack:",
    )
    eval_tokens = _readme_bash_tokens_after(
        readme_text,
        "Run the generated dynamic evalset:",
    )
    create_adk_index = create_tokens.index("adk")
    add_adk_index = add_tokens.index("adk")
    eval_adk_index = eval_tokens.index("adk")
    create_agent_path = create_tokens[create_adk_index + 3]
    create_evalset_id = create_tokens[create_adk_index + 4]
    add_agent_path = add_tokens[add_adk_index + 3]
    add_evalset_id = add_tokens[add_adk_index + 4]
    eval_agent_path = eval_tokens[eval_adk_index + 2]
    eval_evalset_id = eval_tokens[eval_adk_index + 3]
    generated_evalset_path = (
        f"{eval_agent_path}/{eval_evalset_id}.evalset.json"
    )
    root_agent = _load_cli_root_agent(
        monkeypatch,
        repository_root,
        eval_agent_path,
    )

    # Assert
    assert create_tokens[create_adk_index + 1 : create_adk_index + 3] == [
        "eval_set",
        "create",
    ]
    assert add_tokens[add_adk_index + 1 : add_adk_index + 3] == [
        "eval_set",
        "add_eval_case",
    ]
    assert eval_tokens[eval_adk_index + 1] == "eval"
    assert create_agent_path == "orchestrator_demo/orchestrator"
    assert {create_agent_path, add_agent_path, eval_agent_path} == {
        create_agent_path
    }
    assert create_evalset_id == "orchestrator_user_sim"
    assert {create_evalset_id, add_evalset_id, eval_evalset_id} == {
        create_evalset_id
    }
    assert root_agent.name == "orchestrator"
    assert "--scenarios_file" in add_tokens
    assert "orchestrator_demo/evals/conversation_scenarios.json" in add_tokens
    assert "--session_input_file" in add_tokens
    assert "orchestrator_demo/evals/session_input.json" in add_tokens
    assert "--config_file_path" in eval_tokens
    assert "orchestrator_demo/evals/user_sim_eval_config.json" in eval_tokens
    assert "--print_detailed_results" in eval_tokens
    assert _is_git_ignored(repository_root, generated_evalset_path)
    assert "generated User Simulation evalsets" in readme_flat
    assert "checked-in source fixtures" in readme_flat


def test_eval_readme_documents_user_simulation_credentials_and_metric_caveats(
    repository_root: Path,
) -> None:
    # Arrange
    readme_path = repository_root / "orchestrator_demo" / "evals" / "README.md"
    required_credential_notes = {
        "google cloud",
        "vertex",
        "project ownership",
        "api enablement",
        "quota",
        "budgets",
        "costs",
    }
    configured_metric_names = {
        "hallucinations_v1",
        "safety_v1",
        "multi_turn_task_success_v1",
        "multi_turn_trajectory_quality_v1",
        "multi_turn_tool_use_quality_v1",
        "per_turn_user_simulator_quality_v1",
    }

    # Act
    readme_text = readme_path.read_text(encoding="utf-8")
    readme_lower = readme_text.lower()

    # Assert
    assert all(note in readme_lower for note in required_credential_notes)
    assert configured_metric_names <= set(re.findall(r"[a-z0-9_]+_v1", readme_text))
    assert "google-adk==2.1.0" in readme_text
    assert "unsupported or unavailable" in readme_lower
    assert "non-blocking until validated" in readme_lower


def test_generated_user_simulation_evalsets_are_ignored_by_default(
    repository_root: Path,
) -> None:
    # Arrange
    generated_evalset_paths = {
        "orchestrator_demo/evals/generated/orchestrator_user_sim.evalset.json",
        "orchestrator_demo/evals/orchestrator_user_sim.evalset.json",
    }
    checked_in_user_sim_fixture_paths = {
        "orchestrator_demo/evals/conversation_scenarios.json",
        "orchestrator_demo/evals/session_input.json",
        "orchestrator_demo/evals/user_sim_eval_config.json",
        "orchestrator_demo/evals/user_simulation_generation_config.example.json",
    }

    # Act
    ignored_generated_paths = {
        path
        for path in generated_evalset_paths
        if _is_git_ignored(repository_root, path)
    }
    ignored_fixture_paths = {
        path
        for path in checked_in_user_sim_fixture_paths
        if _is_git_ignored(repository_root, path)
    }

    # Assert
    assert ignored_generated_paths == generated_evalset_paths
    assert ignored_fixture_paths == set()
